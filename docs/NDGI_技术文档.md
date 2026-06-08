# NDGI：神经动态全局光照

**技术文档 v1.0** | SomeGI 引擎 | 2026-06-08

---

## 1. 概述

NDGI（Neural Dynamic Global Illumination，神经动态全局光照）是一种基于微型神经网络的实时全局光照技术。它使用一个小型 MLP（多层感知机）作为"辐照度缓存"，替代传统的球面谐波探针网格（如 DDGI）。NDGI 通过在线训练持续学习场景中的光照分布，支持动态光源与动态场景。

核心思路源自 **Müller 等人 2021 年 SIGGRAPH 论文**"Real-time Neural Radiance Caching for Path Tracing"，但针对无 Tensor Core 的 GPU（如 Intel UHD 770）做了大幅简化：将原论文 8 层 × 64 神经元的 MLP 压缩为 **2 层 × 16 神经元**，完全使用通用 Compute Shader 实现推理与训练。

### 1.1 关键特性

| 特性 | 说明 |
|------|------|
| **微型 MLP** | 2 隐藏层 × 16 神经元，共 451 个参数，约 1.8 KB |
| **在线训练** | 每帧从探针光线追踪收集约 8000 个样本，mini-batch SGD 更新权重 |
| **EMA 平滑** | 指数移动平均（α=0.95），抑制帧间闪烁 |
| **通用 GPU** | 无需 Tensor Core，纯 Compute Shader 实现 |
| **双管线** | 同时支持前向渲染与延迟渲染 |

---

## 2. MLP 网络架构

```
输入(6) ──→ 隐藏层1(16) ──→ 隐藏层2(16) ──→ 输出(3)
               LeakyReLU         LeakyReLU         ReLU
```

### 2.1 输入特征

网络接收 6 维向量，编码表面上一点的空间位置与朝向信息：

```
输入 = [世界坐标.x, 世界坐标.y, 世界坐标.z, 法线.x, 法线.y, 法线.z]
```

这 6 个值足以让网络学会"空间中某位置、以某朝向离开的出射辐照度"。与 Müller 2021 相比省略了反照率（albedo）和深度（depth）——辐照度与材质无关，仅由光照决定。

### 2.2 权重参数

权重以 float4 打包格式存储于 Storage Buffer 中：

| 缓冲区 | float4 数量 | float 数量 | 说明 |
|--------|-------------|------------|------|
| W1 | 24 | 96（6×16）| 第 1 层权重 |
| B1 | 4 | 16 | 第 1 层偏置 |
| W2 | 64 | 256（16×16）| 第 2 层权重 |
| B2 | 4 | 16 | 第 2 层偏置 |
| W3 | 12 | 48（16×3）| 第 3 层权重 |
| B3 | 1 | 3 | 第 3 层偏置 |
| **合计** | **109** | **451** | **约 1.8 KB** |

### 2.3 前向传播公式

```
第 1 层: h1 = LeakyReLU(W1 × [世界坐标, 法线] + B1)    // 6 → 16
第 2 层: h2 = LeakyReLU(W2 × h1 + B2)                  // 16 → 16
第 3 层: 辐照度 = ReLU(W3 × h2 + B3)                   // 16 → 3 (RGB)
```

### 2.4 Float4 打包方案

为减少 Storage Buffer 声明数量，权重矩阵按 float4 打包存储。以 W1（96 个 float → 24 个 float4）为例：`W1[0]` 的第 0 个输入到第 0-3 个神经元共 4 个权重，`W1[4]` 存第 1 个输入到第 0-3 个神经元的 4 个权重，以此类推。训练时在开头做 unpack（float4 数组 → float 数组），结尾做 pack（float 数组 → float4 数组）。

---

## 3. 推理过程

推理在光照着色器中完成（`lighting.slang` / `forward_ibl.slang`），每个像素执行一次。

### 3.1 延迟渲染路径

在 `lighting.slang` 的 Compute Shader 中（`ndgiSampleIrradiance` 函数由 `import ndgi_infer` 引入）：

```hlsl
// ddgiCounts.w == 2 时启用 NDGI 路径
float3 worldPos = worldFromDepth(pix, depth);
float3 N = normalize(gNormalRough.Load(pix).xyz);
float3 ndgiIrr = ndgiSampleIrradiance(worldPos, N);  // MLP 推理
diffuse = ndgiIrr * albedo * (1 - metallic) / PI;
```

### 3.2 前向渲染路径

在 `forward_ibl.slang` 的 Fragment Shader 中（推理函数内联，避免 import 路径 binding 号冲突）：

```hlsl
float3 N = normalize(i.normal);
float3 ndgiIrr = ndgiSampleIrradiance(i.worldPos, N);  // MLP 推理
indirect = ndgiIrr * base.rgb * (1.0 - metallic) / PI;
```

### 3.3 推理性能

单次推理计算量（以 float 乘加运算 MAD 计）：

| 层 | 计算 | MAD 数 |
|----|------|--------|
| 第 1 层（6→16）| 6×16 权重 + 16 bias | 96 |
| 第 2 层（16→16）| 16×16 权重 + 16 bias | 256 |
| 第 3 层（16→3）| 16×3 权重 + 3 bias | 48 |
| **合计** | | **约 400 MAD/像素** |

在 1920×1080 分辨率下，仅有几何体覆盖的像素执行推理（通常不到 50%），实际约 4 亿次 MAD/帧。Intel UHD 770 上预计 **< 2ms**。

---

## 4. 训练过程

### 4.1 训练数据收集

每帧从 256 个探针（8×4×8 网格）发射光线收集样本：

- 每个探针发射 **32 条光线**（球面 Fibonacci 采样 + 每帧 Y 轴旋转）
- 总计：**256 × 32 = 8,192 个训练样本/帧**

**每个样本格式（9 个 float）：**
```
[世界坐标.x, 世界坐标.y, 世界坐标.z, 法线.x, 法线.y, 法线.z, 目标辐照度.r, 目标辐照度.g, 目标辐照度.b]
```

**目标辐照度计算方法：**
```
RayQuery 追踪 → 命中三角形表面 → 计算直接光照 = BRDF(太阳) + 自发光 + 环境光
```

使用 `VK_KHR_ray_query` 扩展在 Compute Shader 中实现硬件加速光线追踪。

### 4.2 训练算法

训练在专用 Compute Shader（`ndgi_train.slang`，dispatch `(1,1,1)`）中执行：

```
每帧训练循环：
  1. 从 Probe Trace 输出读取 sampleCount
  2. 将权重从 float4 数组 unpack 为 flat float 数组
  3. 迭代 [1..4] 次：
     a. 随机选 256 个样本（mini-batch）
     b. 前向传播：对每个样本预测辐照度
     c. 计算 MSE 损失：L = mean((预测值 - 目标值)²)
     d. 反向传播：通过链式法则计算每层梯度
     e. SGD 更新：W -= 学习率 × dL/dW
  4. EMA 平滑：W = α×W_旧 + (1-α)×W_新
  5. 将权重 pack 回 float4 数组写入 buffer
```

**超参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 0.01 | SGD 步长 |
| batch 大小 | 256 | 每次梯度更新使用的样本数 |
| 迭代次数 | 4 | 每帧训练的 epoch 数 |
| EMA α | 0.95 | 时间平滑系数 |

### 4.3 反向传播推导

通过链式法则逐层计算梯度：

**第 3 层（16→3，ReLU 激活）：**
```
输出梯度 = 预测误差 × ReLU'(预测值)         // ReLU': >0时=1, ≤0时=0
dL/dW3   = 输出梯度 ⊗ h2                    // 外积，3×16 = 48 个 float
dL/dB3   = 输出梯度                          // 3 个 float
dL/dh2   = W3^T · 输出梯度                  // 16 个 float
```

**第 2 层（16→16，LeakyReLU α=0.01）：**
```
delta    = dL/dh2 ⊙ LeakyReLU'(h2)          // >0时=1, ≤0时=0.01
dL/dW2   = delta ⊗ h1                        // 外积，16×16 = 256 个 float
dL/dB2   = delta                              // 16 个 float
dL/dh1   = W2^T · delta                      // 16 个 float
```

**第 1 层（6→16，LeakyReLU α=0.01）：**
```
delta    = dL/dh1 ⊙ LeakyReLU'(h1)
dL/dW1   = delta ⊗ [世界坐标, 法线]          // 外积，16×6 = 96 个 float
dL/dB1   = delta                              // 16 个 float
```

### 4.4 EMA 平滑原理

指数移动平均减少帧间的权重抖动：

```
W_ema = α × W_上一帧 + (1-α) × W_sgd
```

α=0.95 意味着新梯度仅贡献 5% 到权重更新中。高 α 值的作用：
- **平滑帧间光照变化**：减少闪烁和突变
- **过滤单帧噪声**：每帧仅有约 8000 个样本，噪声较大
- **渐进收敛**：光照在多帧中逐步逼近稳态，而非一帧突变

### 4.5 自训练与多弹射传播

NDGI 采用**自训练（Self-Training）**策略，从单弹射光线追踪数据中隐式传播多弹射光照：

```
训练目标 = 命中点直接光照 + MLP_上一帧(命中点, 命中法线)
```

| 帧数 | MLP 学到的内容 |
|------|---------------|
| 1-10 帧 | 直接光照（太阳 + 自发光表面）|
| 10-30 帧 | 1 次弹射间接光（被直接光照亮的表面向周围反射）|
| 30-60 帧 | 2 次弹射间接光（光继续弹射）|
| 60+ 帧 | 收敛到多弹射 GI 稳态 |

虽然训练数据仅包含直接光照，但 MLP 在训练中学会了将空间中不同点的辐照度关联起来。随着帧数累积，间接光照在场景中逐步传播——等效于多次弹射光路追踪，但每帧每条光线的计算成本恒定不变。

---

## 5. 探针光线追踪

### 5.1 探针布局

复用 DDGI 的探针网格系统：

```
网格：8 × 4 × 8 = 256 个探针
间距：根据场景 AABB 自动计算
  spacing = max(AABB各轴长度) × 1.05 / (每维度探针数 - 1)
```

探针均匀分布在场景包围盒中，覆盖所有可达区域。

### 5.2 光线方向采样

使用**球面 Fibonacci 序列**生成近均匀分布的球面方向：

```
φ_golden = (1+√5)/2 ≈ 1.618  （黄金比）
θ_i = 2π × frac(i × (φ_golden - 1))
cos⁡φ_i = 1 - (2i+1)/N
sin⁡φ_i = √(1-cos²φ_i)
方向 = (cos⁡θ·sin⁡φ, sin⁡θ·sin⁡φ, cos⁡φ)
```

Fibonacci 序列在球面上产生比纯随机采样更均匀的分布。每帧额外应用 Y 轴旋转扰动（`frameIndex × 1°`），避免时间上的方向偏置。

### 5.3 硬件光线查询

使用 `VK_KHR_ray_query` 在 Compute Shader 中进行硬件加速光线追踪：

```hlsl
RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
RayDesc ray;
ray.Origin = probePos + dir * 0.01;     // 轻微偏移避免自交
ray.Direction = dir;
ray.TMin = 0.001; ray.TMax = 1e6;
q.TraceRayInline(gTLAS, RAY_FLAG_NONE, 0xFF, ray);
q.Proceed();

if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    // 读取命中属性：实例ID、图元ID、重心坐标、命中距离t
    // 从三角形顶点插值法线、UV坐标
    // 根据材质属性计算表面材质参数
    // 计算直接光照（太阳 BRDF + 自发光 + 环境光）
    // atomicAdd 写入样本 buffer
}
```

---

## 6. 系统架构

### 6.1 文件结构

| 文件 | 说明 |
|------|------|
| `src/renderer/ndgi_resources.h/cpp` | MLP 权重 buffer + 样本 buffer 资源管理 |
| `src/renderer/ndgi_pass.h/cpp` | 探针追踪 + 训练 Compute Pipeline 封装 |
| `shaders/gi/ndgi/ndgi_infer.slang` | MLP 推理函数模块（被 lighting.slang import）|
| `shaders/gi/ndgi/ndgi_init.slang` | Xavier 权重初始化 Shader |
| `shaders/gi/ndgi/ndgi_train.slang` | 在线训练 Shader（反向传播 + SGD + EMA）|
| `shaders/gi/ndgi/ndgi_probe_trace.slang` | 探针光线追踪采样 Shader |
| `shaders/lighting/lighting.slang` | 延迟渲染光照 Shader（import ndgi_infer）|
| `shaders/forward/forward_ibl.slang` | 前向渲染光照 Shader（内联 NDGI 推理函数）|

### 6.2 每帧执行流程

```
管线步骤 "NDGI"（GI 阶段，Lighting 之前执行）：
  ┌──────────────────────────────────────────────────┐
  │ 1. [仅首帧] initWeights()                       │
  │    dispatch(1,1,1): Xavier 随机初始化 MLP 权重   │
  │                                                  │
  │ 2. record() → ndgi_probe_trace.slang             │
  │    dispatch(128,1,1): 8,192 线程                  │
  │    每线程追踪 1 条光线 → 命中 → 计算直接光照      │
  │    atomicAdd → 写入 sample buffer + sampleCount   │
  │                                                  │
  │ 3. recordTraining() → ndgi_train.slang            │
  │    dispatch(1,1,1): 单线程执行                    │
  │    unpack → 前向传播 → 反向传播 → SGD → EMA      │
  │    → pack 更新权重到 buffer                       │
  └──────────────────────────────────────────────────┘

管线步骤 "Lighting"（延迟）或 ForwardPass::record（前向）：
  ┌──────────────────────────────────────────────────┐
  │ 每个像素调用：                                    │
  │   ndgiSampleIrradiance(世界坐标, 法线)            │
  │   → 6→16→16→3 MLP 前向传播 → 预测辐照度         │
  │   → 辐照度 × albedo × (1-metallic) / PI          │
  └──────────────────────────────────────────────────┘
```

### 6.3 Descriptor 绑定

NDGI 权重通过 descriptor set=0 传入着色器：

| 绑定号（延迟） | 绑定号（前向） | 缓冲区 | 内容 |
|---------------|---------------|--------|------|
| 27 | 4 | `gNdgiW1`（24 float4）| 第 1 层权重（96 float）|
| 28 | 5 | `gNdgiB1`（4 float4）| 第 1 层偏置（16 float）|
| 29 | 6 | `gNdgiW2`（64 float4）| 第 2 层权重（256 float）|
| 30 | 7 | `gNdgiB2`（4 float4）| 第 2 层偏置（16 float）|
| 31 | 8 | `gNdgiW3`（12 float4）| 第 3 层权重（48 float）|
| 32 | 9 | `gNdgiB3`（1 float4）| 第 3 层偏置（3 float）|

### 6.4 训练管线的 Descriptor 绑定

训练 Compute Pipeline 使用 8 个 binding（全部为 `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`）：

| 绑定号 | 缓冲区 | 访问方式 |
|--------|--------|---------|
| 0-5 | W1, B1, W2, B2, W3, B3 | 读写（训练更新）|
| 6 | sample buffer | 只读（Probe Trace 输出）|
| 7 | sample count | 只读（原子计数器）|

---

## 7. 理论基础

### 7.1 神经辐射缓存

传统 GI 方案（如 DDGI）使用固定分辨率的探针网格 + 球面谐波（SH）存储辐照度。SH 方案的主要局限：

- **空间分辨率**受探针密度限制，光照变化剧烈处容易出现漏光
- **方向分辨率**受 SH 阶数限制，高频光照细节丢失
- **插值逻辑复杂**：三线性插值 + 背面剔除 + 切比雪夫可见性检查

神经网络作为**"通用函数逼近器"**，能以更紧凑的方式表示场景中的辐照度场：

- **连续表示**：不同于离散探针，MLP 对空间任意点给出预测
- **自适应分辨率**：网络在光照变化剧烈处自动分配更多表示容量
- **压缩效率**：451 参数 ≈ 1.8 KB，对比 DDGI 的 64×256×16B ≈ 256 KB 辐照度贴图集
- **无插值伪影**：MLP 推理产生天然平滑的输出，无需显式插值

### 7.2 DDGI 与 NDGI 对比

| 维度 | DDGI | NDGI |
|------|------|------|
| **存储** | 探针贴图集（256 KB+）| MLP 权重（1.8 KB）|
| **查询方式** | 三线性插值 + 可见性检查 | MLP 前向传播（400 MAD）|
| **训练** | 无（EMA 直接更新 texel）| SGD + 反向传播（每帧）|
| **空间表示** | 离散探针网格 | 连续函数（MLP）|
| **时间平滑** | Texel EMA（0.92）| 权重 EMA（0.95）|
| **硬件需求** | 无特殊要求 | GPU + 光线查询（Ray Query）|
| **新场景** | 即时可用 | 需约 1-2 秒训练收敛 |

---

## 8. 实现细节

### 8.1 权重初始化

使用 **Xavier（Glorot）均匀分布** 初始化：

```
W ~ Uniform(-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out)))
B = 0

第 1 层 (6→16):  scale = √(6/22) ≈ 0.522
第 2 层 (16→16): scale = √(6/32) ≈ 0.433
第 3 层 (16→3):  scale = √(6/19) ≈ 0.562
```

Xavier 初始化确保激活值的方差在各层之间保持稳定，防止训练开始时出现梯度消失/爆炸。

### 8.2 Buffer 管理细节

| 缓冲区 | 大小 | 内存类型 | 用途 |
|--------|------|---------|------|
| W1-W3, B1-B3 | ~1.8 KB | Device-local | Storage Buffer（Shader 读写）|
| 样本 buffer | 288 KB | Device-local | Probe Trace 写 → Training 读 |
| 样本计数 | 4 字节 | Host-visible | 原子计数器，CPU 读取后设 dispatch 参数 |

### 8.3 用户操作

1. 打开 ImGui 调试窗口的 **GI** 选项卡
2. 勾选 **"NDGI enabled (neural GI)"**
3. 首帧 MLP 权重 Xavier 随机初始化
4. 等待数秒训练收敛，观察 GI 效果逐渐出现

`buildPipelineTable()` 根据 `m_ndgiEnabled` 控制 NDGI 步骤启用：
```cpp
m_pipeline.setEnabled("NDGI", m_ndgiEnabled && m_rtSupported);
```

UI 开关同时触发 `ddgiCounts.w = 2`，令延迟和前向着色器均走 NDGI 推理路径。

### 8.4 渲染模式兼容性

NDGI 同时支持两种渲染模式：

| 模式 | 着色器 | 集成方式 |
|------|--------|---------|
| **延迟渲染** | `lighting.slang` | `import ndgi_infer` → `ndgiSampleIrradiance()` 替换 DDGI 漫反射 |
| **前向渲染** | `forward_ibl.slang` | 内联推理函数 → NDGI 替换 IBL 漫反射 |

---

## 9. 参考文献

1. **Müller, T., Rousselle, F., Novák, J., & Keller, A. (2021).** "Real-time Neural Radiance Caching for Path Tracing." *ACM Transactions on Graphics, 40(4)*, Article 69. [arXiv:2106.12372](https://arxiv.org/abs/2106.12372)

2. **Majercik, Z., Guertin, J.-P., Nowrouzezahrai, D., & McGuire, M. (2019).** "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields." *Journal of Computer Graphics Techniques, 8(2)*.

3. **NVIDIA RTXGI SDK.** [https://github.com/NVIDIA-RTX/RTXGI](https://github.com/NVIDIA-RTX/RTXGI)

4. **Wu, J., Zhou, J., Zhou, Z., Huang, Z., & Li, C. (2026).** "Neural Dynamic GI: Random-Access Neural Compression for Temporal Lightmaps." *CVPR 2026*. [arXiv:2604.12625](https://arxiv.org/abs/2604.12625)

---

*本文档基于 SomeGI 引擎 NDGI 实现的源代码自动生成。*
