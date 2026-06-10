# Shadow System — 阴影算法选择系统 设计文档

日期: 2026-06-10
状态: Draft, 待评审

## 1. 目标

在 SomeGI 中实现与 GI 技术选择器对标的阴影算法选择系统：通过 ImGui 下拉菜单切换不同阴影算法，所有算法输出统一的 `shadowMask`（R8_UNORM），Lighting Pass 读取后乘到直接光上。

非目标：
- 不做 per-light 独立阴影（目前只做 sun/directional shadow）
- 不做 cascade shadow map（场景规模不需要）
- 不做 point light shadow

## 2. 阴影算法

| 索引 | 名称 | 方法 | 质量/开销 |
|------|------|------|-----------|
| 0 | None | 输出全白 mask | 零开销 |
| 1 | Hard Shadow Map | 单次 sun-view 深度渲染 + 深度比较 | 低开销, 锯齿 |
| 2 | PCF Soft Shadow | 3x3 PCF 滤波 shadow map | 低开销, 软边 |
| 3 | PCSS | 基于遮挡物距离的可变 kernel PCF | 中开销, 接触硬化 |
| 4 | VSM | Variance Shadow Map + 2x2 box blur | 中开销, 高质量软阴影 |
| 5 | RT Hard | Ray Query, 1 ray/pixel, first-hit | 中开销(Vulkan), 硬边 |
| 6 | RT Soft | Ray Query, 4-16 rays/pixel + temporal accumulation | 高开销, 真软阴影 |

## 3. 架构 —— 对标 GI 模式

### 3.1 数据流

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│ ShadowPass  │───▶│  shadowMask      │───▶│ Lighting    │
│ (算法分发)  │    │  (R8_UNORM)      │    │ Pass        │
│             │    │  1.0=lit         │    │ ×directLight│
└─────────────┘    └──────────────────┘    └─────────────┘
```

### 3.2 新增文件

```
src/renderer/shadow/
├── shadow_pass.h          // 统一阴影接口 + 算法枚举
└── shadow_pass.cpp        // 算法分发 + pipeline 管理

shaders/shadow/
├── shadow_hard.slang      // Hard Shadow Map (vertex+fragment)
├── shadow_pcf.slang       // PCF Soft Shadow
├── shadow_vsm.slang       // Variance Shadow Map
├── shadow_rt_hard.slang   // RT Hard Shadow (compute)
└── shadow_rt_soft.slang   // RT Soft Shadow (compute)
```

### 3.3 对标结构

```
GI 模式                          Shadow 模式
──────────────────────────────────────────────────
kGis[] (GiEntry 数组)           kShadows[] (ShadowEntry 数组)
m_currentGiIndex                 m_currentShadowIndex
m_giIndexApplied                 m_shadowIndexApplied
applyGiSelection(idx)            applyShadowSelection(idx)
FrameRenderer::applyGiSelection  FrameRenderer::applyShadowSelection
ImGui "GI" Tab Combo             ImGui "Display" Tab Combo
```

### 3.4 核心类

```cpp
enum class ShadowMethod : int {
    None = 0,
    HardShadowMap = 1,
    PCF = 2,
    PCSS = 3,
    VSM = 4,
    RTHard = 5,
    RTSoft = 6,
};

class ShadowPass {
public:
    void init(Device& d, VkExtent2D shadowMapSize);
    void destroy();

    // 每帧录制：根据 m_method 分发到对应算法实现
    void record(VkCommandBuffer cmd, const RenderTargets& rt,
                VkBuffer frameUbo, const SceneGpu& sceneGpu,
                VkBuffer indirectBuf, uint32_t drawCount,
                const SceneRtAS* rtAS = nullptr);

    ShadowMethod method() const;
    void setMethod(ShadowMethod m);

    // 光栅化方法共享的 shadow map 资源
    Image shadowMap;      // D32_SFLOAT, 2048×2048
    Image shadowMask;     // R8_UNORM, 全分辨率

private:
    // 各算法实现
    void recordHardSM(VkCommandBuffer cmd, ...);
    void recordPCF(VkCommandBuffer cmd, ...);
    void recordPCSS(VkCommandBuffer cmd, ...);
    void recordVSM(VkCommandBuffer cmd, ...);
    void recordRTHard(VkCommandBuffer cmd, ...);
    void recordRTSoft(VkCommandBuffer cmd, ...);
};
```

### 3.5 集成点

**app.h 新增:**
```cpp
int m_currentShadowIndex = 1;    // 默认 Hard Shadow Map
int m_shadowIndexApplied = -1;
void applyShadowSelection();
```

**FrameRenderer 新增:**
```cpp
ShadowPass m_shadow;
void applyShadowSelection(int idx);
```

**LightingPass 修改:**
- 新增 `bindShadowMask(Device& d, VkImageView shadowMaskView, VkSampler sampler)`
- shader 中直接光照计算乘上 `shadowMask`

**Pipeline 集成:**
- `registerPipelineSteps()` 中新增 `"Shadow"` step（在 GBuffer/Forward 之前）
- Shadow step 根据当前选中的算法 record shadow map 或 RT shadow

## 4. 关键实现细节

### 4.1 Shadow Map（算法 1-4）

- 复用 RsmGeometryPass 的 sun-view 渲染逻辑
- Shadow map 尺寸：2048×2048 D32_SFLOAT
- sun view/proj 矩阵与 RSM 共享（已在 FrameUBO 中）
- PCF：3×3 depth compare in shader
- PCSS：第一步找平均遮挡物距离 → 第二步可变 kernel PCF
- VSM：渲染 depth + depth² → 2×2 box blur → Chebyshev shadow test

### 4.2 RT Shadow（算法 5-6）

- 使用 KHR_ray_query（与 RtGiPass 同模式）
- RT Hard: 每像素 1 条 ray, `ACCEPT_FIRST_HIT_AND_END_SEARCH`
- RT Soft: 每像素 N 条 ray, 随机采样面光源方向 + temporal accumulation
- 依赖 SceneRtAS（TLAS 已构建）
- 仅在 RT 可用时加载对应 .spv，不可用时下拉项灰掉

### 4.3 Shadow Mask

- 全分辨率 R8_UNORM: 1.0 = lit, 0.0 = fully shadowed
- 光栅化方法：shadow map → fullscreen compute → shadowMask
- RT 方法：直接 compute → shadowMask

## 5. 工作量估算

| 阶段 | 内容 | 预估 |
|------|------|------|
| ShadowPass 框架 | 类定义 + 算法分发骨架 + mask 输出 | ~200 行 |
| Shadow Map 系列 | Hard/PCF/PCSS/VSM 4 个 shader + C++ pipeline | ~400 行 |
| RT Shadow 系列 | RT Hard/Soft 2 个 shader + C++ pipeline | ~250 行 |
| LightingPass 修改 | bind shadowMask + shader 乘 mask | ~50 行 |
| Pipeline 集成 | registerPipelineSteps + FrameRenderer 集成 | ~40 行 |
| UI | ImGui Combo + app.h 状态变量 | ~30 行 |
| **总计** | | **~1000 行** |

## 6. 风险

| 风险 | 缓解 |
|------|------|
| RT shadow 在 Intel UHD 770 上不可用 | RT 算法需 `rtSupported()`；不可用时下拉灰掉显示 "(RT)" 后缀 |
| PCSS/VSM shader 复杂度 | 每算法独立 .slang 文件，不耦合 |
| Shadow map 与 RSM geometry pass 逻辑重复 | Phase 2 考虑抽取共用的 sun-view renderer |
