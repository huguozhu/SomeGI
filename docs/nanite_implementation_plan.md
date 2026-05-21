# UE5 Nanite 技术实现方案

## 总体架构

```
CPU 离线预处理                GPU 实时渲染
─────────────────          ──────────────────
mesh → cluster 划分         cluster cull (frustum + occlusion)
  → LOD DAG 构建              → SW raster (small tris)
  → 压缩打包                  → visibility buffer
                              → material resolve
                              → composite
```

---

## Phase 1：Visibility Buffer（~2 周）

当前 GBuffer 对每个像素存 3 张 RT，改用 visibility buffer 只存一条 `uint2`（triangleID, depth）。

| 子任务 | 说明 |
|--------|------|
| 新增 visibility buffer RT | R32G32_UINT，替代 albedo/normal/emissive 3 张 RT |
| 修改 GBuffer pass | 输出 triangleID（从 push constant 或 instance buffer 拿）到 vis buffer |
| 新增 resolve compute pass | 从 vis buffer 的 triangleID → 查 3 个顶点 → 重建 barycentrics → 采样材质/法线 → 输出 albedo/normal/roughness 到 GBuffer |
| 验证 | 与当前 deferred pipeline 结果一致 |

**价值：** 为后续 cluster rendering 打基础，且显存带宽下降约 2/3。

---

## Phase 2：Cluster Builder 离线工具（~3 周）

对每个 mesh 做预处理，生成 GPU 可消费的 cluster 数据。

| 子任务 | 说明 |
|--------|------|
| cluster 划分 | METIS / 自顶向下八叉分割，每 cluster ≤ 128 三角形 |
| LOD 链构建 | 合并相邻 cluster → 边坍缩简化 → 生成 parent cluster，递归到 root |
| bounding sphere 计算 | 每 cluster 算紧密包围球（用于 frustum cull） |
| 数据打包 | cluster header（triangleOffset, vertexOffset, bsphere, lodError, parentIndex, children[]）→ SSBO |
| 写入 .nanite 文件 | 自定义二进制格式，meta + cluster DAG + packed vertices + indices |

---

## Phase 3：GPU-Driven Culling（~3 周）

用 compute shader 在 GPU 上做可见性判断，彻底消除 CPU draw call。

| 子任务 | 说明 |
|--------|------|
| Hi-Z pyramid 构建 | 从 depth buffer 逐级 downsample → mip chain，每级取 max |
| Frustum cull pass | 读 cluster bsphere，clip space 测试，标记 visible instances |
| Occlusion cull pass | 对 frustum-pass 的 cluster，bound 投到屏幕，查 Hi-Z mip level 做遮挡判断 |
| Two-pass occlusion | 先 cull 上一帧可见的 cluster（cache 命中），再 cull 上帧不可见的（保守兜底） |
| LOD 选择 | 按 screen-space error + distance 阈值选 cluster 层级，从 root DAG 向下 walk |
| Indirect dispatch | `vkCmdDispatchIndirect`，cull pass 输出 draw command buffer |

---

## Phase 4：Compute Rasterizer（~4 周）

对小三角形（Nanite 典型场景），硬件光栅器效率下降，需要用 compute 软光栅。

| 子任务 | 说明 |
|--------|------|
| 8×8 tile raster | 每个 workgroup 处理一个 tile，tile 内并行扫描 triangle |
| Bin-triangle 分配 | cull pass 产出 per-tile triangle list |
| 逐像素 triangle setup | 边函数 + 重心坐标计算 |
| depth test + atomic | 用 `interlockedMin` 写 visibility buffer（triangleID + depth） |
| 大三角形回退 HW | triangle 屏幕面积 > 阈值 → 走传统 VkCmdDrawIndexedIndirect |

---

## Phase 5：Virtual Texturing & Streaming（~4 周）

大量高精度模型需要按需流式加载，无法全量常驻显存。

| 子任务 | 说明 |
|--------|------|
| 页面表 | 固定页大小（64 KB），映射 cluster ID → GPU VA |
| 请求反馈 | cull pass 输出"请求加载但未就绪"的 cluster 列表 |
| 异步加载 | CPU 后台线程从 .nanite 文件读 cluster 数据 → staging buffer → GPU |
| PSO 切换 | SW raster 里检测未加载的 cluster，回退到 parent LOD 或 skip |
| 预算管理 | 显存使用超阈值时 evict 最远 cluster |

---

## 实用建议

- **先做 Phase 1**，vis buffer 本身就有价值（带宽优化），且为后面铺路
- **Cluster builder 可手写简化版**：对单个 mesh 用 METIS 做分割，LOD 用 meshoptimizer 的 simplifier
- **SW raster 可先跳过**：在小场景（数十万三角形）下 HW raster 完全够用，SW raster 只在百万级以上场景才占优势
- **整个方案约 15 周全职工作量**，如果业余开发建议先做到 Phase 2 就上 visibility buffer + cluster culling，已经能看到"几何面数大幅上升"的效果
