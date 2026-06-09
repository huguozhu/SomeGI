# SomeGI GPU-Driven Rendering 全流程

日期: 2026-06-09

## 1. 架构总览

SomeGI 实现了完整的 GPU-Driven Rendering 管线，包括 GPU Frustum Culling、Hi-Z Occlusion Culling、Indirect Draw。所有 mesh primitive 以 `DrawEntry[]` 形式提交到 GPU，culling 完全在 compute shader 中完成，最终由 `vkCmdDrawIndexedIndirect` 执行。

```
┌─────────────────────────────────────────────────────────────────────┐
│                         每帧执行序列                                   │
│                                                                     │
│  CPU                         GPU                                    │
│  ───                         ───                                    │
│  ① 构建 DrawEntry[]          ② Frustum Cull (Compute)               │
│     (per-mesh 世界AABB)           ↓                                  │
│                                 ③ Hi-Z Build (Compute)              │
│  ④ Fill UBO (VP/planes)          ↓                                  │
│                                ⑤ Cull Compute Shader                │
│                                   ├─ AABB vs 6 Frustum Planes       │
│                                   ├─ Hi-Z Occlusion Test            │
│                                   └─ InterlockedAdd → count+output   │
│                                     ↓                               │
│                                ⑥ Barrier: COMPUTE → INDIRECT        │
│                                     ↓                               │
│  ⑦ Pipeline Steps:            ⑧ vkCmdDrawIndexedIndirect            │
│     └─ GBuffer/Forward              (count=来自GPU的culled count)     │
│        └─ VS: gl_InstanceIndex       ↓                              │
│           → DrawData[index]      ⑨ Lighting → AO → GI → Post        │
│           → modelMatrix               ↓                              │
│           → materialIndex          ⑩ Present                         │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. 数据结构

### 2.1 CPU 端 (`src/scene/draw_list.h`)

```cpp
struct DrawEntry {
    glm::mat4 worldTransform;    // 世界变换矩阵
    int32_t   materialIndex;     // 材质索引
    uint32_t  firstIndex;
    uint32_t  indexCount;
    int32_t   vertexOffset;
    glm::vec3 aabbMin;           // 世界空间 AABB 最小点
    uint32_t  _pad0;
    glm::vec3 aabbMax;           // 世界空间 AABB 最大点
    uint32_t  _pad1;
};
static_assert(sizeof(DrawEntry) == 112);
```

### 2.2 GPU 端 (`shaders/common/shared_types.slang`)

```hlsl
struct IndirectDrawCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int  vertexOffset;
    uint firstInstance;    // ← 携带原始 draw index, VS 通过 SV_InstanceID 读取
};

struct DrawData {
    float4x4 modelMatrix;
    uint     materialIndex;
    int      firstIndex;
    uint     indexCount;
    int      vertexOffset;
    float3   aabbMin;
    uint     _pad0;
    float3   aabbMax;
    uint     _pad1;
};
```

### 2.3 三缓冲布局

| 缓冲 | Usage | 方向 | 用途 |
|---|---|---|---|
| `DrawDataBuffer` | `STORAGE_BUFFER` | CPU→GPU | 场景所有 draw 的完整数据（AABB、material、transform） |
| `IndirectBuf` | `STORAGE_BUFFER \| INDIRECT_BUFFER` | GPU→GPU | Cull 后的 `VkDrawIndexedIndirectCommand[]` |
| `CountBuf` | `STORAGE_BUFFER \| TRANSFER_DST` | GPU→CPU | 原子累加的存活 draw 数量 |

## 3. Phase 1: CPU 构建 Draw List

**文件**: `src/scene/draw_list.cpp`

场景加载时（`App::applySceneSelection()`），遍历 glTF node 树，为每个 mesh primitive 生成一个 `DrawEntry`：

```cpp
void buildDrawList(const SceneCpu& cpu, vector<DrawEntry>& out) {
    for (node : cpu.nodes) {
        if (node.meshIndex < 0) continue;
        for (primitive : mesh.primitives) {
            DrawEntry e;
            e.worldTransform = node.worldTransform;
            e.materialIndex   = primitive.materialIndex;
            e.firstIndex      = primitive.firstIndex;
            e.indexCount      = primitive.indexCount;
            e.vertexOffset    = primitive.vertexOffset;

            // 关键：计算世界空间 AABB（8 个角点逐一变换取 min/max）
            auto [mn, mx] = wAABB(mesh.localAabbMin, mesh.localAabbMax,
                                  node.worldTransform);
            e.aabbMin = mn;  e.aabbMax = mx;
            out.push_back(e);
        }
    }
}
```

每个 glTF primitive 对应一个 draw call。世界空间 AABB 在 CPU 端预计算，供 GPU culling 使用。

## 4. Phase 2: GPU Frustum Culling

**文件**: `src/renderer/culling/frustum_cull_pass.cpp` + `shaders/culling/frustum_cull.slang`

### 4.1 CPU 端触发

```cpp
void App::recordIndirectDraws(VkCommandBuffer cmd, uint32_t flightIdx,
                               const glm::mat4& viewProj) {
    if (m_useGpuCulling) {
        if (m_useHiZOcclusion)
            m_renderer.hizPass().record(cmd, m_renderer.rt()); // 先构建 Hi-Z

        m_renderer.cullPass().record(cmd,
            drawDataBuf,                  // binding=0: StructuredBuffer<DrawData>
            m_drawCount,                  // UBO 内传递
            m_indirectBuf,                // binding=2: RWStructuredBuffer<IndirectDrawCommand>
            m_countBuf,                   // binding=3: RWStructuredBuffer<uint> (原子累加)
            viewProj, extent, flightIdx,
            hizMip1..4);                  // binding=4-7: Hi-Z 纹理（可选）
    }
}
```

### 4.2 UBO 结构

```cpp
struct CullUbo {
    glm::mat4 viewProj;          // 用于 AABB 屏幕投影
    glm::vec4 frustum[6];        // 世界空间 6 个 frustum plane (归一化)
    glm::vec2 screenSize;
    uint32_t  drawCount;
    uint32_t  hizMaxMip;         // Hi-Z 最大 mip level (启用时=4, 否则=0)
};
```

Frustum planes 由 `extractFrustumPlanes()` 从 viewProj 矩阵提取：

```cpp
void extractFrustumPlanes(const glm::mat4& vp, glm::vec4 f[6]) {
    glm::vec4 r1=vp[0], r2=vp[1], r3=vp[2], r4=vp[3];
    f[0]=r4+r1; f[1]=r4-r1;  // left, right
    f[2]=r4+r2; f[3]=r4-r2;  // bottom, top
    f[4]=r4+r3; f[5]=r4-r3;  // near, far
    for (int i=0; i<6; ++i) {
        float L=glm::length(glm::vec3(f[i]));
        if (L>1e-10f) f[i]/=L;
    }
}
```

### 4.3 GPU Compute Shader

```hlsl
[numthreads(256, 1, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= gCull.drawCount) return;

    DrawData dd = gIn[tid.x];

    // 测试 1: AABB vs 6 Frustum Planes
    if (!aabbInFrustum(dd.aabbMin, dd.aabbMax, gCull.frustum))
        return;

    // 测试 2: Hi-Z Occlusion (仅当 hizMaxMip > 0)
    if (isOccluded(dd.aabbMin, dd.aabbMax, gCull.viewProj,
                   gCull.screenSize, gCull.hizMaxMip))
        return;

    // 通过：原子累加 + 写入 compact 输出
    uint idx;
    InterlockedAdd(gCount[0], 1, idx);
    IndirectDrawCommand cmd;
    cmd.indexCount    = dd.indexCount;
    cmd.instanceCount = 1;
    cmd.firstIndex    = dd.firstIndex;
    cmd.vertexOffset  = dd.vertexOffset;
    cmd.firstInstance = tid.x;   // 保留原始 draw index
    gOut[idx] = cmd;
}
```

**Frustum 测试** (`aabbInFrustum`): 对 6 个 plane 逐一测试 AABB 的 n-p-vertex（法向最远的角点）。若 n-p-vertex 在 plane 外侧 → AABB 完全在 frustum 外 → cull。

### 4.4 Dispatch

```cpp
vkCmdDispatch(cmd, (drawCount + 255) / 256, 1, 1);
```

每个线程处理一个 draw entry，256 线程一组。

## 5. Phase 3: Hi-Z Depth Pyramid

**文件**: `src/renderer/culling/hiz_build_pass.cpp` + `shaders/culling/hiz_build.slang`

### 5.1 构建

使用**上一帧**的 depth buffer，通过 2×2 max-reduce 构建 4 级深度金字塔：

```
Depth (上一帧, full res)
  ↓ 2×2 max → Mip1 (half)    ← 小 rect 遮挡测试
  ↓ 2×2 max → Mip2 (quarter)
  ↓ 2×2 max → Mip3 (eighth)
  ↓ 2×2 max → Mip4 (16th)    ← 大 rect 遮挡测试
```

每个 mip 存储 2×2 区域的**最大深度**（最远），确保保守的遮挡测试。

### 5.2 Compute Shader 实现

```hlsl
[numthreads(16, 16, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    uint2 coord = tid.xy;

    // Level 0→1: full → half
    if (coord.x < srcSize.x/2 && coord.y < srcSize.y/2) {
        float d00 = gDepthIn[uint2(coord.x*2,   coord.y*2)];
        float d10 = gDepthIn[uint2(coord.x*2+1, coord.y*2)];
        float d01 = gDepthIn[uint2(coord.x*2,   coord.y*2+1)];
        float d11 = gDepthIn[uint2(coord.x*2+1, coord.y*2+1)];
        gHiZMip1[coord] = max(max(d00, d10), max(d01, d11));
    }
    // Level 1→2, 2→3, 3→4 同理...
}
```

所有 4 级 mip 在**同一个 dispatch** 中完成（利用 guard 条件确保越界安全）。

### 5.3 Hi-Z Occlusion 测试

GPU cull shader 中的 `isOccluded()` 函数：

```hlsl
bool isOccluded(float3 aabbMin, float3 aabbMax, float4x4 vp,
                float2 screenSize, uint maxMip) {
    // 1. 投影 AABB 8 个角点到 clip space
    // 2. 任意角点在近平面后方 → 不遮挡（保守）
    // 3. 计算 screen-space bounding rect + 最近深度
    // 4. 根据 rect 大小自动选择 mip level:
    //    mip = max(1, log2(max(rectW, rectH)) + 1)
    // 5. 采样 rect 四角的 Hi-Z 最大深度
    // 6. 若 AABB 最近深度 > Hi-Z 最远深度 → 完全被遮挡 → cull
}
```

## 6. Phase 4: Barrier

```cpp
// Compute Shader 写入 indirect buffer 后，需要 barrier 确保可见
VkBufferMemoryBarrier2 b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
b.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
b.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
b.buffer = m_indirectBuf.handle();
vkCmdPipelineBarrier2(cmd, &di);
```

## 7. Phase 5: SPIR-V InstanceIndex 补丁

### 7.1 问题

Vulkan 中 `SV_InstanceID` = `gl_InstanceIndex - gl_BaseInstance`。Slang 编译器生成的 SPIR-V 包含 `OpLoad %gl_BaseInstance` + `OpISub` 减法。对于 indirect draw，我们需要 `firstInstance` 字段直接等于 draw data index，不经过 BaseInstance 减法的干扰。

### 7.2 解决方案

`cmake/patch_spv.py` 在 CMake POST_BUILD 阶段对目标 SPIR-V 文件执行：

1. `spirv-dis` 反汇编 → SPIR-V 文本
2. 定位 `OpLoad %gl_BaseInstance` 和 `OpISub` 指令
3. 删除：BaseInstance 的 OpLoad、OpVariable、OpDecorate
4. 将对 `%ISub_result` 的引用替换为 `%InstanceIndex_load`（直接使用 InstanceIndex）
5. `spirv-as` 重新汇编

### 7.3 补丁目标

```cmake
set(SPV_PATCH_LIST
    gbuffer/gbuffer.spv         # GBuffer VS 用 SV_InstanceID 索引 DrawData
    forward/forward.spv         # Forward VS 同上
    gi/rsm/rsm_geometry.spv     # RSM Sun-view VS 同上
    forward/forward_ibl.spv     # Forward IBL 同上
)
```

## 8. Phase 6: Vertex Shader — InstanceID → DrawData

**文件**: `shaders/gbuffer/gbuffer.slang`

```hlsl
[[vk::binding(10, 0)]] StructuredBuffer<DrawData> gDrawData;

[shader("vertex")]
VsOut vs_main(VsIn i, uint drawID : SV_InstanceID) {
    DrawData dd = gDrawData[drawID];  // InstanceID 即 draw index

    float4 wp = mul(dd.modelMatrix, float4(i.pos, 1.0));
    o.worldPos = wp.xyz;
    o.svPos    = mul(gFrame.viewProj, wp);
    o.normal   = normalize(mul((float3x3)dd.modelMatrix, i.normal));
    o.tangent  = ...;
    o.matIndex = int(dd.materialIndex);
    return o;
}
```

Vertex buffer 只存几何数据（position、normal、tangent、uv），model matrix 和 material index 从 `DrawData` buffer 按 `SV_InstanceID` 索引读取。这意味着每个 draw 拥有独立的世界变换和材质，无需 per-draw UBO push constants 切换。

## 9. Phase 7: Indirect Draw 执行

Culling 完成后，渲染 Pass 直接使用 indirect buffer 绘制：

```cpp
// GBuffer pass 内部
vkCmdDrawIndexedIndirect(cmd,
    m_indirectBuf,              // 间接命令缓冲
    0,                          // offset
    drawCount,                  // maxDrawCount (保守值)
    sizeof(VkDrawIndexedIndirectCommand));
```

或使用 count buffer 的版本：

```cpp
vkCmdDrawIndexedIndirectCount(cmd,
    m_indirectBuf,  0,
    m_countBuf,                  // GPU cull 填充的存活数量
    0, maxDrawCount,
    sizeof(VkDrawIndexedIndirectCommand));
```

**整个场景仅需 3 次 indirect call**：

| Call | 目标 | 缓冲 |
|---|---|---|
| 1 | GBuffer / Deferred Opaque | `m_indirectBuf` (主相机 cull 结果) |
| 2 | Forward / Transparent | `m_indirectBuf` (同上) |
| 3 | RSM Geometry (Sun 视角) | `m_indirectBufSun` (全量, 无 cull) |

## 10. Phase 8: CPU Fallback 路径

当 `m_useGpuCulling == false` 时，CPU 直接填充 indirect buffer：

```cpp
auto* icmds = (VkDrawIndexedIndirectCommand*)m_indirectBuf.mapped();
for (uint32_t i = 0; i < m_drawCount; ++i) {
    const auto& e = m_drawEntries[i];
    icmds[i] = {
        e.indexCount, 1,           // instanceCount = 1
        e.firstIndex, e.vertexOffset,
        i                          // firstInstance = draw index
    };
}
```

`m_indirectBufSun`（RSM sun 视角）始终走 CPU fill，不做 cull（sun 方向需要全量几何）。

## 11. Phase 9: 读取 Cull 结果（1 帧延迟）

```cpp
// 从上一帧的 count buffer 读取
uint32_t culled = *(uint32_t*)m_countBuf.mapped();
if (culled > 0 && culled <= m_drawCount)
    m_culledDrawCount = culled;
```

`m_countBuf` 使用 host-coherent 内存，CPU 无需等待 GPU fence 即可读取——代价是读取的是**上一帧**的 cull 结果，仅在 UI 展示用。

## 12. 完整帧时序

```
帧 N 开始
│
├─ vkBeginCommandBuffer
├─ Read GPU timestamps (帧 N-2)
├─ Read m_countBuf (帧 N-1 cull 结果, host-coherent, 无 stall)
├─ Write FrameUBO (CPU)
├─ Reset timestamp query pool (帧 N)
├─ Write start timestamp
│
├─ ★ recordIndirectDraws ★
│   ├─ Hi-Z Build (Compute)
│   │   └─ 读 depth(帧 N-1) → 4级 max-reduce → Mip1..4
│   ├─ GPU Frustum Cull (Compute)
│   │   ├─ 输入: DrawData[m_drawCount] + Hi-Z mips(可选)
│   │   ├─ Dispatch: ceil(drawCount/256) × 1 × 1
│   │   └─ 输出: m_indirectBuf + m_countBuf (InterlockedAdd)
│   ├─ Barrier: COMPUTE_WRITE → INDIRECT_READ
│   └─ CPU fallback (m_useGpuCulling == false)
│       └─ memset VkDrawIndexedIndirectCommand
│
├─ ★ RenderPipeline::execute ★
│   ├─ RSM Geometry     (indirect draw, sun-view depth+flux)
│   ├─ GBuffer/Forward  (indirect draw, 主相机 MRT)
│   │   └─ VS: SV_InstanceID → DrawData[i] → modelMatrix + matIndex
│   ├─ SSAO/GTAO        (compute, screen-space)
│   ├─ GI Passes        (compute: voxelize, inject, propagate, trace...)
│   ├─ Lighting         (compute: direct + GI indirect)
│   ├─ Skybox           (graphics)
│   └─ ...
│
├─ ★ recordPostProcessing ★
│   ├─ Tonemap (HDR→LDR/HDR, compute)
│   ├─ TAA/SMAA (compute, 写 swapchain image)
│   └─ hdrPrev copy (供下一帧 SSR/SSGI reproject)
│
├─ vkEndCommandBuffer
├─ vkQueueSubmit2
│   wait:  imageAvailable semaphore
│   signal: renderFinished semaphore
├─ Present
│
└─ 帧 N 结束，CPU 继续到帧 N+1
```

## 13. 设计决策总结

| 决策 | 说明 | 权衡 |
|---|---|---|
| **`firstInstance` = draw index** | indirect draw 的 `firstInstance` 携带 draw data index，VS 直接索引 | 需 SPIR-V 补丁；Slang 生成的 BaseInstance 减法要移除 |
| **Hi-Z 用上一帧 depth** | 避免 depth 写入→读取的 barrier 延迟 | 快速旋转时遮挡滞后 1 帧 |
| **Hi-Z 4 级, 单 dispatch** | 一个 compute dispatch 生成全部 4 级 mip | 代码简单，但 16×16 线程组利用率非 100%（每级只有 1/4 线程活跃） |
| **Host-coherent count 回读** | CPU 读 cull 结果无 GPU stall | 1 帧延迟，仅 UI 展示 |
| **仅 3 次 indirect call** | Opaque + Forward + RSM Sun-view | 不需要 per-material CPU 排序，GPU culling 的结果是紧凑数组 |
| **Cull 粒度 = mesh primitive** | 每个 glTF primitive 一个 DrawEntry | Sponza ~25 个 primitives 太少；大场景才看到 cull 收益 |
| **RSM Sun 不做 cull** | Sun 方向需要全量的 worldPos + normal + flux | 若场景极大可加 cascade RSM |
| **SPIR-V 文本反汇编→patch→汇编** | 用 Vulkan SDK 的 `spirv-dis`/`spirv-as` 做后处理 | 比改 Slang 编译器或手写 GLSL 更实用 |
| **间接 buffer 同时有 STORAGE + INDIRECT** | Cull compute shader 写入，draw call 读取 | 必须加 barrier |

## 14. 涉及文件索引

### C++ 源文件

| 文件 | 职责 |
|---|---|
| `src/scene/draw_list.h` | `DrawEntry` 结构体定义 |
| `src/scene/draw_list.cpp` | `buildDrawList()` CPU 构建 draw 列表 |
| `src/renderer/culling/frustum_cull_pass.h` | Frustum cull pass 接口 |
| `src/renderer/culling/frustum_cull_pass.cpp` | Frustum cull compute pipeline 创建 + record |
| `src/renderer/culling/hiz_build_pass.h` | Hi-Z 构建 pass 接口 |
| `src/renderer/culling/hiz_build_pass.cpp` | Hi-Z compute pipeline 创建 + record |
| `src/app/app.h` | `App` 类: `m_indirectBuf`, `m_countBuf`, `m_useGpuCulling`, `m_useHiZOcclusion` |
| `src/app/app.cpp` | `recordIndirectDraws()`, `applySceneSelection()` 中创建 indirect buffers |
| `cmake/patch_spv.py` | SPIR-V InstanceIndex 补丁脚本 |

### Shader 文件

| 文件 | 职责 |
|---|---|
| `shaders/common/shared_types.slang` | `DrawData`, `IndirectDrawCommand` GPU 结构体 |
| `shaders/culling/frustum_cull.slang` | Cull compute shader: AABB frustum test + Hi-Z occlusion |
| `shaders/culling/hiz_build.slang` | Hi-Z 构建 compute shader: 4 级 max-reduce |
| `shaders/gbuffer/gbuffer.slang` | GBuffer VS: `SV_InstanceID` → `DrawData` |
| `shaders/forward/forward.slang` | Forward VS: 同上 |
| `shaders/gi/rsm/rsm_geometry.slang` | RSM VS: 同上 |
| `shaders/forward/forward_ibl.slang` | Forward IBL VS: 同上 |
