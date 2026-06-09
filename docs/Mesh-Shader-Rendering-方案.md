# Mesh Shader Rendering — 实现方案

日期: 2026-06-09

## 概述

SomeGI 当前使用传统的顶点着色器管线 + 独立的 compute shader（frustum culling）实现 GPU-Driven Rendering。`VK_EXT_MESH_SHADER` 扩展已在设备层检测但未启用。

本方案分两阶段实现 Mesh Shader 渲染：

- **Phase 1（本次实现）**：Task Shader 替代 compute culling + Mesh Shader 替代 VS，无 meshlet。保留 VS 路径作为 fallback。
- **Phase 2（后续）**：Meshlet 管线 —— 加载时生成 meshlet、Task Shader 逐个 meshlet cull、Mesh Shader 消费 meshlet 数据。

---

## Phase 1: Mesh Shader 直替 VS + Task Shader 替代 Culling

### Step 0 — Device Extension 启用

**文件**: `src/core/device.cpp`

参考 RT extension 的三步模式，添加 Mesh Shader 支持。

1. **Detection**（已有，line 55）：`m_features.meshShader = true` 已设置
2. **Enable extension**（在 RT 的 `enable_extension_if_present` 块后新增）：

```cpp
if (m_features.meshShader) {
    m_physicalDevice.enable_extension_if_present(VK_EXT_MESH_SHADER_EXTENSION_NAME);
}
```

3. **Feature struct + DeviceBuilder.add_pNext**（在 RT feature 块后新增）：

```cpp
VkPhysicalDeviceMeshShaderFeaturesEXT meshFeat{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
if (m_features.meshShader) {
    meshFeat.meshShader = VK_TRUE;
    meshFeat.taskShader = VK_TRUE;  // Task Shader 用于 culling
    db.add_pNext(&meshFeat);
    std::printf("[device] Mesh Shader enabled (EXT_mesh_shader)\n");
}
```

4. **`src/app/app.h`**：在 App 中添加 `bool m_useMeshShader = false;`（默认关），UI 中加 toggle

---

### Step 1 — 新 Shader 文件

创建三组 mesh shader .slang 文件：

#### `shaders/common/mesh_common.slang`（公共模块）

- 复用 `shared_types`、`gbuffer_layout` 的 import
- Task Shader payload 定义：

```hlsl
struct TaskPayload { uint drawIndex; };
```

- 共享函数：`aabbInFrustum()`、`isOccluded()`、`sampleHiZ()`（从 `frustum_cull.slang` 迁移）
- Task Shader UBO：`CullUniforms`（与现有 `CullUbo` 对齐）

#### `shaders/gbuffer/gbuffer_mesh.slang`（GBuffer Mesh + Task Shader）

- **Task Shader**（64 threads）：读 `StructuredBuffer<DrawData>`，frustum cull + Hi-Z test，通过者 `DispatchMesh(1,1,1, payload)`
- **Mesh Shader**（64 threads）：读 payload.drawIndex → DrawData → SSBO 中 vertex/index 数据 → `SetMeshOutputsEXT(vtxCount, triCount)` → 输出到 `gbuffer_layout` MRT
- Fragment Shader：复用 `shaders/gbuffer/gbuffer.slang` 中的 `ps_main`

**Mesh Shader 伪代码**：

```hlsl
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void ms_main(
    in payload TaskPayload pld,
    out vertices GBufferVsOut verts[MAX_VERTS],
    out indices uint3 tris[MAX_TRIS],
    out primitives GBufferOut prims[MAX_TRIS])
{
    DrawData dd = gDrawData[pld.drawIndex];
    uint triCount = dd.indexCount / 3;
    uint vtxCount = triCount * 3;
    if (tid == 0) SetMeshOutputsEXT(vtxCount, triCount);
    GroupMemoryBarrierWithGroupSync();
    // 并行读 vertex/index → transform → output
    for (uint i = tid; i < triCount; i += 64) {
        uint3 idx = 从 index SSBO 读取 3 个索引;
        // 从 vertex SSBO 读取 3 个顶点
        // transform with dd.modelMatrix + gFrame.viewProj
        // write to verts[i*3..i*3+2], tris[i], prims[i]
    }
}
```

**Task Shader 伪代码**：

```hlsl
[shader("task")]
[numthreads(64, 1, 1)]
void ts_main(uint tid : SV_DispatchThreadID) {
    if (tid >= gCull.drawCount) return;
    DrawData dd = gDrawData[tid];
    if (!aabbInFrustum(dd.aabbMin, dd.aabbMax, gCull.frustum)) return;
    if (gCull.hizMaxMip > 0 && isOccluded(...)) return;
    TaskPayload pld = { tid };
    DispatchMesh(1, 1, 1, pld);
}
```

#### `shaders/forward/forward_mesh.slang`（Forward Mesh + Task Shader）

模式同上。Task Shader 完全相同（cull 逻辑不变）。Mesh Shader 输出到单 color attachment + depth。Fragment Shader 复用 `shaders/forward/forward.slang` 中的 `ps_main`。

#### `shaders/gi/rsm/rsm_geometry_mesh.slang`（RSM Mesh + Task Shader）

RSM 不做 cull（sun 视角需要全量几何），Task Shader 只透传所有 draw。输出到 RSM 3×MRT（position, normal, flux — 全部 R16G16B16A16_SFLOAT）。

---

### Step 2 — C++ Pipeline 改造

#### 新建 `src/renderer/core/mesh_pipeline.h/cpp`

封装 Mesh Shader pipeline 创建的通用逻辑：

```cpp
struct MeshPipelineDesc {
    std::filesystem::path meshSpv;
    std::filesystem::path taskSpv;
    std::filesystem::path fragSpv;
    const char* meshEntry = "ms_main";
    const char* taskEntry = "ts_main";
    const char* fragEntry = "ps_main";
    std::vector<VkDescriptorSetLayout> setLayouts; // set=0: mesh+task bindings, set=1..N: FS bindings
    std::vector<VkFormat> colorFormats;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    bool depthTest = true;
    bool depthWrite = true;
};

VkPipeline createMeshPipeline(Device& d, const MeshPipelineDesc& desc);
```

实现要点：
- 3 个 shader stage：`VK_SHADER_STAGE_TASK_BIT_EXT` → `VK_SHADER_STAGE_MESH_BIT_EXT` → `VK_SHADER_STAGE_FRAGMENT_BIT`
- `VkPipelineVertexInputStateCreateInfo` 设 `vertexBindingDescriptionCount=0, vertexAttributeDescriptionCount=0`
- `VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST` 保持不变
- 使用 dynamic rendering（`VkPipelineRenderingCreateInfo` 链在 pNext）
- 其他 fixed-function 状态与现有 VS pipeline 一致

#### 修改 `GBufferPass`

新增接口：
```cpp
void setMeshShaderEnabled(bool v);   // 切换 mesh/VS 模式，内部 destroy + rebuild pipeline
bool meshShaderEnabled() const;
```

新增成员：
```cpp
bool m_useMeshShader = false;
VkPipeline m_meshPipeline = VK_NULL_HANDLE;
VkDescriptorSetLayout m_meshSetLayout = VK_NULL_HANDLE;  // 含 Task Shader bindings
VkDescriptorSet m_meshSet = VK_NULL_HANDLE;               // 含 Hi-Z + vertex/index SSBO
```

`record()` 修改：
```cpp
void GBufferPass::record(VkCommandBuffer cmd, ...) {
    // Dynamic rendering begin (same as before)
    if (m_useMeshShader) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_meshPipeline);
        // Bind mesh-specific descriptor set (includes Hi-Z + vertex/index SSBOs)
        vkCmdBindDescriptorSets(cmd, ..., m_meshSet, ...);
        // Dispatch mesh tasks — non-indirect: task shader handles culling internally
        vkCmdDrawMeshTasksEXT(cmd, (drawCount + 63) / 64, 1, 1);
    } else {
        // Existing VS path: vkCmdBindVertexBuffers + vkCmdBindIndexBuffer + vkCmdDrawIndexedIndirectCount
    }
    // Dynamic rendering end (same as before)
}
```

#### 修改 `ForwardPass`

同 GBufferPass 模式。区别是 forward 的 mesh pipeline 只有 1 个 color attachment。

#### 修改 `RsmGeometryPass`

同 GBufferPass 模式。区别：
- RSM 的 Task Shader 不做 cull（直通全部 draw）
- RSM 的 render target 固定 512×512

---

### Step 3 — App 集成

#### `recordIndirectDraws()` 改造

```cpp
void App::recordIndirectDraws(VkCommandBuffer cmd, uint32_t frameInFlight,
                               const glm::mat4& viewProj) {
    if (m_drawCount == 0) return;

    if (m_useMeshShader) {
        // Build Hi-Z if occlusion enabled (Task Shader reads it)
        if (m_useHiZOcclusion)
            m_renderer.hizPass().record(cmd, m_renderer.rt());

        // Update cull UBO for task shader (frustum planes + drawCount)
        // Pass this data to each pass's mesh descriptor set

        // Fill sun indirect buf for RSM (RSM uses direct dispatch)
        auto* sunCmds = (VkDrawIndexedIndirectCommand*)m_indirectBufSun.mapped();
        for (uint32_t i = 0; i < m_drawCount; ++i) {
            const auto& e = m_drawEntries[i];
            sunCmds[i] = {e.indexCount, 1, e.firstIndex, e.vertexOffset, i};
        }
        return;  // No compute cull dispatch, no indirect barrier
    }

    // === Existing indirect draw path ===
    if (m_useGpuCulling) {
        // ... (unchanged)
    } else {
        // ... (unchanged)
    }
}
```

关键技术点：
- `vkCmdDrawMeshTasksEXT(cmd, (drawCount + 63) / 64, 1, 1)` — 非 indirect 版本，Task Shader 内部 early-out
- 无需 indirect buffer → 无需 compute→indirect barrier
- Hi-Z 仍需单独构建（compute 在 task shader 之前）

#### `buildPipelineTable()` 改造

pipeline 执行表中的 GBuffer/Forward/RSM 步骤，根据 `m_useMeshShader` 切换 pass 的 internal mode。这些 lambda 已经捕获 `this`，内部直接调用 `m_renderer.gbuffer().record()` 等 → pass 内部根据 `m_useMeshShader` 选择 pipeline。

```cpp
// GBuffer step (pipeline table):
.record = [this](VkCommandBuffer cmd) {
    if (m_useMeshShader) {
        // mesh shader path: no vertex input barrier needed
        m_renderer.gbuffer().record(cmd, m_renderer.rt(),
            VK_NULL_HANDLE, m_drawCount, m_sceneGpu);
    } else {
        // existing VS path
        m_renderer.gbuffer().record(cmd, m_renderer.rt(),
            m_indirectBuf.handle(), m_drawCount, m_sceneGpu);
    }
}
```

#### `buildUI()` 添加 Mesh Shader toggle

在 Display tab 中添加：

```cpp
bool meshSupported = m_device->features().meshShader;
if (meshSupported) {
    if (ImGui::Checkbox("Mesh Shader", &m_useMeshShader)) {
        m_renderer.setUseMeshShader(m_useMeshShader);
        m_pipelineDirty = true;
    }
} else {
    ImGui::TextDisabled("Mesh Shader (GPU not supported)");
}
```

---

### Step 4 — CMake / Shader 编译

#### 新增 Shader 编译

在 `CMakeLists.txt` 的 `SHADER_SOURCES` glob 中，mesh shader 文件自然被包含（glob 已覆盖 `shaders/**/*.slang` 并排除 `shaders/common/`）：

- `shaders/gbuffer/gbuffer_mesh.slang` → `gbuffer_mesh.spv`
- `shaders/forward/forward_mesh.slang` → `forward_mesh.spv`
- `shaders/gi/rsm/rsm_geometry_mesh.slang` → `rsm_geometry_mesh.spv`
- `shaders/common/mesh_common.slang` → 不编译（import-only，已通过 common 排除规则忽略）

#### SPIR-V 补丁

Mesh Shader 不经过传统 VS 管线，不存在 `gl_InstanceIndex` / `gl_BaseInstance` 问题。`patch_spv.py` 的补丁列表**保持不变**（仍补丁 VS 路径的 spv）。

#### CMake OBJECT 库

在 `src/renderer/core/CMakeLists.txt` 中添加 `mesh_pipeline.cpp`。

---

### Step 5 — Descriptor 布局设计

#### Task Shader bindings（set=0 的扩展部分）

| Binding | Type | Stage | 内容 |
|---------|------|-------|------|
| 0 | `STORAGE_BUFFER` | Task | `StructuredBuffer<DrawData>` |
| 1 | `UNIFORM_BUFFER` | Task | CullUniforms (frustum planes, screen size, drawCount, hizMaxMip) |
| 4 | `SAMPLED_IMAGE` | Task | Hi-Z Mip 1 |
| 5 | `SAMPLED_IMAGE` | Task | Hi-Z Mip 2 |
| 6 | `SAMPLED_IMAGE` | Task | Hi-Z Mip 3 |
| 7 | `SAMPLED_IMAGE` | Task | Hi-Z Mip 4 |

#### GBuffer Mesh Shader bindings（同一 set，Mesh Shader 可见部分）

| Binding | Type | Stage | 内容 |
|---------|------|-------|------|
| 0 | `UNIFORM_BUFFER` | Mesh+Fragment | FrameUniforms |
| 2 | `SAMPLER` | Fragment | Linear sampler |
| 3 | `SAMPLED_IMAGE[]` | Fragment | Texture array (partially bound) |
| 8 | `STORAGE_BUFFER` | Mesh | `StructuredBuffer<Vertex>` (vertex data SSBO) |
| 9 | `STORAGE_BUFFER` | Mesh | `StructuredBuffer<uint>` (index data SSBO) |
| 10 | `STORAGE_BUFFER` | Mesh | `StructuredBuffer<DrawData>` |

注意：
- Binding 1 同时被 Task Shader（CullUniforms）和 Fragment Shader（MaterialGpu）使用，但在不同的 shader stage 可见。实际上需要分开：Task Shader 的 binding 1 是 CullUbo，Fragment Shader 的 binding 1 是 MaterialGpu。需要在不同的 descriptor set 中处理，或者使用不同的 binding 号。
- 最终方案：**set=0 供 Task+Mesh Shader**（含 CullUbo, DrawData, Hi-Z, vertex/index SSBO, FrameUniforms），**set=1 供 Fragment Shader**（含 MaterialGpu, sampler, textures），与现有 IBL set=1 对齐。

#### 最终 Descriptor Set 分配

| Set | Binding | Type | Stage | 内容 |
|-----|---------|------|-------|------|
| 0 | 0 | `STORAGE_BUFFER` | Task+Mesh | DrawData[] |
| 0 | 1 | `UNIFORM_BUFFER` | Task | CullUniforms |
| 0 | 2 | `UNIFORM_BUFFER` | Mesh+Fragment | FrameUniforms |
| 0 | 3 | `STORAGE_BUFFER` | Mesh | Vertex[] SSBO |
| 0 | 4 | `STORAGE_BUFFER` | Mesh | Index[] SSBO |
| 0 | 5-8 | `SAMPLED_IMAGE` | Task | Hi-Z Mip 1-4 |
| 1 | 0 | `STORAGE_BUFFER` | Fragment | MaterialGpu[] |
| 1 | 1 | `SAMPLER` | Fragment | Linear sampler |
| 1 | 2 | `SAMPLED_IMAGE[]` | Fragment | Texture array |

#### Vertex/Index Buffer STORAGE 使用

当前 `uploadScene()` 仅在 RT 可用时给 vertex/index buffer 加 `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`。需要修改 `src/scene/upload.cpp`，在 Mesh Shader 可用时也加此标志（或始终添加）。

---

### Step 6 — Mesh Shader 与 Vertex Shader 双模式架构

在 `FrameRenderer` 中添加：

```cpp
// frame_renderer.h
bool meshShaderSupported() const { return m_meshShaderSupported; }
bool useMeshShader() const { return m_useMeshShader; }
void setUseMeshShader(bool v);
```

`setUseMeshShader()` 实现：
1. 设置 `m_useMeshShader = v`
2. 对 GBufferPass / ForwardPass / RsmGeometryPass 调用 `setMeshShaderEnabled(v)`
3. 每个 Pass 内部 `destroyPipeline()` → 重建对应的 pipeline（mesh 或 VS）

---

### Step 7 — Hi-Z 在 Mesh Shader 路径中的处理

当前 Hi-Z 在 compute cull 之前构建（读上一帧 depth）。Mesh Shader 路径中：

- Hi-Z build 仍作为独立 compute pass 执行（在 `recordIndirectDraws()` 中，Task Shader dispatch 之前）
- Hi-Z mip views 绑定到每个 pass 的 mesh descriptor set（set=0, bindings 5-8）
- 当 `hizMaxMip == 0`（Hi-Z 未启用）时，Task Shader 中 `isOccluded()` 直接返回 false

---

### Step 8 — 验证计划

| # | 验证项 | 方法 | 预期 |
|---|--------|------|------|
| 1 | 编译通过 | CMake build | 所有 .spv 生成无报错 |
| 2 | 启动检测 | 运行程序 | 控制台输出 `[device] Mesh Shader enabled` |
| 3 | VS fallback | Mesh Shader off 时渲染 | 现有 GBuffer/Forward/RSM 正常 |
| 4 | MS 切换 | ImGui 开启 Mesh Shader toggle | 画面与 VS 模式一致 |
| 5 | Culling 正确 | Sponza 场景，对比 culled count | MS 模式与 VS 模式一致 |
| 6 | Hi-Z 遮挡 | 站在墙后看物体 | 被遮挡物体正确剔除 |
| 7 | 场景切换 | Sponza ↔ Cube ↔ DamagedHelmet | 无 crash，pipeline 正常重建 |
| 8 | GI 兼容 | 分别切换各 GI 技术 | 所有 GI 技术与 MS 模式兼容 |

---

## Phase 2: Meshlet 管线（后续实现概要）

### Meshlet 数据结构

```cpp
struct Meshlet {
    uint32_t vertexOffset;     // meshlet vertex buffer 中的偏移
    uint32_t triangleOffset;   // primitive index buffer 中的偏移
    uint32_t vertexCount;      // 最多 64
    uint32_t triangleCount;    // 最多 126
    glm::vec3 aabbMin;         // meshlet 级 AABB（culling 用）
    glm::vec3 aabbMax;
    uint32_t meshIndex;        // 所属 mesh（查材质用）
};
```

### 加载时生成

```cpp
void buildMeshlets(const SceneCpu& cpu, SceneGpu& gpu) {
    // 使用 meshoptimizer::meshopt_buildMeshlets()
    // 生成 3 个 SSBO: meshlet vertices, meshlet triangles (uint8), meshlet metadata
}
```

### 收益

- Cull 粒度从 per-draw（~1000s triangles）降到 per-meshlet（~126 triangles）
- 遮挡剔除更精确
- Mesh Shader 读写紧凑的 meshlet 数据（cache-friendly），不再随机访问 global vertex/index buffer

---

## 涉及文件索引

### 新增文件

| 文件 | 职责 |
|------|------|
| `src/renderer/core/mesh_pipeline.h` | Mesh pipeline 创建辅助 |
| `src/renderer/core/mesh_pipeline.cpp` | `createMeshPipeline()` 实现 |
| `shaders/common/mesh_common.slang` | Task/Mesh Shader 公共定义 |
| `shaders/gbuffer/gbuffer_mesh.slang` | GBuffer Task + Mesh Shader |
| `shaders/forward/forward_mesh.slang` | Forward Task + Mesh Shader |
| `shaders/gi/rsm/rsm_geometry_mesh.slang` | RSM Task + Mesh Shader |

### 修改文件

| 文件 | 改动 |
|------|------|
| `src/core/device.cpp` | 启用 VK_EXT_MESH_SHADER 扩展 + feature struct |
| `src/renderer/core/gbuffer_pass.h/cpp` | 双 pipeline 模式 (VS + MS) |
| `src/renderer/core/forward_pass.h/cpp` | 同上 |
| `src/renderer/gi/rsm/rsm_geometry_pass.h/cpp` | 同上 |
| `src/renderer/core/frame_renderer.h/cpp` | `setUseMeshShader()` + `meshShaderSupported()` |
| `src/app/app.h/cpp` | `m_useMeshShader`, `recordIndirectDraws()`, `buildUI()` |
| `src/scene/upload.cpp` | vertex/index buffer 始终添加 STORAGE 使用 |
| `src/renderer/core/CMakeLists.txt` | 添加 `mesh_pipeline.cpp` |
