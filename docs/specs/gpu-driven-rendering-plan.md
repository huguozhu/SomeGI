# GPU Driven Rendering 方案与实施步骤

## 目标

将 Geometry Pass（GBuffer/Forward/RSM）从 CPU 遍历 Push Constants → 逐 primitive Draw，
改为 GPU 间接多绘制 + 视锥剔除。

## 当前瓶颈（Sponza 103 draws）

```
for (node : cpu.nodes)
  for (primitive : mesh.primitives)
    vkCmdPushConstants(model, materialIndex)
    vkCmdDrawIndexed(firstIndex, indexCount)  // 每 primitive 1 次
```

## 目标管线

```
[Scene Load] → buildDrawList() → DrawData[] SSBO (static, GPU)
[每帧]
  GPU Cull CS: DrawData[] → IndirectDrawCommand[] + count
  Barrier: COMPUTE_WRITE → INDIRECT_READ
  vkCmdDrawIndexedIndirectCount(buf, count)
```

---

## Phase 1: 间接多绘制 (Indirect Multi-Draw)

### Step 1.1: 数据结构

**shaders/common/shared_types.slang** — 在 PushConsts 前添加：

```slang
// GPU-Driven: per-draw data indexed by SV_InstanceID (via firstInstance)
struct DrawData {
    float4x4 modelMatrix;    // 64 bytes
    uint     materialIndex;  // 4
    int      firstIndex;     // 4
    uint     indexCount;     // 4
    int      vertexOffset;   // 4
    float3   aabbMin;        // 12
    uint     _pad0;          // 4
    float3   aabbMax;        // 12
    uint     _pad1;          // 4
}; // 112 bytes — 7 x vec4
```

**src/scene/draw_list.h/cpp** — 新文件，CPU 端构建函数：

```cpp
struct DrawEntry { /* same layout, 112 bytes, static_assert */ };
void buildDrawList(const SceneCpu& cpu, std::vector<DrawEntry>& out);
```

**src/scene/scene.h** — SceneGpu 添加：

```cpp
Buffer drawDataBuffer;
uint32_t drawCount = 0;
```

### Step 1.2: 三个 Shader 改造

**gbuffer.slang / forward.slang / rsm_geometry.slang**：

```slang
// 替换 push_constant → DrawData SSBO
[[vk::binding(4, 0)]] StructuredBuffer<DrawData> gDrawData;

// VS 改为 SV_InstanceID
VsOut vs_main(VsIn i, uint drawID : SV_InstanceID) {
    DrawData dd = gDrawData[drawID];
    // dd.modelMatrix 替代 gPC.model
    // int(dd.materialIndex) 替代 gPC.materialIndex
}
```

关键：Slang SPIR-V 不支持 `SV_DrawID`，用 `SV_InstanceID` + `firstInstance = drawIndex` 绕过。

### Step 1.3: 三个 C++ Pass 改造

每个 pass（GBuffer/Forward/RSM）：

1. **Descriptor Set Layout** 加 binding 4:
   ```cpp
   b[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT};
   ```

2. **Pool Size** 改 STORAGE_BUFFER count 1→2

3. **Pipeline Layout** 移除 push constants

4. **bindDrawData()** 新方法：写 binding 4 descriptor

5. **record()** 改为接受 `VkBuffer indirectBuf, uint32_t drawCount`:
   ```cpp
   vkCmdDrawIndexedIndirectCount(cmd, indirectBuf, 0,
       countBuf, 0, maxDrawCount, sizeof(VkDrawIndexedIndirectCommand));
   ```

### Step 1.4: App 集成

**app.cpp 构造函数**：`m_renderer.init()` 之后无额外操作（draw list 在 scene load 时构建）

**applySceneSelection()**：场景加载后
```cpp
std::vector<DrawEntry> draws;
buildDrawList(m_scene, draws);
// 上传到 m_sceneGpu.drawDataBuffer（host-visible）
// 创建 indirect buffer（host-visible, maxDraws * 20 bytes）
// 调用 m_renderer.gbuffer().bindDrawData(...) 等
```

**run() 每帧**：
```cpp
// CPU 填充 indirect buffer（Phase 1 无 culling）
auto* icmds = (VkDrawIndexedIndirectCommand*)m_indirectBuf.mapped();
for (uint32_t i = 0; i < drawCount; ++i) {
    icmds[i] = {draws[i].indexCount, 1, draws[i].firstIndex, draws[i].vertexOffset, i};
}
// 传递到 record()
m_renderer.gbuffer().record(cmd, m_renderer.rt(), m_indirectBuf.handle(), drawCount, m_sceneGpu);
```

### Step 1.5: ImGui Toggle

Display 面板添加：
```
☐ GPU-Driven Rendering
```
勾选时走 indirect draw，不勾选时走原始 CPU 路径（兼容回退）。

### Step 1.6: device.cpp

启用 `f12.drawIndirectCount = VK_TRUE;`

---

## Phase 2: GPU 视锥剔除

### Step 2.1: Culling Shader

**shaders/culling/frustum_cull.slang**（新文件）：

```slang
struct CullUniforms { float4 frustum[6]; uint drawCount; };

[[vk::binding(0, 0)]] StructuredBuffer<DrawData> gIn;
[[vk::binding(1, 0)]] ConstantBuffer<CullUniforms> gCull;
[[vk::binding(2, 0)]] RWStructuredBuffer<IndirectDrawCommand> gOut;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> gCount;

[numthreads(256,1,1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= gCull.drawCount) return;
    DrawData d = gIn[tid.x];
    if (!aabbInFrustum(d.aabbMin, d.aabbMax, gCull.frustum)) return;
    uint idx; InterlockedAdd(gCount[0], 1, idx);
    gOut[idx] = {d.indexCount, 1, d.firstIndex, d.vertexOffset, tid.x};
}
```

### Step 2.2: C++ Pass

**src/renderer/core/frustum_cull_pass.h/cpp**（新文件）：
- Descriptor layout: DrawData(in) + CullUniforms(ubo) + IndirectCmd(out) + Count(out)
- `record(cmd, drawBuf, count, indirectBuf, countBuf, viewProj, flightIdx)`
- UBO 每帧更新 frustum planes
- 双缓冲 descriptor set（kFramesInFlight=2）

### Step 2.3: 集成

```cpp
// run() 中：
if (m_gpuDriven && drawCount > 0) {
    m_cullPass.record(cmd, drawDataBuf, drawCount, indirectBuf, countBuf, vp, flightIdx);
    // Barrier: COMPUTE_SHADER WRITE → DRAW_INDIRECT READ
    VkBufferMemoryBarrier2 b{..., INDIRECT_BUFFER};
    vkCmdPipelineBarrier2(cmd, &b);
    m_culledDrawCount = drawCount;  // GPU 控制实际数量
} else {
    // CPU fill (Phase 1 回退)
}
m_renderer.gbuffer().record(cmd, ..., indirectBuf, m_culledDrawCount, ...);
```

### Step 2.4: RSM 特殊处理

RSM 太阳视锥覆盖整个场景 → 保留 CPU fill 路径（不需要 culling）。

---

## Phase 3: Hi-Z 遮挡剔除（可选，延期）

---

## 实施顺序

| 步 | 内容 | 文件改动 | 预估 |
|---|---|---|---|
| 1 | DrawData + draw_list + SceneGpu | shared_types.slang, draw_list.h/cpp, scene.h/.cpp, scene/CMakeLists | 6 files |
| 2 | 三个 shader 改造 | gbuffer.slang, forward.slang, rsm_geometry.slang | 3 files |
| 3 | Pass 改造 + device.cpp | gbuffer_pass.h/cpp, forward_pass.h/cpp, rsm_geometry_pass.h/cpp, device.cpp | 7 files |
| 4 | App 集成 + toggle | app.h, app.cpp | 2 files |
| 5 | 编译验证 + 测试 | — | — |
| 6 | frustum_cull.slang + FrustumCullPass | 3 new files | 3 files |
| 7 | Phase 2 App 集成 | app.cpp, CMakeLists | 2 files |

每步后 `cmake --build` 验证零错误。

## 验证

1. Phase 1 后：禁用 toggle 走原始路径，启用走 indirect draw → 画面应完全一致
2. Phase 2 后：旋转相机观察 FPS 提升（视锥外物体被剔除）
3. F2 Benchmark 对比各 GI 模式的 GPU time
