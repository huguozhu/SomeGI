# Renderer Pass RHI 迁移设计文档

日期：2026-06-21
分支：d3d12
状态：设计阶段
父文档：[[rhi-design]]

---

## 1. 目标

将 `src/renderer/` 下 11 个 Hybrid pass 从混合 Vulkan/RHI 状态迁移为纯 RHI，
消除所有 `core::Device&` 依赖和原生 `vkCmd*` 调用。

**不纳入本轮：**
- gbuffer_pass / forward_pass 的 mesh shader 路径（D3D12 `drawMeshTasks` 空壳）
- imgui_pass（ImGui 内部使用 Vulkan backend，无法通过 RHI 抽象）
- shadow_pass 的 RT 路径（TLAS 创建仍用原生 Vulkan，`createRayTracingPSO` 未实现）

---

## 2. 迁移模式（以 ssao_pass 为参考）

SSAO pass (`src/renderer/ao/ssao_pass.cpp`) 是已完成的纯 RHI 参考实现，所有 pass 以此为模板：

### 2.1 资源创建

```cpp
// 旧模式（core::Device + 原生 Vulkan 类型）
class SomePass {
    core::Device& m_device;
    core::Buffer  m_ubo;
    core::Image   m_tex;
    void init(core::Device& d) { ... }
};

// 新模式（纯 RHI）
class SomePass {
    rhi::RHIDevice& m_device;
    std::unique_ptr<rhi::RHIBuffer>  m_ubo;
    std::unique_ptr<rhi::RHITexture> m_tex;
    void init(rhi::RHIDevice& d) { ... }
};
```

### 2.2 命令录制

```cpp
// 旧模式（原生 Vulkan）
void record(VkCommandBuffer cmd) {
    vkCmdBindPipeline(cmd, ...);
    vkCmdBindDescriptorSets(cmd, ...);
    vkCmdDispatch(cmd, gx, gy, gz);
}

// 新模式（RHI）
void record(rhi::RHICommandBuffer& cmd) {
    cmd.bindPipelineState(*m_pso);
    cmd.bindDescriptorSet(0, *m_set);
    cmd.dispatch(gx, gy, gz);
}
```

### 2.3 删除 VkCompat 重载

```cpp
// 删除：
void record(VkCommandBuffer vkCmd, ...) {
    VkRHICommandBuffer wrapper(vkCmd);
    record(wrapper, ...);
}
// 调用方统一使用 RHI 路径
```

### 2.4 Descriptor 写入

```cpp
// 旧模式
VkDescriptorImageInfo imageInfo{...};
VkWriteDescriptorSet w{...};
vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);

// 新模式
m_set->write({
    {.binding=0, .type=DescriptorType::SampledImage, .textureView=view},
    {.binding=1, .type=DescriptorType::UniformBuffer, .buffer=ubo},
});
```

---

## 3. API 映射速查表

### 3.1 类型映射

| 旧 (`core::`) | RHI (`rhi::`) | 备注 |
|---|---|---|
| `Device&` | `RHIDevice&` | 全局替换 |
| `Buffer` | `unique_ptr<RHIBuffer>` | 通过 `createBuffer(BufferDesc)` 创建 |
| `Image` | `unique_ptr<RHITexture>` + `unique_ptr<RHITextureView>` | 纹理与视图分离 |
| `ShaderModule` | `unique_ptr<RHIShader>` | 通过 `createShader(desc, bytecode, size)` 创建 |
| `VkSampler` | `unique_ptr<RHISampler>` | 通过 `createSampler(SamplerDesc)` 创建 |
| `VkCommandBuffer` | `RHICommandBuffer&` | 通过 `RHICommandPool::allocateRaw()` 获取 |
| `VkFence` | `unique_ptr<RHIFence>` | `createFence()` |
| `VkSemaphore` | `unique_ptr<RHISemaphore>` | `createSemaphore()` |

### 3.2 命令映射

| 原生 Vulkan | RHI (`RHICommandBuffer`) |
|---|---|
| `vkCmdBindPipeline` | `bindPipelineState(pso)` |
| `vkCmdBindDescriptorSets` | `bindDescriptorSet(slot, set)` |
| `vkCmdPushConstants` | `pushConstants(stage, data, size, offset)` |
| `vkCmdBindVertexBuffers` | `bindVertexBuffer(binding, buf, offset, stride)` |
| `vkCmdBindIndexBuffer` | `bindIndexBuffer(buf, offset, uint16)` |
| `vkCmdDraw / vkCmdDrawIndexed` | `draw(...)` / `drawIndexed(...)` |
| `vkCmdDrawIndexedIndirect` | `drawIndexedIndirect(...)` |
| `vkCmdDrawIndexedIndirectCount` | `drawIndexedIndirectCount(...)` |
| `vkCmdDrawMeshTasksEXT` | `drawMeshTasks(...)`（D3D12 暂未实现） |
| `vkCmdDispatch` | `dispatch(gx, gy, gz)` |
| `vkCmdDispatchIndirect` | `dispatchIndirect(buf, offset)` |
| `vkCmdCopyBuffer` | `copyBuffer(src, dst, size)` |
| `vkCmdCopyImage` | `copyTexture(src, dst)` |
| `vkCmdPipelineBarrier2` | `textureBarrier(tex, old, new)` / `bufferBarrier(buf, ...)` / `globalBarrier()` |
| `vkCmdClearColorImage` | `clearColor(tex, r, g, b, a)` |
| `vkCmdClearDepthStencilImage` | `clearDepth(tex, depth, stencil)` |
| `vkCmdSetViewport` | `setViewport(x, y, w, h)` |
| `vkCmdSetScissor` | `setScissor(x, y, w, h)` |
| `vkCmdBeginRendering` | `beginRendering(colors, count, depth, w, h)` |
| `vkCmdEndRendering` | `endRendering()` |
| `vkCmdWriteTimestamp` | `writeTimestamp(pool, index)` |
| `vkCmdResetQueryPool` | `resetQueryPool(pool, first, count)` |

### 3.3 资源创建映射

```cpp
// Buffer: core::Buffer(d, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, ...)
// → RHI:
auto buf = device.createBuffer({
    .size = size,
    .usage = BufferUsage::Uniform,
    .memoryType = MemoryType::Upload,
});

// Image: core::Image(d, ImageDesc{.format=VK_FORMAT_R16G16B16A16_SFLOAT, ...})
// → RHI:
auto tex = device.createTexture({
    .format = Format::R16G16B16A16_SFLOAT,
    .width = w, .height = h,
    .usage = TextureUsage::Sampled | TextureUsage::Storage,
    .mipLevels = 1,
});
auto view = device.createTextureView(*tex, {
    .type = TextureViewType::View2D,
    .baseMip = 0, .mipCount = 1,
});
```

---

## 4. 分 Pass 迁移清单

### 4.1 低难度（4 个）

#### skybox_pass
- **残留**：`m_device`（`core::Device*`）未使用
- **改法**：删除 `m_device` 成员和 `init()` 中的 `Device&` 参数

#### smaa_pass
- **残留**：`init()` 中用 `core::Image(d, desc)` 创建 edge texture
- **改法**：改用 `rhiDevice.createTexture()` + `createTextureView()`

#### frustum_cull_pass
- **残留**：`init()` 接受 `Device& d` 参数
- **改法**：删除 `Device&` 参数，仅保留 `RHIDevice&`

#### hiz_build_pass
- **残留**：`init()` 接受 `Device& d` 参数
- **改法**：删除 `Device&` 参数

### 4.2 中难度（5 个）

#### rsm_geometry_pass
- **残留**：`core::Image` + 原生 `transition()` helper（`vkCmdPipelineBarrier2`）
- **改法**：Image → `createTexture()`；`transition()` → `textureBarrier()`

#### vxgi_relight_pass
- **残留**：VkCompat 重载中有原生 `vkCmdBindPipeline`、`vkCmdBindDescriptorSets`、`vkCmdPushConstants`、`vkCmdDispatch`
- **改法**：删除 VkCompat，调用方用 RHI 路径

#### vxgi_aniso_pass
- **残留**：`record()` 中提取原生 `VkCommandBuffer` 调用 `vkCmdBindPipeline`、`vkCmdPipelineBarrier2`
- **改法**：改用 `cmd.bindPipelineState()` + RHI barrier

#### vxgi_mipmap_pass
- **残留**：`record()` 中提取 `VkCommandBuffer` 调用 `vkCmdPipelineBarrier2`
- **改法**：改用 `cmd.textureBarrier()` / `cmd.globalBarrier()`

#### ddgi_pass
- **残留**：`record()` 全原生 `vkCmd*`（bindPipeline、bindDescriptorSets、pushConstants、dispatch、pipelineBarrier2）
- **改法**：全部替换为 RHI 命令

### 4.3 高难度（2 个）

#### lumen_probe_pass
- **残留**：原生 `vkCmd*` + RT TLAS 桥接（`VkAccelerationStructureKHR` + `createNonOwning()`）
- **改法**：命令改为 RHI；TLAS 桥接保留（`createRayTracingPSO` 未实现）

#### lighting_pass
- **残留**：RT TLAS + `Device&` 依赖 + VkCompat 重载
- **改法**：资源创建改 RHI；删除 VkCompat；TLAS 桥接保留

---

## 5. 迁移后验证

每个 pass 迁移后：
1. **编译通过**：`cmake --build build --target somegi_rhi`
2. **VkCompat 删除**：确认 `record(VkCommandBuffer, ...)` 重载已删除
3. **`core::Device&` 消除**：确认 pass 头文件中无 `core::Device` 引用
4. **全量编译**：确认所有调用方的 `record(VkCommandBuffer)` 已改为 `record(RHICommandBuffer&)`

---

## 6. 不迁移的路径

| Pass | 不迁移路径 | 原因 |
|------|-----------|------|
| gbuffer_pass | mesh shader | `drawMeshTasks` D3D12 空壳 |
| forward_pass | mesh shader | 同上 |
| shadow_pass | RT TLAS 创建 | `createRayTracingPSO` 未实现 |
| imgui_pass | 全部 | ImGui 内部原生 Vulkan backend |
| lighting_pass | RT TLAS 桥接 | 保留 `createNonOwning()` 模式（正确用法） |
| lumen_probe_pass | RT TLAS 桥接 | 同上 |
| ndgi_pass | RT TLAS 桥接 | 已 RHI，仅 RT 桥接保留 |
| restir_pass | RT TLAS 桥接 | 已 RHI，仅 RT 桥接保留 |
| rt_gi_pass | RT TLAS 桥接 | 已 RHI，仅 RT 桥接保留 |
