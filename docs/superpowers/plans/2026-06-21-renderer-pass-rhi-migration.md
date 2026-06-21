# Renderer Pass RHI 迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 11 个 Hybrid renderer pass 迁移为纯 RHI，消除 `core::Device&` 依赖和原生 `vkCmd*` 调用。

**Architecture:** 每个 pass 独立迁移，按 4 步模式：消除 `core::Device&` → 消除原生 `vkCmd*` → 删除 VkCompat 重载 → 清理非拥有型桥接。按难度分 3 组：低（4 个）→ 中（5 个）→ 高（2 个）。

**Tech Stack:** C++17, Vulkan, D3D12, RHI

---

### Task 1: skybox_pass — 消除 Device& 依赖 + 删除 VkCompat

**Files:**
- Modify: `src/renderer/core/skybox_pass.h`
- Modify: `src/renderer/core/skybox_pass.cpp`

- [ ] **Step 1: 删除头文件中 core::Device 引用**

在 `skybox_pass.h` 中：
- 删除第 12 行 `class Device;` 前向声明
- 第 25 行 `init(Device& d, rhi::RHIDevice& rhiDevice, ...)` → `init(rhi::RHIDevice& d, ...)`
- 第 35 行删除 `void record(VkCommandBuffer cmd, const RenderTargets& rt);` 声明
- 第 38 行删除 `Device* m_device = nullptr;`
- 第 45 行 `Buffer m_ubo;` → `std::unique_ptr<rhi::RHIBuffer> m_ubo;`
- 第 47-48 行 `VkFormat m_colorFmt, m_depthFmt` → `rhi::Format m_colorFmt, m_depthFmt`

- [ ] **Step 2: 修改 init() — Buffer 创建改用 RHI**

在 `skybox_pass.cpp` 的 `init()` 中：
- 用 `rhi::RHIDevice& d` 替换 `Device& d` 参数
- 删除 `m_device = &d;`
- `m_ubo = Buffer(d, sizeof(SkyUbo), ...)` 替换为：
```cpp
m_ubo = d.createBuffer({
    .size = sizeof(SkyUbo),
    .usage = rhi::BufferUsage::Uniform,
    .memoryType = rhi::MemoryType::Upload,
});
```
- `m_colorFmt = rtHdrFormat; m_depthFmt = rtDepthFormat;` 中的赋值改用 `rhi::Format`

- [ ] **Step 3: 更新 bindEnv / bindEnvRHI 签名和内部引用**

- `bindEnv` 的参数 `VkImageView envCubeView, VkSampler linearSampler` → `const rhi::RHITextureView& envCubeView, const rhi::RHISampler& linearSampler`
- `bindEnvRHI` 参数 `VkImageView envCubeView` → `const rhi::RHITextureView& envCubeView`
- 更新 `m_set->write()` 中的 createNonOwning 桥接调用，改用 RHI 对象直接引用

- [ ] **Step 4: 删除 VkCompat record() 重载**

删除 `skybox_pass.cpp` 中 `record(VkCommandBuffer vkCmd, ...)` 的函数定义（~157-161 行）。

- [ ] **Step 5: 更新调用方**

搜索所有调用 `skybox_pass.record(VkCommandBuffer, ...)` 的代码，改为 `record(rhi::RHICommandBuffer&, ...)`。
同时更新 `init(Device&, ...)` 调用方为 `init(RHIDevice&, ...)`。

- [ ] **Step 6: 编译验证并提交**

```powershell
cmake --build build --target somegi_rhi
```

```bash
git add src/renderer/core/skybox_pass.h src/renderer/core/skybox_pass.cpp
git commit -m "迁移 skybox_pass 到纯 RHI：消除 Device& 依赖 + 删除 VkCompat

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: smaa_pass — 消除 core::Image + 原生 barrier

**Files:**
- Modify: `src/renderer/core/smaa_pass.h`
- Modify: `src/renderer/core/smaa_pass.cpp`

- [ ] **Step 1: 删除头文件中 core 依赖**

在 `smaa_pass.h` 中：
- 删除第 3 行 `#include "core/image.h"`
- 删除第 8 行 `class Device;`
- 第 15 行 `init(Device& dev, rhi::RHIDevice& d, ...)` → `init(rhi::RHIDevice& d, ...)`
- 第 20 行删除 `void record(VkCommandBuffer cmd, const RenderTargets& rt);`
- 第 35 行 `Image m_edgeTex;` → `std::unique_ptr<rhi::RHITexture> m_edgeTex;` + 添加 `std::unique_ptr<rhi::RHITextureView> m_edgeView;`
- 第 36 行 `VkImageLayout m_edgeLayout` 删除

- [ ] **Step 2: 修改 init() — Image 创建改用 RHI**

在 `smaa_pass.cpp` 的 `init()` 中：
- 用 `rhi::RHIDevice& d` 替换 `Device& dev, rhi::RHIDevice&` 双参数
- `m_edgeTex = Image(dev, ed)` 替换为：
```cpp
m_edgeTex = d.createTexture({
    .format = toRHIFormat(ed.format),
    .width = ed.extent.width, .height = ed.extent.height,
    .usage = rhi::TextureUsage::Sampled | rhi::TextureUsage::Storage,
    .mipLevels = 1,
});
m_edgeView = d.createTextureView(*m_edgeTex, {
    .type = rhi::TextureViewType::View2D,
    .baseMip = 0, .mipCount = 1,
});
```

- [ ] **Step 3: 消除原生 vkCmdPipelineBarrier2（line 118）**

在 `record(rhi::RHICommandBuffer&)` 中，找到 `vkCmdPipelineBarrier2` 调用（~106-118 行），替换为：
```cpp
cmd.textureBarrier(*m_edgeTex, oldLayout, newLayout);
```
其中 `oldLayout` 和 `newLayout` 用 `rhi::TextureLayout` 枚举值（根据原来 barrier 的 `srcAccessMask`/`dstAccessMask` 推断）。

- [ ] **Step 4: 删除 VkCompat record() 重载**

删除 `record(VkCommandBuffer vkCmd, ...)` 定义（~149-152 行）。

- [ ] **Step 5: 更新 createNonOwning 桥接为直接引用**

`bindResources` / `bindOutput` 中的 `VkRHITextureView::createNonOwning(d, ...)` 改为直接使用 `m_edgeView.get()` 或其他 RHI texture view 指针。

- [ ] **Step 6: 编译验证并提交**

```powershell
cmake --build build --target somegi_rhi
```

```bash
git add src/renderer/core/smaa_pass.h src/renderer/core/smaa_pass.cpp
git commit -m "迁移 smaa_pass 到纯 RHI：消除 core::Image + 原生 barrier

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: frustum_cull_pass — 消除 Device& 依赖 + 原生 vkCmdFillBuffer/barrier

**Files:**
- Modify: `src/renderer/culling/frustum_cull_pass.h`
- Modify: `src/renderer/culling/frustum_cull_pass.cpp`

- [ ] **Step 1: 删除头文件中 core 依赖**

在 `frustum_cull_pass.h` 中：
- 删除第 10 行 `class Device;`
- 第 25 行 `init(Device& d, rhi::RHIDevice& rhiDevice, ...)` → `init(rhi::RHIDevice& d, ...)`
- 删除第 41-49 行所有 `record(VkCommandBuffer, ...)` 重载声明
- 删除第 52 行 `Device* m_device = nullptr;`
- 第 58 行 `Buffer m_ubo;` → `std::unique_ptr<rhi::RHIBuffer> m_ubo;`

- [ ] **Step 2: 修改 init() — Buffer 改用 RHI**

在 `frustum_cull_pass.cpp` 中：
- 删除 `m_device = &d;`
- `m_ubo = Buffer(d, sizeof(CullUbo), ...)` 替换为：
```cpp
m_ubo = d.createBuffer({
    .size = sizeof(CullUbo),
    .usage = rhi::BufferUsage::Uniform,
    .memoryType = rhi::MemoryType::Upload,
});
```

- [ ] **Step 3: 消除 record() 中 vkCmdFillBuffer + vkCmdPipelineBarrier2**

在 RHI 路径的 `record()` 中（~137-182 行），找到提取 `VkCommandBuffer` 的代码块（~146-162 行）：
- `vkCmdFillBuffer(vkCmd, vkCountOut, ...)` → 使用 RHI 的 `fillBuffer`
- `VkBufferMemoryBarrier2` + `vkCmdPipelineBarrier2` → 使用 `cmd.bufferBarrier(...)`

- [ ] **Step 4: 删除 VkCompat record() 重载**

删除两个 `record(VkCommandBuffer, ...)` 重载定义（~185-208 行）。

- [ ] **Step 5: 编译验证并提交**

```powershell
cmake --build build --target somegi_rhi
```

```bash
git add src/renderer/culling/frustum_cull_pass.h src/renderer/culling/frustum_cull_pass.cpp
git commit -m "迁移 frustum_cull_pass 到纯 RHI：消除 Device& + 原生 fillBuffer/barrier

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: hiz_build_pass — 消除 Device& 依赖

**Files:**
- Modify: `src/renderer/culling/hiz_build_pass.h`
- Modify: `src/renderer/culling/hiz_build_pass.cpp`

- [ ] **Step 1: 搜索 core::Device 引用并消除**

在 `hiz_build_pass.h` 和 `.cpp` 中：
- 删除 `class Device;` 前向声明
- `init(Device& d, rhi::RHIDevice& rhiDevice, ...)` → `init(rhi::RHIDevice& d, ...)`
- 删除 `Device* m_device` 成员
- 删除 `record(VkCommandBuffer, ...)` 重载声明和定义

- [ ] **Step 2: 编译验证并提交**

```powershell
cmake --build build --target somegi_rhi
```

```bash
git add src/renderer/culling/hiz_build_pass.h src/renderer/culling/hiz_build_pass.cpp
git commit -m "迁移 hiz_build_pass 到纯 RHI：消除 Device& 依赖

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: rsm_geometry_pass — 消除 core::Image + 原生 barrier

**Files:**
- Modify: `src/renderer/gi/rsm/rsm_geometry_pass.h`
- Modify: `src/renderer/gi/rsm/rsm_geometry_pass.cpp`

- [ ] **Step 1: 消除 core::Device& 依赖**

在 `rsm_geometry_pass.h` 和 `.cpp` 中：
- 删除 `class Device;` 前向声明
- `init(Device& d, ...)` → `init(rhi::RHIDevice& d, ...)`（保留 RHIDevice 参数）
- 删除 `Device* m_device` 成员

- [ ] **Step 2: core::Image 替换为 RHITexture**

搜索 `core::Image(` 或 `Image(` 构造函数调用，替换为 `d.createTexture(...)` + `d.createTextureView(...)`。

- [ ] **Step 3: 原生 transition() helper 替换为 RHI barrier**

搜索 `transition(` helper 函数（内部调用 `vkCmdPipelineBarrier2`），替换为 `cmd.textureBarrier(...)`。

- [ ] **Step 4: 删除 VkCompat record() 重载**

删除 `record(VkCommandBuffer, ...)` 的声明和定义。

- [ ] **Step 5: 编译验证并提交**

```powershell
cmake --build build --target somegi_rhi
```

```bash
git add src/renderer/gi/rsm/rsm_geometry_pass.h src/renderer/gi/rsm/rsm_geometry_pass.cpp
git commit -m "迁移 rsm_geometry_pass 到纯 RHI：消除 core::Image + 原生 barrier

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: vxgi_relight_pass — 删除 VkCompat + 消除原生 vkCmd*

**Files:**
- Modify: `src/renderer/gi/vxgi/vxgi_relight_pass.h`
- Modify: `src/renderer/gi/vxgi/vxgi_relight_pass.cpp`

- [ ] **Step 1: 删除 VkCompat record() 重载**

在 `.h` 中删除 `record(VkCommandBuffer, ...)` 声明。
在 `.cpp` 中删除 VkCompat 重载的定义（包含 `vkCmdBindPipeline`、`vkCmdBindDescriptorSets`、`vkCmdPushConstants`、`vkCmdDispatch` 的版本）。

- [ ] **Step 2: 确认 RHI record() 路径完整**

检查 `record(rhi::RHICommandBuffer&, ...)` 已正确使用 RHI API。如有缺漏补齐。

- [ ] **Step 3: 编译验证并提交**

```powershell
cmake --build build --target somegi_rhi
```

```bash
git add src/renderer/gi/vxgi/vxgi_relight_pass.h src/renderer/gi/vxgi/vxgi_relight_pass.cpp
git commit -m "迁移 vxgi_relight_pass 到纯 RHI：删除 VkCompat 重载

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: vxgi_aniso_pass — 消除原生 vkCmd* + barrier

**Files:**
- Modify: `src/renderer/gi/vxgi/vxgi_aniso_pass.h`
- Modify: `src/renderer/gi/vxgi/vxgi_aniso_pass.cpp`

- [ ] **Step 1: 消除原生 vkCmd* 调用**

在 `record()` 中找到所有通过 `(VkCommandBuffer)(uintptr_t)cmd.nativeHandle()` 提取原生句柄后调用 `vkCmd*` 的代码：
- `vkCmdBindPipeline` → `cmd.bindPipelineState(*m_pso)`
- `vkCmdBindDescriptorSets` → `cmd.bindDescriptorSet(0, *m_set)`
- `vkCmdPushConstants` → `cmd.pushConstants(...)`
- `vkCmdDispatch` → `cmd.dispatch(gx, gy, gz)`
- `vkCmdPipelineBarrier2` → `cmd.globalBarrier()` 或 `cmd.textureBarrier(...)`

- [ ] **Step 2: 删除 VkCompat record() 重载**

删除 `record(VkCommandBuffer, ...)` 声明和定义。

- [ ] **Step 3: 编译验证并提交**

```powershell
cmake --build build --target somegi_rhi
```

```bash
git add src/renderer/gi/vxgi/vxgi_aniso_pass.h src/renderer/gi/vxgi/vxgi_aniso_pass.cpp
git commit -m "迁移 vxgi_aniso_pass 到纯 RHI：消除原生 vkCmd* + barrier

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: vxgi_mipmap_pass — 消除原生 vkCmdPipelineBarrier2

**Files:**
- Modify: `src/renderer/gi/vxgi/vxgi_mipmap_pass.h`
- Modify: `src/renderer/gi/vxgi/vxgi_mipmap_pass.cpp`

- [ ] **Step 1: 替换原生 barrier 为 RHI barrier**

在 `record()` 中找到 `(VkCommandBuffer)(uintptr_t)cmd.nativeHandle()` 提取 + `vkCmdPipelineBarrier2` 调用：
- 替换为 `cmd.textureBarrier(...)`（针对 image barrier）或 `cmd.globalBarrier()`

- [ ] **Step 2: 删除 VkCompat record() 重载**

删除 `record(VkCommandBuffer, ...)` 声明和定义。

- [ ] **Step 3: 编译验证并提交**

```powershell
cmake --build build --target somegi_rhi
```

```bash
git add src/renderer/gi/vxgi/vxgi_mipmap_pass.h src/renderer/gi/vxgi/vxgi_mipmap_pass.cpp
git commit -m "迁移 vxgi_mipmap_pass 到纯 RHI：消除原生 vkCmdPipelineBarrier2

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: ddgi_pass — 新增 RHI record() 路径

**Files:**
- Modify: `src/renderer/gi/ddgi/ddgi_pass.h`
- Modify: `src/renderer/gi/ddgi/ddgi_pass.cpp`

- [ ] **Step 1: 新增 record(rhi::RHICommandBuffer&, ...) 实现**

当前 `ddgi_pass.cpp` 只有 `record(VkCommandBuffer, ...)` 定义（59-81 行），需新增 RHI 版本。

将每条原生 Vulkan 调用映射到 RHI：
- `vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updatePipe)` → `cmd.bindPipelineState(*m_updatePSO)`
- `vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updateLayout, 0, 1, &vkSet, 0, 0)` → `cmd.bindDescriptorSet(0, *m_set)`
- `vkCmdPushConstants(cmd, m_updateLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DDGIConstants), &m_constants)` → `cmd.pushConstants(rhi::ShaderStage::Compute, &m_constants, sizeof(DDGIConstants))`
- `vkCmdDispatch(cmd, ...)` → `cmd.dispatch(gx, gy, gz)`
- `vkCmdPipelineBarrier2(cmd, &di)` → `cmd.globalBarrier()`

所有 4 个 compute dispatch（update、classify、blendIrr、blendDist）均需转换。

- [ ] **Step 2: 删除 VkCompat record() 重载**

删除 `record(VkCommandBuffer, ...)` 声明和定义。

- [ ] **Step 3: 编译验证并提交**

```powershell
cmake --build build --target somegi_rhi
```

```bash
git add src/renderer/gi/ddgi/ddgi_pass.h src/renderer/gi/ddgi/ddgi_pass.cpp
git commit -m "迁移 ddgi_pass 到纯 RHI：新增 RHI record() + 删除 VkCompat

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: lumen_probe_pass — 消除原生 vkCmd*（保留 RT TLAS 桥接）

**Files:**
- Modify: `src/renderer/gi/lumen/lumen_probe_pass.h`
- Modify: `src/renderer/gi/lumen/lumen_probe_pass.cpp`

- [ ] **Step 1: 消除原生 vkCmd* 调用**

在 `record()` 中找到所有 `vkCmd*` 调用，替换为 RHI 对应方法。

- [ ] **Step 2: 保留 RT TLAS 桥接**

`VkAccelerationStructureKHR` + `VkRHIAccelerationStructure::createNonOwning()` 桥接保留不变——这是正确的模式，因为 `createRayTracingPSO` 尚未实现。

- [ ] **Step 3: 删除 VkCompat record() 重载（如有）**

- [ ] **Step 4: 编译验证并提交**

```powershell
cmake --build build --target somegi_rhi
```

```bash
git add src/renderer/gi/lumen/lumen_probe_pass.h src/renderer/gi/lumen/lumen_probe_pass.cpp
git commit -m "迁移 lumen_probe_pass 到纯 RHI：消除原生 vkCmd*（RT TLAS 桥接保留）

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: lighting_pass — 消除 Device& + 删除 VkCompat + 资源改用 RHI

**Files:**
- Modify: `src/renderer/core/lighting_pass.h`
- Modify: `src/renderer/core/lighting_pass.cpp`

- [ ] **Step 1: 删除头文件中 core 依赖**

在 `lighting_pass.h` 中：
- 删除第 7 行 `#include "core/buffer.h"`
- 删除第 17 行 `class Device;`
- 第 23 行 `init(Device& d, rhi::RHIDevice& rhiDevice)` → `init(rhi::RHIDevice& d)`
- 第 26-34 行 `bindIblResources(Device& d, ...)`、`bindFrame(Device& d, ...)`、`bindShadowMask(Device& d, ...)`、`setNdgiWeights(Device& d, ...)` → 全部删除 `Device& d,` 参数
- 删除第 36 行 `void record(VkCommandBuffer cmd, const RenderTargets& rt);`
- 删除第 41 行 `Device* m_device = nullptr;`
- 第 49 行 `Buffer m_dummyBuf;` → `std::unique_ptr<rhi::RHIBuffer> m_dummyBuf;`
- 第 57 行 `Buffer m_iblParamsUbo;` → `std::unique_ptr<rhi::RHIBuffer> m_iblParamsUbo;`

- [ ] **Step 2: 修改 init() — Buffer 创建改用 RHI**

在 `lighting_pass.cpp` 中：
- 用 `rhi::RHIDevice& d` 替换 `Device& d, rhi::RHIDevice& rhiDevice` 双参数
- 删除 `m_device = &d;`
- `m_dummyBuf = Buffer(d, 4, ...)` 替换为 `d.createBuffer({.size = 4, .usage = rhi::BufferUsage::Uniform, .memoryType = rhi::MemoryType::Upload})`
- `m_iblParamsUbo = Buffer(d, sizeof(IblParams), ...)` 同样改用 RHI 创建

- [ ] **Step 3: 更新 bind* 方法签名**

`bindIblResources`、`bindFrame`、`bindShadowMask`、`setNdgiWeights` 全部删除 `Device&` 参数，更新内部实现中的 createNonOwning 桥接为直接使用 RHI 对象。

- [ ] **Step 4: 删除 VkCompat record() 重载**

删除 `record(VkCommandBuffer, ...)` 定义（~296-305 行），以及 `VkSet()`/`VkLay()` 辅助函数（如仅 VkCompat 使用）。

- [ ] **Step 5: 编译验证并提交**

```powershell
cmake --build build --target somegi_rhi
```

```bash
git add src/renderer/core/lighting_pass.h src/renderer/core/lighting_pass.cpp
git commit -m "迁移 lighting_pass 到纯 RHI：消除 Device& + 删除 VkCompat

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: 全量编译 + 调用方修复

- [ ] **Step 1: 编译并修复所有调用方错误**

```powershell
cmake --build build 2>&1 | Select-Object -Last 50
```

编译错误主要来自：
- `init(Device&, ...)` 调用方需改为 `init(RHIDevice&, ...)`
- `record(VkCommandBuffer, ...)` 调用方需改为 `record(RHICommandBuffer&, ...)`
- `bind*(Device&, ...)` 调用方需删除 `Device&` 参数

每修复一个文件即编译验证，直至零错误。

- [ ] **Step 2: 提交调用方修复**

```bash
git add -u src/
git commit -m "修复所有 Renderer Pass RHI 迁移的调用方编译错误

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 3: 最终验证**

```bash
git status
git diff --stat main..HEAD
```

Expected: 所有 pass 的 `core::Device&` 依赖已消除，`record(VkCommandBuffer)` 重载已删除。
