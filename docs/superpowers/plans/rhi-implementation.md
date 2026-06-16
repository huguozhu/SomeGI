# RHI 实施计划

> 基于 `docs/superpowers/specs/rhi-design.md` 的 Phase 分步实施

**分支**：`rhi`

---

## Phase 1: RHI 核心抽象 + Vulkan thin wrapper

### Task 1.0: 目录和 CMake 骨架

- [ ] 创建 `src/rhi/CMakeLists.txt`
- [ ] 创建 `src/rhi/common.h` — 通用类型定义
- [ ] 修改 `src/CMakeLists.txt` — 添加 `add_subdirectory(rhi)`

### Task 1.1: RHIDevice

- [ ] `src/rhi/device.h` — RHIDevice 接口声明
- [ ] `src/rhi/vulkan/vk_device.h` — Vulkan 实现（复用现有 Device 内部）
- [ ] `src/rhi/vulkan/vk_device.cpp` — RHIDevice::create(Vulkan, ...)

### Task 1.2: RHIBuffer + RHITexture

- [ ] `src/rhi/buffer.h` — BufferDesc, RHIBuffer
- [ ] `src/rhi/vulkan/vk_buffer.h/cpp` — 封装 VkBuffer + VMA
- [ ] `src/rhi/texture.h` — TextureDesc, RHITexture, RHITextureView
- [ ] `src/rhi/vulkan/vk_texture.h/cpp` — 封装 VkImage + VkImageView

### Task 1.3: RHIShader + RHISwapchain

- [ ] `src/rhi/shader.h` — ShaderDesc, ShaderStage
- [ ] `src/rhi/vulkan/vk_shader.h/cpp` — 封装 VkShaderModule
- [ ] `src/rhi/swapchain.h` — SwapchainDesc, RHISwapchain
- [ ] `src/rhi/vulkan/vk_swapchain.h/cpp`

### Task 1.4: 验证编译

```bash
cmake --build build --target somegi_rhi
```

---

## Phase 2: Pipeline State + Descriptor

### Task 2.0: RHIPipelineState

- [ ] `src/rhi/pipeline_state.h` — GraphicsPSODesc, ComputePSODesc
- [ ] `src/rhi/vulkan/vk_pso.h/cpp` — VkPipeline + VkPipelineLayout 创建
- [ ] GBuffer PSO 迁移为测试用例

### Task 2.1: RHIDescriptorSet

- [ ] `src/rhi/descriptor.h` — DescriptorSetLayout, DescriptorSet
- [ ] `src/rhi/vulkan/vk_descriptor.h/cpp`
- [ ] SSAO descriptor set 迁移

### Task 2.2: 验证

- [ ] 现有 GPU 输出与 Phase 1 前一致

---

## Phase 3: Command Buffer

### Task 3.0: RHICommandBuffer

- [ ] `src/rhi/command_buffer.h`
- [ ] `src/rhi/vulkan/vk_command.h/cpp` — VkCommandBuffer 封装
- [ ] 迁移 1 个简单 compute pass（如 SSAO）验证

### Task 3.1: 全 Pass 迁移

- [ ] 所有 compute pass → RHICommandBuffer
- [ ] 所有 graphics pass → RHICommandBuffer

---

## Phase 4: D3D12 后端（未来）

- [ ] SPIRV-Cross → HLSL → DXIL 工具链
- [ ] D3D12 device/buffer/texture
- [ ] Root signature → DescriptorSetLayout 映射
- [ ] 描述符堆 ring-buffer 管理
- [ ] 资源状态追踪器
- [ ] Win32 Window + Swapchain

---

## Phase 5: Metal 后端（未来）

- [ ] SPIRV-Cross → MSL 工具链
- [ ] Metal device/buffer/texture
- [ ] Argument buffer → DescriptorSetLayout 映射
- [ ] Render pass 适配
- [ ] macOS/iOS 窗口集成

---

## 当前状态

- [x] RHI 设计文档完成
- [ ] Phase 1 未开始
