// rhi/device.h — RHIDevice 接口
#pragma once
#include "common.h"
#include <memory>

namespace somegi {
namespace rhi {

class RHIBuffer;
class RHITexture;
class RHITextureView;
class RHIShader;
class RHISwapchain;
class RHIPipelineState;
class RHIDescriptorSetLayout;
class RHIDescriptorSet;
class RHICommandPool;
class RHICommandBuffer;
class RHIFence;
class RHISemaphore;
class RHIQueryPool;

struct GraphicsPSODesc;
struct ComputePSODesc;
struct DescSetLayoutDesc;

// ════════════════════════════════════════════════════════════════
// RHIDevice — 设备抽象
// ════════════════════════════════════════════════════════════════
class RHIDevice {
public:
    static std::unique_ptr<RHIDevice> createVulkan(void* nativeWindowHandle, bool enableValidation);

    virtual ~RHIDevice() = default;
    virtual Backend backend() const = 0;
    virtual DeviceLimits limits() const = 0;

    // 资源创建
    virtual std::unique_ptr<RHIBuffer> createBuffer(const BufferDesc& desc) = 0;
    virtual std::unique_ptr<RHITexture> createTexture(const TextureDesc& desc) = 0;
    virtual std::unique_ptr<RHITextureView> createTextureView(const RHITexture& tex, const TextureViewDesc& desc) = 0;
    virtual std::unique_ptr<RHIShader> createShader(const ShaderDesc& desc, const void* bytecode, size_t size) = 0;
    virtual std::unique_ptr<RHISwapchain> createSwapchain(void* nativeWindow, uint32_t width, uint32_t height) = 0;
    virtual std::unique_ptr<RHIPipelineState> createGraphicsPSO(const GraphicsPSODesc& desc) = 0;
    virtual std::unique_ptr<RHIPipelineState> createComputePSO(const ComputePSODesc& desc) = 0;
    virtual std::unique_ptr<RHIDescriptorSetLayout> createDescriptorSetLayout(const DescSetLayoutDesc& desc) = 0;
    virtual std::unique_ptr<RHIDescriptorSet> createDescriptorSet(const RHIDescriptorSetLayout& layout) = 0;
    virtual std::unique_ptr<RHICommandPool> createCommandPool() = 0;
    virtual std::unique_ptr<RHIFence> createFence(bool signaled = false) = 0;
    virtual std::unique_ptr<RHISemaphore> createSemaphore() = 0;
    virtual std::unique_ptr<RHIQueryPool> createQueryPool(uint32_t count) = 0;

    // 同步
    virtual void waitIdle() = 0;

    // 原生句柄
    virtual void* nativeDevice() const = 0;
    virtual void* nativePhysicalDevice() const = 0;
    virtual void* nativeQueue() const = 0;
    virtual uint32_t queueFamily() const = 0;
};

} // namespace rhi
} // namespace somegi
