// rhi/vulkan/vk_device.h — RHIDevice 的 Vulkan 实现
#pragma once
#include "../base/device.h"
#include <VkBootstrap.h>
#include <VulkanMemoryAllocator/vk_mem_alloc.h>

namespace somegi {
namespace rhi {

class VkRHIDevice : public RHIDevice {
public:
    // 独立创建（拥有所有 Vulkan 句柄，构造时初始化整个 Vulkan 环境）
    VkRHIDevice(void* nativeWindowHandle, bool enableValidation);

    // 从已有 core::Device 共享句柄（不拥有 Vulkan 对象，析构时不销毁）
    VkRHIDevice(VkInstance instance, VkPhysicalDevice physicalDevice,
                VkDevice device, VmaAllocator allocator,
                VkQueue queue, uint32_t queueFamily,
                const vkb::DispatchTable& dispatch,
                const DeviceLimits& limits);

    ~VkRHIDevice() override;

    Backend backend() const override { return Backend::Vulkan; }
    DeviceLimits limits() const override;

    // 资源创建
    std::unique_ptr<RHIBuffer> createBuffer(const BufferDesc& desc) override;
    std::unique_ptr<RHITexture> createTexture(const TextureDesc& desc) override;
    std::unique_ptr<RHITextureView> createTextureView(const RHITexture& tex, const TextureViewDesc& desc) override;
    std::unique_ptr<RHIShader> createShader(const ShaderDesc& desc, const void* bytecode, size_t size) override;
    std::unique_ptr<RHISampler> createSampler(const SamplerDesc& desc) override;
    std::unique_ptr<RHISwapchain> createSwapchain(void* nativeWindow, uint32_t width, uint32_t height) override;
    std::unique_ptr<RHIPipelineState> createGraphicsPSO(const GraphicsPSODesc& desc) override;
    std::unique_ptr<RHIPipelineState> createComputePSO(const ComputePSODesc& desc) override;
    std::unique_ptr<RHIPipelineState> createRayTracingPSO(const RayTracingPSODesc& desc) override;
    std::unique_ptr<RHIDescriptorSetLayout> createDescriptorSetLayout(const DescSetLayoutDesc& desc) override;
    std::unique_ptr<RHIDescriptorSet> createDescriptorSet(const RHIDescriptorSetLayout& layout) override;
    std::unique_ptr<RHICommandPool> createCommandPool() override;
    std::unique_ptr<RHIFence> createFence(bool signaled) override;
    std::unique_ptr<RHISemaphore> createSemaphore() override;
    std::unique_ptr<RHIQueryPool> createQueryPool(uint32_t count) override;

    void submit(const SubmitDesc& desc) override;
    void present(const RHISwapchain& swapchain, const RHISemaphore* waitSemaphore) override;
    void waitForFence(const RHIFence& fence, uint64_t timeoutNs) override;
    void waitIdle() override;
    void* nativeDevice() const override { return (void*)m_vkDevice; }
    void* nativePhysicalDevice() const override { return (void*)m_vkPhysicalDevice; }
    void* nativeQueue() const override { return (void*)m_graphicsQueue; }
    uint32_t queueFamily() const override { return m_graphicsQueueFamily; }

    // 内部使用（Vulkan 后端各实现类需要）
    VkDevice vkDevice() const { return m_vkDevice; }
    VkInstance vkInstance() const { return m_vkInstance; }
    VkPhysicalDevice vkPhysicalDevice() const { return m_vkPhysicalDevice; }
    VkQueue vkQueue() const { return m_graphicsQueue; }
    VmaAllocator vma() const { return m_allocator; }
    const vkb::DispatchTable& dispatch() const { return m_dispatch; }
    VkSurfaceKHR surface() const { return m_surface; }

private:
    bool m_ownsDevice = true;  // 是否拥有 Vulkan 句柄（决定析构时是否销毁）

    // 原生 Vulkan 句柄（两种构造方式下都有效）
    VkInstance m_vkInstance = VK_NULL_HANDLE;
    VkPhysicalDevice m_vkPhysicalDevice = VK_NULL_HANDLE;
    VkDevice m_vkDevice = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueFamily = 0;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    DeviceLimits m_limits{};
    vkb::DispatchTable m_dispatch{};

    // 仅在 m_ownsDevice=true 时需要的额外句柄
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
};

} // namespace rhi
} // namespace somegi
