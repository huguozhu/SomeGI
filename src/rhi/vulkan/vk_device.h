// rhi/vulkan/vk_device.h — RHIDevice 的 Vulkan 实现
#pragma once
#include "../device.h"
#include <VkBootstrap.h>
#include <VulkanMemoryAllocator/vk_mem_alloc.h>

namespace somegi {
namespace rhi {

class VkRHIDevice : public RHIDevice {
public:
    VkRHIDevice(void* nativeWindowHandle, bool enableValidation);
    ~VkRHIDevice() override;

    Backend backend() const override { return Backend::Vulkan; }
    DeviceLimits limits() const override;

    // 资源创建
    std::unique_ptr<RHIBuffer> createBuffer(const BufferDesc& desc) override;
    std::unique_ptr<RHITexture> createTexture(const TextureDesc& desc) override;
    std::unique_ptr<RHITextureView> createTextureView(const RHITexture& tex, const TextureViewDesc& desc) override;
    std::unique_ptr<RHIShader> createShader(const ShaderDesc& desc, const void* bytecode, size_t size) override;
    std::unique_ptr<RHISwapchain> createSwapchain(void* nativeWindow, uint32_t width, uint32_t height) override;
    std::unique_ptr<RHIPipelineState> createGraphicsPSO(const GraphicsPSODesc& desc) override;
    std::unique_ptr<RHIPipelineState> createComputePSO(const ComputePSODesc& desc) override;
    std::unique_ptr<RHIDescriptorSetLayout> createDescriptorSetLayout(const DescSetLayoutDesc& desc) override;
    std::unique_ptr<RHIDescriptorSet> createDescriptorSet(const RHIDescriptorSetLayout& layout) override;
    std::unique_ptr<RHICommandPool> createCommandPool() override;
    std::unique_ptr<RHIFence> createFence(bool signaled) override;
    std::unique_ptr<RHISemaphore> createSemaphore() override;
    std::unique_ptr<RHIQueryPool> createQueryPool(uint32_t count) override;

    void waitIdle() override;
    void* nativeDevice() const override { return (void*)m_device.device; }
    void* nativePhysicalDevice() const override { return (void*)m_physicalDevice.physical_device; }
    void* nativeQueue() const override { return (void*)m_graphicsQueue; }
    uint32_t queueFamily() const override { return m_graphicsQueueFamily; }

    // 内部使用（Vulkan 后端各实现类需要）
    VkDevice vkDevice() const { return m_device.device; }
    VkInstance vkInstance() const { return m_instance.instance; }
    VkPhysicalDevice vkPhysicalDevice() const { return m_physicalDevice.physical_device; }
    VkQueue vkQueue() const { return m_graphicsQueue; }
    VmaAllocator vma() const { return m_allocator; }
    const vkb::DispatchTable& dispatch() const { return m_dispatch; }
    VkSurfaceKHR surface() const { return m_surface; }

private:
    vkb::Instance m_instance{};
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    vkb::PhysicalDevice m_physicalDevice{};
    vkb::Device m_device{};
    vkb::DispatchTable m_dispatch{};
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueFamily = 0;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    DeviceLimits m_limits{};
};

} // namespace rhi
} // namespace somegi
