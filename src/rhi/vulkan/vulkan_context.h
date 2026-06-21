// rhi/vulkan/vulkan_context.h
#pragma once
#include "../base/context.h"
#include "vk_device.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

namespace somegi {
namespace rhi {

class VkRHIDevice;

class VulkanContext : public RHIContext {
public:
    // 构造时创建命令池、命令缓冲区、fence（kFramesInFlight 组）
    VulkanContext(VkRHIDevice& device, uint32_t framesInFlight);
    ~VulkanContext() override;

    RHICommandBuffer& beginFrame(uint32_t frameIndex) override;
    void endFrame(uint32_t frameIndex,
                  const RHISemaphore* waitSemaphore,
                  const RHISemaphore* signalSemaphore) override;
    RHICommandBuffer& commandBuffer(uint32_t frameIndex) override;
    void waitIdle() override;
    RHIDevice& device() override { return m_device; }

    // Vulkan 专用访问器（ImGui 等需要原生句柄 / 尚未迁移到 RHI 的调用方）
    VkCommandBuffer vkCommandBuffer(uint32_t frameIndex) const;
    VkCommandPool vkCommandPool() const { return m_pool; }
    // 直接使用 VkSemaphore 的 endFrame 重载（swapchain 尚未迁移到 RHI）
    void endFrame(uint32_t frameIndex, VkSemaphore waitSem, VkSemaphore signalSem);

private:
    VkRHIDevice& m_device;
    uint32_t m_framesInFlight;
    VkCommandPool m_pool = VK_NULL_HANDLE;

    struct FrameResources {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool fenceSignaled = true;  // 首帧 fence 已 signaled
    };
    std::vector<FrameResources> m_frames;

    // RHI 命令缓冲区包装（临时对象，valid until next beginFrame）
    std::unique_ptr<RHICommandBuffer> m_tempCmd;

    // 内部分配 + 清理
    void createResources();
    void destroyResources();
};

} // namespace rhi
} // namespace somegi
