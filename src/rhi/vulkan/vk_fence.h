// rhi/vulkan/vk_fence.h
#pragma once
#include "../base/fence.h"
#include "vk_device.h"

namespace somegi {
namespace rhi {

class VkRHIFence : public RHIFence {
public:
    static std::unique_ptr<RHIFence> create(VkRHIDevice& device, bool signaled);
    ~VkRHIFence() override;
    void wait(uint64_t timeoutNs) override;
    void reset() override;
    void* nativeHandle() const override { return (void*)m_fence; }
private:
    VkRHIDevice& m_device;
    VkFence m_fence = VK_NULL_HANDLE;
    VkRHIFence(VkRHIDevice& d) : m_device(d) {}
};

class VkRHISemaphore : public RHISemaphore {
public:
    static std::unique_ptr<RHISemaphore> create(VkRHIDevice& device);
    ~VkRHISemaphore() override;
    void* nativeHandle() const override { return (void*)m_semaphore; }
private:
    VkRHIDevice& m_device;
    VkSemaphore m_semaphore = VK_NULL_HANDLE;
    VkRHISemaphore(VkRHIDevice& d) : m_device(d) {}
};

} // namespace rhi
} // namespace somegi
