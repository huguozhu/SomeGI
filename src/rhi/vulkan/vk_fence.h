// rhi/vulkan/vk_fence.h
#pragma once
#include "../base/fence.h"
#include "vk_device.h"

namespace somegi {
namespace rhi {

class VkRHIFence : public RHIFence {
public:
    static std::unique_ptr<RHIFence> create(VkRHIDevice& device, bool signaled);
    static std::unique_ptr<RHIFence> createNonOwning(VkRHIDevice& device, VkFence fence) {
        auto f = std::unique_ptr<VkRHIFence>(new VkRHIFence(device));
        f->m_fence = fence; f->m_owns = false; return f;
    }
    ~VkRHIFence() override;
    void wait(uint64_t timeoutNs) override;
    void reset() override;
    void* nativeHandle() const override { return (void*)m_fence; }
private:
    VkRHIDevice& m_device;
    VkFence m_fence = VK_NULL_HANDLE;
    bool m_owns = true;
    VkRHIFence(VkRHIDevice& d) : m_device(d) {}
};

class VkRHISemaphore : public RHISemaphore {
public:
    static std::unique_ptr<RHISemaphore> create(VkRHIDevice& device);
    static std::unique_ptr<RHISemaphore> createNonOwning(VkRHIDevice& device, VkSemaphore sem) {
        auto s = std::unique_ptr<VkRHISemaphore>(new VkRHISemaphore(device));
        s->m_semaphore = sem;
        s->m_owns = false;
        return s;
    }
    ~VkRHISemaphore() override;
    void* nativeHandle() const override { return (void*)m_semaphore; }
private:
    VkRHIDevice& m_device;
    VkSemaphore m_semaphore = VK_NULL_HANDLE;
    bool m_owns = true;
    VkRHISemaphore(VkRHIDevice& d) : m_device(d) {}
};

} // namespace rhi
} // namespace somegi
