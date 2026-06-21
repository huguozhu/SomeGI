// rhi/vulkan/vk_fence.cpp
#include "vk_fence.h"

namespace somegi {
namespace rhi {

std::unique_ptr<RHIFence> VkRHIFence::create(VkRHIDevice& device, bool signaled) {
    auto f = std::unique_ptr<VkRHIFence>(new VkRHIFence(device));
    VkFenceCreateInfo ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (signaled) ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(device.vkDevice(), &ci, nullptr, &f->m_fence);
    return f;
}
VkRHIFence::~VkRHIFence() { if (m_owns && m_fence) vkDestroyFence(m_device.vkDevice(), m_fence, nullptr); }
void VkRHIFence::wait(uint64_t timeoutNs) { vkWaitForFences(m_device.vkDevice(), 1, &m_fence, VK_TRUE, timeoutNs); }
void VkRHIFence::reset() { vkResetFences(m_device.vkDevice(), 1, &m_fence); }

std::unique_ptr<RHISemaphore> VkRHISemaphore::create(VkRHIDevice& device) {
    auto s = std::unique_ptr<VkRHISemaphore>(new VkRHISemaphore(device));
    VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(device.vkDevice(), &ci, nullptr, &s->m_semaphore);
    return s;
}
VkRHISemaphore::~VkRHISemaphore() { if (m_owns && m_semaphore) vkDestroySemaphore(m_device.vkDevice(), m_semaphore, nullptr); }

} // namespace rhi
} // namespace somegi
