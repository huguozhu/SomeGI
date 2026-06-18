// rhi/vulkan/vk_sampler.h — RHISampler 的 Vulkan 实现
#pragma once
#include "../base/sampler.h"
#include "vk_device.h"
#include <vulkan/vulkan.h>

namespace somegi {
namespace rhi {

class VkRHISampler : public RHISampler {
public:
    static std::unique_ptr<RHISampler> create(VkRHIDevice& device, const SamplerDesc& desc);
    // 非拥有型包装（用于现有 VkSampler 的过渡，不管理生命周期）
    static std::unique_ptr<RHISampler> createNonOwning(VkRHIDevice& device, VkSampler sampler) {
        auto s = std::unique_ptr<VkRHISampler>(new VkRHISampler(device));
        s->m_ownsSampler = false;
        s->m_sampler = sampler;
        return s;
    }
    ~VkRHISampler() override;
    void* nativeHandle() const override { return (void*)m_sampler; }
private:
    VkRHIDevice& m_device;
    VkSampler m_sampler = VK_NULL_HANDLE;
    bool m_ownsSampler = true;
    VkRHISampler(VkRHIDevice& d) : m_device(d) {}
};

} // namespace rhi
} // namespace somegi
