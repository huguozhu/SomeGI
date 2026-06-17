// rhi/vulkan/vk_descriptor.h
#pragma once
#include "../descriptor.h"
#include "vk_device.h"
#include <vulkan/vulkan.h>

namespace somegi {
namespace rhi {

class VkRHIDescSetLayout : public RHIDescriptorSetLayout {
public:
    static std::unique_ptr<RHIDescriptorSetLayout> create(VkRHIDevice& device, const DescSetLayoutDesc& desc);
    ~VkRHIDescSetLayout() override;
    void* nativeHandle() const override { return (void*)m_layout; }
private:
    VkRHIDevice& m_device;
    VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
    VkRHIDescSetLayout(VkRHIDevice& d) : m_device(d) {}
};

class VkRHIDescSet : public RHIDescriptorSet {
public:
    static std::unique_ptr<RHIDescriptorSet> create(VkRHIDevice& device, const RHIDescriptorSetLayout& layout);
    ~VkRHIDescSet() override;
    void write(const std::vector<DescriptorWrite>& writes) override;
    void* nativeHandle() const override { return (void*)m_set; }
private:
    VkRHIDevice& m_device;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkRHIDescSet(VkRHIDevice& d) : m_device(d) {}
};

} // namespace rhi
} // namespace somegi
