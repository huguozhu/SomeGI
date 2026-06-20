// VxgiRelightPass — 体素 bounce relight (Compute, sampler)，已迁移到 RHI。
#pragma once
#include "rhi/base/sampler.h"
#include "rhi/base/descriptor.h"
#include "core/vk_common.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>
namespace somegi {
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }

class VxgiRelightPass {
public:
    ~VxgiRelightPass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindResources(const VxgiResources& vxgi, VkImageView dstMip0View);
    void bindResourcesPingPong(const VxgiResources& vxgi, bool swap);
    void record(rhi::RHICommandBuffer& cmd, const rhi::RHIDescriptorSet& set, uint32_t gr, uint32_t ml, float cs, const glm::vec3& gm, float bs);
    void record(VkCommandBuffer cmd, VkDescriptorSet set, uint32_t gr, uint32_t ml, float cs, const glm::vec3& gm, float bs);

    // Vk 兼容访问器（迁移期间，调用方仍使用 VkCommandBuffer 路径）
    VkDescriptorSet voxelSet() const { return (VkDescriptorSet)(uintptr_t)m_set->nativeHandle(); }
    VkDescriptorSet pingSet0() const { return (VkDescriptorSet)(uintptr_t)m_setPP0->nativeHandle(); }
    VkDescriptorSet pingSet1() const { return (VkDescriptorSet)(uintptr_t)m_setPP1->nativeHandle(); }

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set, m_setPP0, m_setPP1;
    std::unique_ptr<rhi::RHISampler> m_linearClamp;
};
} // namespace somegi
