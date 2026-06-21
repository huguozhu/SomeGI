// LumenGatherPass — Final Gather (Compute), 已迁移到 RHI。
#pragma once
#include "rhi/base/sampler.h"
#include <memory>
#include <vulkan/vulkan.h>
namespace somegi { class LumenResources; struct RenderTargets;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }
class LumenGatherPass {
public:
    ~LumenGatherPass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindResources(const LumenResources& res, const RenderTargets& rt, VkBuffer frameUbo, bool useFiltered);
    void record(rhi::RHICommandBuffer& cmd, const LumenResources& res, const RenderTargets& rt, uint32_t debugMode=0);
private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
    std::unique_ptr<rhi::RHISampler> m_pointClamp;
}; } // namespace somegi
