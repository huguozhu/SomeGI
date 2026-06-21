// LumenFilterPass — SH9 spatial+temporal filter (Compute), 已迁移到 RHI。
#pragma once
#include "rhi/base/sampler.h"
#include <memory>
#include <vulkan/vulkan.h>
namespace somegi { class LumenResources; struct RenderTargets;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }
class LumenFilterPass {
public:
    ~LumenFilterPass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindResources(const LumenResources& res, const RenderTargets& rt, VkBuffer frameUbo);
    void record(rhi::RHICommandBuffer& cmd, const LumenResources& res, const RenderTargets& rt);
    float sigmaDepth=0.3f, normalPower=8.f, sigmaDist=200.f, temporalAlpha=0.95f;
private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
    std::unique_ptr<rhi::RHISampler> m_pointClamp;
}; } // namespace somegi
