// TonemapPass —— ACES tonemap (Compute)，已迁移到 RHI。
#pragma once
#include "rhi/base/sampler.h"
#include "renderer/core/render_targets.h"
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {

namespace rhi {
class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState;
class RHIDescriptorSet; class RHICommandBuffer;
}

class TonemapPass {
public:
    ~TonemapPass();
    void init(rhi::RHIDevice& d, VkSampler linearSampler);
    void init(rhi::RHIDevice& d, rhi::RHISampler& linearSampler); // D3D12 路径
    void destroy();
    void bindTargets(const RenderTargets& rt);
    void bindOutput(VkImageView outView, uint32_t frameIdx);
    auto& sets() { return m_sets; }
    auto* pipeline() { return m_pipeline.get(); } // D3D12 手动 bind
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt, uint32_t frameIdx,
                bool hdrMode = false, float exposure = 1.0f);


private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_sets[2];
    std::unique_ptr<rhi::RHISampler> m_sampler;
};

} // namespace somegi
