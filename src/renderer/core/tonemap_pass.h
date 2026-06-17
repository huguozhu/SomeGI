// TonemapPass —— ACES tonemap (Compute)，已迁移到 RHI。
#pragma once
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
    void destroy();
    void bindTargets(const RenderTargets& rt);
    void bindOutput(VkImageView outView, uint32_t frameIdx);
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt, uint32_t frameIdx,
                bool hdrMode = false, float exposure = 1.0f);
    void record(VkCommandBuffer cmd, const RenderTargets& rt, uint32_t frameIdx,
                bool hdrMode = false, float exposure = 1.0f);

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_sets[2];
    VkSampler m_sampler = VK_NULL_HANDLE;
};

} // namespace somegi
