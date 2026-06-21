// SmaaPass —— SMAA (Compute)，已迁移到纯 RHI。双 Pass: edge detection + blending。
#pragma once
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {
struct RenderTargets;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; class RHITexture; class RHITextureView; }

class SmaaPass {
public:
    ~SmaaPass();
    void init(rhi::RHIDevice& d, VkExtent2D ext);
    void destroy();
    void bindResources(const RenderTargets& rt);
    void bindOutput(VkImageView outView);
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt);

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;

    // Edge detection
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_edgeSetLayout;
    std::unique_ptr<rhi::RHIPipelineState>       m_edgePipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_edgeSet;

    // Blending
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_blendSetLayout;
    std::unique_ptr<rhi::RHIPipelineState>       m_blendPipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_blendSet;

    std::unique_ptr<rhi::RHITexture>     m_edgeTex;
    std::unique_ptr<rhi::RHITextureView> m_edgeView;
};

} // namespace somegi
