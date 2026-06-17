// SmaaPass —— SMAA (Compute)，已迁移到 RHI。双 Pass: edge detection + blending。
#pragma once
#include "core/image.h"
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {
class Device;
struct RenderTargets;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }

class SmaaPass {
public:
    ~SmaaPass();
    void init(Device& dev, rhi::RHIDevice& d, VkExtent2D ext);
    void destroy();
    void bindResources(const RenderTargets& rt);
    void bindOutput(VkImageView outView);
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt);
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

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

    Image m_edgeTex;
    VkImageLayout m_edgeLayout = VK_IMAGE_LAYOUT_UNDEFINED;
};

} // namespace somegi
