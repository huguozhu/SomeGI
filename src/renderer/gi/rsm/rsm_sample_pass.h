// RsmSamplePass — RSM 采样间接光 (Compute)，9 bindings，已迁移到 RHI。
#pragma once
#include "renderer/core/render_targets.h"
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {
class Device; class Image;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }

class RsmSamplePass {
public:
    ~RsmSamplePass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindFrame(const RenderTargets& rt, VkBuffer frameUbo, VkBuffer rsmUbo,
                   const Image& rsmPos, const Image& rsmN, const Image& rsmFlux);
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt);
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

    bool enabled=true; int sampleCount=32; float radius=0.05f, intensity=1.0f;

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
    VkSampler m_linearClamp = VK_NULL_HANDLE;
};

} // namespace somegi
