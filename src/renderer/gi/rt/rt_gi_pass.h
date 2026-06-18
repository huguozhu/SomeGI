// RtGiPass — 硬件 RT GI (Ray Query, Compute)，已迁移到 RHI。
#pragma once
#include "rhi/base/sampler.h"
#include "rhi/base/texture.h"
#include "renderer/core/render_targets.h"
#include "renderer/gi/rt/scene_rt_as.h"
#include <memory>
#include <vulkan/vulkan.h>
namespace somegi { struct SceneGpu;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }
class RtGiPass {
public:
    ~RtGiPass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindFrame(const RenderTargets& rt, VkBuffer frameUbo, const SceneRtAS& rtAS, const SceneGpu& sceneGpu);
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt);
    void record(VkCommandBuffer cmd, const RenderTargets& rt);
private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
    std::unique_ptr<rhi::RHISampler> m_linearClamp;
    std::vector<std::unique_ptr<rhi::RHITextureView>> m_texViews;
    std::vector<const rhi::RHITextureView*> m_texViewPtrs;
};
} // namespace somegi
