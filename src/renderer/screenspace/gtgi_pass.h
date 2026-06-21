// GtgiPass —— Ground-Truth GI，已迁移到 RHI。与 SsgiPass 共享 rt.ssgi 输出。
#pragma once
#include "rhi/base/sampler.h"
#include "renderer/core/render_targets.h"
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }

class GtgiPass {
public:
    ~GtgiPass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindFrame(const RenderTargets& rt, VkBuffer frameUbo);
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt);

    bool enabled = false;
    int sliceCount = 4, samplesPerSlice = 6;
    float radiusPixels = 32.0f, falloff = 5.0f;

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
    std::unique_ptr<rhi::RHISampler> m_linearClamp;
};

}
