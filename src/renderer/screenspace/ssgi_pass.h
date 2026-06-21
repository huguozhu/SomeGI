// SsgiPass —— 屏幕空间一次反弹漫反射（M4.3），已迁移到 RHI。
#pragma once
#include "rhi/base/sampler.h"
#include "renderer/core/render_targets.h"
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {

namespace rhi {
class RHIDevice;
class RHIDescriptorSetLayout;
class RHIPipelineState;
class RHIDescriptorSet;
class RHICommandBuffer;
}

class SsgiPass {
public:
    ~SsgiPass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindFrame(const RenderTargets& rt, VkBuffer frameUbo);
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt);

    bool  enabled     = true;
    int   sampleCount = 8;
    int   maxSteps    = 24;
    float maxDist     = 10.0f;
    float thickness   = 0.05f;

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_set;
    std::unique_ptr<rhi::RHISampler> m_linearClamp;
};

}
