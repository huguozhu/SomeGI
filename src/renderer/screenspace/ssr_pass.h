// SsrPass —— 屏幕空间反射（M4.2），已迁移到 RHI。
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
class RHISampler;
}

class SsrPass {
public:
    SsrPass() = default;
    ~SsrPass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindFrame(const RenderTargets& rt, VkBuffer frameUbo);

    // RHI 路径
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt);
    // 兼容 VkCommandBuffer
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

    bool  enabled         = true;
    int   maxSteps        = 32;
    float maxDist         = 50.0f;
    float thickness       = 0.05f;
    float roughThreshold  = 0.7f;

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;

    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_set;

    std::unique_ptr<rhi::RHISampler> m_linearClamp;
};

} // namespace somegi
