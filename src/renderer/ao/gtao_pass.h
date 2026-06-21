// GtaoPass —— Ground-Truth AO (Activision 2016)，已迁移到 RHI。
// 与 SsaoPass 共享输出 rt.ssao（R8），调用方 per-frame 二选一 dispatch。
#pragma once
#include "renderer/core/render_targets.h"
#include <glm/glm.hpp>
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

class GtaoPass {
public:
    ~GtaoPass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindFrame(const RenderTargets& rt);

    // RHI 路径
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt,
                const glm::mat4& proj, const glm::mat4& view);

    bool  enabled = false;
    int   sliceCount = 4;
    int   samplesPerSlice = 4;
    float radiusPixels = 32.0f;
    float falloff = 5.0f;

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_set;
};

} // namespace somegi
