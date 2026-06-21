// SkyboxPass —— Graphics PSO 天空盒，已迁移到 RHI（首个 Graphics PSO 验证）。
#pragma once
#include "rhi/base/sampler.h"
#include "renderer/core/render_targets.h"
#include <glm/glm.hpp>
#include <memory>

namespace somegi {

namespace rhi {
class RHIDevice;
class RHIDescriptorSetLayout;
class RHIPipelineState;
class RHIDescriptorSet;
class RHICommandBuffer;
class RHIBuffer;
class RHITextureView;
}

class SkyboxPass {
public:
    ~SkyboxPass();
    void init(rhi::RHIDevice& d, rhi::Format colorFmt, rhi::Format depthFmt);
    void destroy();

    void bindEnv(const rhi::RHITextureView& envCubeView, const rhi::RHISampler& linearSampler);
    void bindEnvRHI(const rhi::RHITextureView& envCubeView, const rhi::RHISampler& linearSampler);
    void updateFrame(const glm::mat4& invViewProj, const glm::vec3& cameraPos);

    // RHI 路径
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt);

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;

    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_set;

    std::unique_ptr<rhi::RHIBuffer> m_ubo;
    rhi::Format m_colorFmt = rhi::Format::Unknown;
    rhi::Format m_depthFmt = rhi::Format::Unknown;
};

} // namespace somegi
