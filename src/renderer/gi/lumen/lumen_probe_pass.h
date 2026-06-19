// LumenProbePass — L.2 Screen Probe pass (2 compute)，已迁移到 RHI。
#pragma once
#include "rhi/base/device.h"
#include "rhi/base/sampler.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "core/vk_common.h"
#include "renderer/core/render_targets.h"
#include "renderer/gi/lumen/lumen_resources.h"
#include <glm/glm.hpp>

namespace somegi {
class Device;
class SceneRtAS;
struct SceneGpu;
class VxgiResources;

class LumenProbePass {
public:
    void init(rhi::RHIDevice& d);
    void destroy();

    void bindResources(const LumenResources& res, const SceneRtAS& rtAS,
                       const SceneGpu& sceneGpu, const VxgiResources& vxgi,
                       const RenderTargets& rt, VkBuffer frameUbo, bool hasSixAxis);

    void record(VkCommandBuffer cmd, const LumenResources& res,
                uint32_t frameIndex, bool useSixAxis);

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;

    std::unique_ptr<rhi::RHISampler> m_linearClamp;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
    std::unique_ptr<rhi::RHIPipelineState> m_pipelineRays;
    std::unique_ptr<rhi::RHIPipelineState> m_pipelineSH;
};

} // namespace somegi
