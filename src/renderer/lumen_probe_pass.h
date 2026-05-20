#pragma once
#include "core/vk_common.h"
#include "render_targets.h"
#include "lumen_resources.h"
#include <glm/glm.hpp>

// LumenProbePass —— L.2 Screen Probe pass。
//
// 两个 compute dispatch：
//   1. cs_generateRays：TLAS RayQuery + voxelGrid 采样 → ray buffer
//   2. cs_projectSH：ray buffer → SH9 → probe atlas
//
// 调用方负责在调用 record 前保证：
//   - TLAS / scene SSBO 已构建
//   - GBuffer SHADER_READ_ONLY
//   - voxelGrid mip 0 SHADER_READ_ONLY
//   - probeAtlas 在 GENERAL（首次 transition 由外部一次性完成）

namespace somegi {
class Device;
class SceneRtAS;
struct SceneGpu;
class VxgiResources;

class LumenProbePass {
public:
    void init(Device& d);
    void destroy();

    // 一次性 bind（per-scene 或 per-resize）
    void bindResources(Device& d, const LumenResources& res,
                       const SceneRtAS& rtAS, const SceneGpu& sceneGpu,
                       const VxgiResources& vxgi, const RenderTargets& rt,
                       VkBuffer frameUbo);

    // 每帧录制两个 dispatch，中间加 pipeline barrier
    void record(VkCommandBuffer cmd, const LumenResources& res,
                uint32_t frameIndex);

private:
    Device* m_device = nullptr;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipelineRays = VK_NULL_HANDLE;
    VkPipeline m_pipelineSH = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkSampler m_linearClamp = VK_NULL_HANDLE;
};

}
