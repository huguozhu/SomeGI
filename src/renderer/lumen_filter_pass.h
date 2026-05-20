#pragma once
#include "core/vk_common.h"

// LumenFilterPass —— L.4 spatial + temporal SH9 filter。
//
// 一个 compute dispatch：5×5 spatial bilateral + prevAtlas reprojection temporal blend。
// 消除 screen probe 间 discontinuity 和帧间 flicker。

namespace somegi {
class Device;
class LumenResources;
struct RenderTargets;

class LumenFilterPass {
public:
    void init(Device& d);
    void destroy();

    void bindResources(Device& d, const LumenResources& res,
                       const RenderTargets& rt, VkBuffer frameUbo);

    void record(VkCommandBuffer cmd, const LumenResources& res,
                const RenderTargets& rt);

    // ImGui-tweakable
    float sigmaDepth    = 0.3f;
    float normalPower   = 8.0f;
    float sigmaDist     = 200.0f;
    float temporalAlpha = 0.95f;   // blend factor: 0=all new, 1=all prev

private:
    Device* m_device = nullptr;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkSampler m_pointClamp = VK_NULL_HANDLE;
};

}
