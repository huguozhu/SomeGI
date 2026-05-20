#pragma once
#include "core/vk_common.h"

// LumenFilterPass —— L.4 spatial cross-bilateral SH9 filter.
//
// 一个 compute dispatch：3×3 邻域 bilateral → filteredAtlas。
// 消除 screen probe 间 discontinuity。

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
    float sigmaDepth  = 0.15f;
    float normalPower = 32.0f;
    float sigmaDist   = 100.0f;

private:
    Device* m_device = nullptr;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
};

}
