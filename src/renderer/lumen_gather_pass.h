#pragma once
#include "core/vk_common.h"
#include <glm/glm.hpp>

// LumenGatherPass —— L.5 Final Gather。
//
// 一个 compute dispatch：bilinear 插值屏幕 probe SH9 → irradiance 重建
// → 乘 albedo/π → 写 rt.lumenGI。

namespace somegi {
class Device;
class LumenResources;
struct RenderTargets;

class LumenGatherPass {
public:
    void init(Device& d);
    void destroy();

    // 一次性 bind（per-resize）
    // useFiltered: true = 读 filteredAtlas（L.4 打开时），false = 读 probeAtlas
    void bindResources(Device& d, const LumenResources& res,
                       const RenderTargets& rt, VkBuffer frameUbo,
                       bool useFiltered);

    // 每帧 dispatch
    void record(VkCommandBuffer cmd, const LumenResources& res,
                const RenderTargets& rt, uint32_t debugMode = 0);

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
