#pragma once
#include "renderer/core/render_targets.h"
#include <glm/glm.hpp>

// GtaoPass —— B.1：GTAO（Ground-Truth AO，Activision 2016）。
// 与 SsaoPass 共享输出 rt.ssao（R8），调用方 per-frame 二选一 dispatch
// 哪个 pass。算法见 shaders/ssao/gtao.slang。

namespace somegi {
class Device;

class GtaoPass {
public:
    void init(Device& d);
    void destroy();
    void bindFrame(Device& d, const RenderTargets& rt, VkBuffer frameUbo);
    void record(VkCommandBuffer cmd, const RenderTargets& rt,
                const glm::mat4& proj, const glm::mat4& view);

    bool  enabled = false;        // 由 App 的 AO method 选择驱动
    int   sliceCount = 4;
    int   samplesPerSlice = 4;
    float radiusPixels = 32.0f;   // 屏幕空间 march 半径（px）
    float falloff = 5.0f;         // 远 occluder 衰减距离（world unit）

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
};

}
