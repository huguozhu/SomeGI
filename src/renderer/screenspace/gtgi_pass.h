#pragma once
#include "renderer/core/render_targets.h"

// GtgiPass —— C.1：Ground-Truth GI（Sucker Punch 2024）。GTAO 的 GI 升级。
// 算法见 shaders/gi/gtgi/gtgi.slang。
//
// 与 SsgiPass 共享 rt.ssgi 输出 + ssgiPrev 时序 history，per-frame 二
// 选一 dispatch；lighting.slang 读 gSsgi 不区分来源。

namespace somegi {
class Device;

class GtgiPass {
public:
    void init(Device& d);
    void destroy();
    void bindFrame(Device& d, const RenderTargets& rt, VkBuffer frameUbo);
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

    bool  enabled = false;
    int   sliceCount = 4;
    int   samplesPerSlice = 6;
    float radiusPixels = 32.0f;
    float falloff = 5.0f;

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkSampler m_linearClamp = VK_NULL_HANDLE;
};

}
