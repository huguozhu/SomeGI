#pragma once
#include "renderer/core/render_targets.h"
#include <glm/glm.hpp>

// SsaoPass —— 屏幕空间环境光遮蔽（M4.1）。
//
// 采样模型：从 gDepth 重建当前像素的 view-space 位置 vP，从 gNormalRough
// 取世界法线后 view 转到 view-space 法线 vN；在 vN 周围的 cosine 半球
// 取 N 个样本方向，沿 vN 推 radius 距离得到样本点，投影回屏幕看实际
// 场景在那里的深度，深于样本点 → 视为被遮挡。
//
// 阶段位置：GBufferPass 之后、LightingPass 之前。LightingPass 在
// indirect diffuse 项上乘 ssao 来给"几何凹处"加上额外暗化。
//
// 参数对场景敏感：radius 默认在 App::applySceneSelection 里按场景
// AABB 直径自适应（length(d) * 0.005），让 cube（~2400u）和 Sponza
// （~40u）都能用合适的范围。

namespace somegi {
class Device;

class SsaoPass {
public:
    void init(Device& d);
    void destroy();

    // GBuffer image view 在 swapchain resize 时换新；调用方在 onSwapchainResized
    // 重新绑一次。scene 切换不影响（GBuffer 是 per-swapchain）。
    void bindFrame(Device& d, const RenderTargets& rt);

    // 调用方把自己的 frame UBO 矩阵传进来（避免 SSAO pass 单独再绑一次
    // FrameUBO 描述符）；shader 用 proj/invProj 重建 view-space 位置和
    // 投影回屏幕，view 把 world 法线转到 view-space。
    void record(VkCommandBuffer cmd, const RenderTargets& rt,
                const glm::mat4& proj, const glm::mat4& invProj,
                const glm::mat4& view);

    // ImGui 可调的运行时状态。radius 单位是世界单位；默认会被 App 按
    // 场景 AABB 直径覆盖。bias 用于避免自遮挡（深度比较的小偏移）。
    bool  enabled      = true;
    float radius       = 0.5f;
    float bias         = 0.025f;
    int   sampleCount  = 16;     // 每像素采样数；增加降噪但线性加成本

private:
    Device* m_device = nullptr;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
};

}
