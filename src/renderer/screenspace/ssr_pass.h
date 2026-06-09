#pragma once
#include "renderer/core/render_targets.h"

// SsrPass —— 屏幕空间反射（M4.2）。
//
// 算法：每像素从 GBuffer 取 N + 重建 worldPos，算反射方向 R = reflect(-V, N)；
// 沿 R 在世界空间线性 march，每步把当前点投影到屏幕看 sceneDepth，
// 当 ray 越过 surface 且超出量在 thickness 内 → 命中。命中处用线性
// sampler 在 hdrPrev（上一帧的 hdr）取颜色，出 rgba16f：rgb=反射颜色，
// a=置信度（屏幕边缘 + 粗糙度 fade）。
//
// 关键权衡：用 hdrPrev 而不是当前帧的 hdrColor，避免 lighting → ssr →
// lighting 的同帧依赖循环。代价是反射有 1 帧延迟（相机静止时无视觉
// 影响；快速移动时有轻微 ghosting）。
//
// 阶段位置：GBufferPass 之后、LightingPass 之前。LightingPass 用 ssr.a
// 在 IBL specular 与 ssr.rgb 之间 lerp。
//
// 已知瑕疵（plan §13）：linear march + 固定 thickness 在远距离反射
// 出现阶梯感、薄物体（栏杆、网）会被漏过，第一帧 hdrPrev 全 0 反射
// 偏暗。Hi-Z + 二分细化 + 时序累积留给 M7。

namespace somegi {
class Device;

class SsrPass {
public:
    void init(Device& d);
    void destroy();

    // resize 后重写 set=0：gNormalRough/depth/hdrPrev/ssr 所有 image view
    // 都换了。FrameUBO 由 GBufferPass 共享，传 buffer handle 进来即可。
    void bindFrame(Device& d, const RenderTargets& rt, VkBuffer frameUbo);

    void record(VkCommandBuffer cmd, const RenderTargets& rt);

    // ImGui 可调；maxDist 由 App 按 length(scene AABB) * 0.5 自适应。
    bool  enabled         = true;
    int   maxSteps        = 32;
    float maxDist         = 50.0f;
    float thickness       = 0.05f;   // NDC z 命中容差；非常依赖场景尺度
    // 粗糙度 > 此阈值的像素直接跳过 SSR（输出 a=0，lighting 端 fallback
    // 到 IBL specular）。0.4 默认偏严会让 Sponza 大理石（~0.5）拿不到
    // SSR；0.7 是放宽后的经验值，shader 的 roughness fade 让过渡平滑。
    float roughThreshold  = 0.7f;

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
