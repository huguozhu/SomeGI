#pragma once
#include "renderer/core/render_targets.h"

// SsgiPass —— 屏幕空间一次反弹漫反射（M4.3）。
//
// 算法：每像素 cosine-weighted 半球抽 N 条 ray（Hammersley 序列 +
// per-pixel 哈希抖动让邻居走不同方向），沿 ray 在世界空间 march；
// 命中处用线性 sampler 在 hdrPrev 取色累加；输出 rgba16f：
// rgb = hits 的平均 radiance（注意是"原始 incoming"，未上 BRDF），
// a = hits / sampleCount（hit fraction）。
//
// LightingPass 用 ssgi.a 当混合权重，把 evalIBLDiffuse 的输出与
// applyIBLDiffuseBRDF(ssgi.rgb,...) 之间 lerp。命中越多 → 越偏屏幕空间
// 真值；命中少（比如 ray 飞出屏幕） → 回落到 IBL diffuse。
//
// 已知瑕疵：8 sample 噪点明显（lighting.slang 端有 5×5 bilateral 过滤
// 减噪）；第一帧 hdrPrev 全 0 → SSGI 第一帧偏暗，几帧后通过 feedback
// 收敛。无 spatial filter 单独 pass，无时序累积。

namespace somegi {
class Device;

class SsgiPass {
public:
    void init(Device& d);
    void destroy();
    void bindFrame(Device& d, const RenderTargets& rt, VkBuffer frameUbo);
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

    // ImGui 可调；maxDist 由 App 按 length(scene AABB) * 0.25 自适应
    // （diffuse 反弹衰减快，比 SSR 短即可）。sampleCount 主要影响噪点。
    bool  enabled     = true;
    int   sampleCount = 8;     // 每像素半球采样数；增大降噪但线性加成本
    int   maxSteps    = 24;
    float maxDist     = 10.0f;
    float thickness   = 0.05f;

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
