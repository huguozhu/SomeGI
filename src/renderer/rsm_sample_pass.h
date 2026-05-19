#pragma once
#include "render_targets.h"

// RsmSamplePass —— M5.1：把 RSM（4 张 sun-view 图）当 VPL 数组，per
// 屏幕像素抽 disk 样本累加一次反弹漫反射间接光，写入 rt.rsmGI（RGBA16F）。
//
// 算法见 shaders/gi/rsm/rsm_sample.slang：
//  - GBuffer 重建 worldPos / worldNormal
//  - 投到 sun light space → centerUv
//  - centerUv 周围 disk 上 N 个 Hammersley + 哈希抖动样本
//  - 标准 RSM 论文权重累加
//  - 输出 rgb = sumE / sampleCount · intensity，a = hits / sampleCount
//
// 描述符（set=0）：
//   0  ConstantBuffer<FrameUniforms>     gFrame    (主 camera UBO)
//   1  Texture2D                         gNormalRough
//   2  Texture2D                         gDepth
//   3  Texture2D                         gRsmPosition
//   4  Texture2D                         gRsmNormal
//   5  Texture2D                         gRsmFlux
//   6  Sampler                           gLinearClamp
//   7  ConstantBuffer<RsmFrameUniforms>  gRsm
//   8  RWTexture2D                       gOutRsmGI

namespace somegi {
class Device;

class RsmSamplePass {
public:
    void init(Device& d);
    void destroy();

    // 由 App 在初始化 / scene 切换时调用。frameUbo = 主 FrameUniforms，
    // rsmFrameUbo = RsmGeometryPass.frameUboHandle()，3 张 RSM 图直接来
    // 自 RsmGeometryPass 的访问器。
    void bindFrame(Device& d, const RenderTargets& rt,
                   VkBuffer frameUbo, VkBuffer rsmFrameUbo,
                   const Image& rsmPos, const Image& rsmN, const Image& rsmFlux);

    // 录制 dispatch。调用前提：GBuffer / RSM 3 张图已转 SHADER_READ_ONLY；
    // rt.rsmGI 已在 GENERAL（与 SSGI/SSR 同模式，App 端统一管理）。
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

    // ImGui 调参：
    //   sampleCount —— per-pixel 抽样数。32 是噪点 / 性能折中起点。
    //   radius      —— RSM UV 单位的 disk 半径。0.05 ≈ 25 texel @512²。
    //   intensity   —— 整体强度倍率（RSM 公式吐出来值偏小，UI 留个旋钮）。
    bool  enabled     = true;
    int   sampleCount = 32;
    float radius      = 0.05f;
    float intensity   = 1.0f;

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
