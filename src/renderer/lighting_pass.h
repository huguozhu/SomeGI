#pragma once
#include "core/buffer.h"
#include "render_targets.h"
#include "lpv_grid.h"
#include "vxgi_resources.h"
#include "prt_resources.h"
#include "ddgi_resources.h"

// LightingPass —— M4 deferred 路径的核心 compute 阶段。
// 每帧 GBufferPass 写完几何后由它消费：从 GBuffer 重建 worldPos / N、
// 计算 direct sun + IBL/SSR/SSGI 间接光，写到 hdrColor。
//
// 描述符组织：
//   set=0（本类自己 own）：frame UBO + GBuffer 三张图 + depth + storage
//                          hdrColor + ssao + ssr + ssgi 共 9 个 binding。
//                          per-swapchain（resize 时重写 descriptor，但
//                          layout / pipeline 不变）。
//   set=1（来自 IGITechnique）：IBL 资源（diffuse/specular cubes + brdfLut
//                              + sampler + intensity UBO）。切换 GI 技术
//                              时通过 setTechnique() 用新 DSL 重建 pipeline。

namespace somegi {
class Device;
class IGITechnique;

class LightingPass {
public:
    void init(Device& d);
    void destroy();

    // 用 tech->descriptorSetLayout() 作为 set=1 重建 compute pipeline。
    // M4.0 起要求非空：lighting.slang 的 evalIBLDiffuse/Specular 直接
    // 引用 set=1 binding，没法运行时跳过。当用户选 None 时，App 仍然
    // 保留一个 IBLTechnique 让 lighting 有合法的 set=1，shader 自己用
    // FrameUBO.counts.z 切到 hemispheric ambient 分支。
    void setTechnique(IGITechnique* tech);

    // 写 set=0 描述符。调用时机：init 之后一次；onSwapchainResized 后
    // （rt 里所有 image 都换了新 view）；scene 切换不触发，因为 GBuffer
    // 是 per-swapchain 不是 per-scene。
    // M6 LPV / M7 VXGI：除了 GBuffer / SSAO / SSR / SSGI / RSM，还要绑
    // LPV grid[0] 三张 SH image + VXGI voxel mipchain。资源稳定，一次性
    // bind 不需 per-frame 更新。
    void bindFrame(Device& d, const RenderTargets& rt, VkBuffer frameUbo,
                   const LpvGrid& lpvGrid0, const VxgiResources& vxgi,
                   const PrtResources& prt, const DdgiResources& ddgi,
                   VkBuffer ddgiProbeStatesBuf);

    // 更新 set=0 中的 NDGI MLP 权重 binding (27-32)。
    // 权重 buffer 生命周期由 NdgiResources 管理。
    void setNdgiWeights(Device& d,
        VkBuffer w1, VkBuffer b1, VkBuffer w2, VkBuffer b2,
        VkBuffer w3, VkBuffer b3);

    // 录制 dispatch：bindPipeline → bind 两个 set → push constant
    // (outSize/invOutSize) → dispatch (W+7)/8 × (H+7)/8。
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

private:
    void buildPipeline(VkDescriptorSetLayout giDsl);
    void destroyPipeline();

    Device* m_device = nullptr;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkSampler m_lpvSampler = VK_NULL_HANDLE;   // 给 LPV trilinear 用
    Buffer m_dummyBuf;                          // NDGI weights fallback (zero buffer)

    IGITechnique* m_tech = nullptr;
};

}
