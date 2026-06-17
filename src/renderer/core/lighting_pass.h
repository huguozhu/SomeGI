#pragma once
#include "rhi/base/device.h"
#include "core/buffer.h"
#include "renderer/core/render_targets.h"
#include "renderer/gi/lpv/lpv_grid.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include "renderer/gi/prt/prt_resources.h"
#include "renderer/gi/ddgi/ddgi_resources.h"
#include "gi/ibl_baker.h"                  // IblResources

// LightingPass —— M4 deferred 路径的核心 compute 阶段。
// 每帧 GBufferPass 写完几何后由它消费：从 GBuffer 重建 worldPos / N、
// 计算 direct sun + IBL/SSR/SSGI 间接光，写到 hdrColor。
//
// 描述符组织：
//   set=0（本类自己 own）：frame UBO + GBuffer 三张图 + depth + storage
//                          hdrColor + ssao + ssr + ssgi + 各路 GI 输出，
//                          共 33 个 binding。
//   set=1（本类自己 own）：IBL 资源（diffuse/specular cubes + brdfLut
//                         + sampler + intensity UBO），启动时一次性创建。
// GI 模式切换通过 FrameUBO 闸门标志在 shader 内分支，不重建 pipeline。

namespace somegi {
class Device;

class LightingPass {
public:
    void init(Device& d, rhi::RHIDevice& rhiDevice);
    void destroy();

    // 绑定 IBL 预烘焙资源到 set=1，创建 pipeline（仅调用一次，init 之后）
    void bindIblResources(Device& d, const IblResources& ibl);

    // 绑定 set=0 描述符。调用时机：init 之后一次；onSwapchainResized 后重新调用。
    void bindFrame(Device& d, const RenderTargets& rt, VkBuffer frameUbo,
                   const LpvGrid& lpvGrid0, const VxgiResources& vxgi,
                   const PrtResources& prt, const DdgiResources& ddgi,
                   VkBuffer ddgiProbeStatesBuf);

    // 绑定 shadowMask（set=0, binding=33） —— R8_UNORM, 1=lit, 0=shadow
    void bindShadowMask(Device& d, VkImageView shadowMaskView);

    // 更新 set=0 中的 NDGI MLP 权重 binding (27-32)
    void setNdgiWeights(Device& d,
        VkBuffer w1, VkBuffer b1, VkBuffer w2, VkBuffer b2,
        VkBuffer w3, VkBuffer b3);

    // 录制 dispatch：bindPipeline → bind 两个 set → push constant → dispatch
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

    // IBL intensity —— 供 App 的 buildUI() 滑条使用
    float iblIntensity() const { return m_iblIntensity; }
    // 写 intensity 到 UBO（host-coherent，slider 改变时调用一次即可）
    void setIblIntensity(float v);

private:
    void createIblDescriptorSetLayout();   // set=1 布局
    void buildPipeline();                  // 用 m_setLayout + m_iblDsl 创建 pipeline
    void destroyPipeline();

    Device* m_device = nullptr;
    rhi::RHIDevice* m_rhiDevice = nullptr;

    // set=0
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkSampler m_lpvSampler = VK_NULL_HANDLE;
    VkImageView m_shadowMaskView = VK_NULL_HANDLE;
    Buffer m_dummyBuf;

    // Pipeline
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    // IBL set=1（init 时创建布局，bindIblResources 时分配 set + 写描述符）
    VkDescriptorSetLayout m_iblDsl = VK_NULL_HANDLE;
    VkDescriptorPool m_iblPool = VK_NULL_HANDLE;
    VkDescriptorSet m_iblSet = VK_NULL_HANDLE;
    Buffer m_iblParamsUbo;              // set=1 binding 4: { intensity, pad×3 }
    float m_iblIntensity = 1.0f;       // ImGui slider 状态
};

}
