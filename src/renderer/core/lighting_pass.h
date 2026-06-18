// LightingPass — M4 deferred 核心 compute，已深度迁移到 RHI。
#pragma once
#include "rhi/base/sampler.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/device.h"
#include "core/buffer.h"
#include "renderer/core/render_targets.h"
#include "renderer/gi/lpv/lpv_grid.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include "renderer/gi/prt/prt_resources.h"
#include "renderer/gi/ddgi/ddgi_resources.h"
#include "gi/ibl_baker.h"

namespace somegi {
class Device;

class LightingPass {
public:
    ~LightingPass();
    void init(Device& d, rhi::RHIDevice& rhiDevice);
    void destroy();

    void bindIblResources(Device& d, const IblResources& ibl);
    void bindFrame(Device& d, const RenderTargets& rt, VkBuffer frameUbo,
                   const LpvGrid& lpvGrid0, const VxgiResources& vxgi,
                   const PrtResources& prt, const DdgiResources& ddgi,
                   VkBuffer ddgiProbeStatesBuf);
    void bindShadowMask(Device& d, VkImageView shadowMaskView);
    void setNdgiWeights(Device& d,
        VkBuffer w1, VkBuffer b1, VkBuffer w2, VkBuffer b2,
        VkBuffer w3, VkBuffer b3);
    void record(VkCommandBuffer cmd, const RenderTargets& rt);
    float iblIntensity() const { return m_iblIntensity; }
    void setIblIntensity(float v);

private:
    Device* m_device = nullptr;
    rhi::RHIDevice* m_rhiDevice = nullptr;

    // set=0 (RHI, 34 bindings, UPDATE_AFTER_BIND on 33)
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
    std::unique_ptr<rhi::RHISampler> m_lpvSampler;
    VkImageView m_shadowMaskView = VK_NULL_HANDLE;
    Buffer m_dummyBuf;

    // Pipeline (RHI)
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;

    // IBL set=1 (RHI, 5 bindings)
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_iblDsl;
    std::unique_ptr<rhi::RHIDescriptorSet> m_iblSet;
    Buffer m_iblParamsUbo;
    float m_iblIntensity = 1.0f;
};

} // namespace somegi
