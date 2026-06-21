// LpvPropagatePass — 6-邻居 SH 传播 (Compute)，已迁移到 RHI。
#pragma once
#include "renderer/gi/lpv/lpv_grid.h"
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {
class Device;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }
class Image;

class LpvPropagatePass {
public:
    ~LpvPropagatePass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindResources(const LpvGrid& g0, const LpvGrid& g1, const Image& gv);
    void record(rhi::RHICommandBuffer& cmd, int srcIdx, uint32_t gridRes,
                float occAmp, float gvOccStr);

    int iterations = 8; float occlusionAmplifier = 1.0f; float gvOcclusionStrength = 1.0f;

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_sets[2];
};

} // namespace somegi
