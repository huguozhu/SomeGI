// NdgiPass — NDGI MLP (Compute+RT)，已迁移到 RHI。record 保留 VkCompat。
#pragma once
#include "core/buffer.h"
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>
namespace somegi { class NdgiResources; class SceneRtAS; struct SceneGpu; struct RenderTargets;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }
class NdgiPass {
public:
    ~NdgiPass();
    void init(rhi::RHIDevice& d, bool rtSupported);
    void destroy();
    void bindResources(const NdgiResources& res, SceneRtAS& rtAS, const SceneGpu& scene, const RenderTargets& rt, VkBuffer frameUbo);
    void writeInitDescriptors(const NdgiResources& res);
    void initWeights(rhi::RHICommandBuffer& cmd);
    void initWeights(VkCommandBuffer cmd);
    void record(rhi::RHICommandBuffer& cmd, NdgiResources& res, uint32_t frameIndex, glm::vec3 ddgiOrigin, glm::vec3 ddgiSpacing);
    void record(VkCommandBuffer cmd, NdgiResources& res, uint32_t frameIndex, glm::vec3 ddgiOrigin, glm::vec3 ddgiSpacing);
    void recordTraining(rhi::RHICommandBuffer& cmd, NdgiResources& res, uint32_t frameIndex);
    void recordTraining(VkCommandBuffer cmd, NdgiResources& res, uint32_t frameIndex);
private:
    rhi::RHIDevice* m_rhiDevice = nullptr; bool m_rtSupported = false;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_traceDsl, m_initDsl;
    std::unique_ptr<rhi::RHIPipelineState> m_tracePipeline, m_initPipeline, m_trainPipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_traceSet, m_initSet;
}; } // namespace somegi
