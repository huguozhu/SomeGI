#pragma once
#include "core/vk_common.h"
#include "core/buffer.h"
#include "core/shader.h"
#include <glm/glm.hpp>

namespace somegi {

class Device;
class NdgiResources;
struct SceneGpu;
class SceneRtAS;
struct RenderTargets;
class VxgiResources;

// NDGI Pass: manages the compute pipelines for probe tracing and weight init
class NdgiPass {
public:
    void init(Device& d, bool rtSupported);
    void destroy();

    // Bind resources that live across frames
    void bindResources(Device& d, NdgiResources& res, SceneRtAS& rtAS,
                       const SceneGpu& scene, const RenderTargets& rt,
                       VkBuffer frameUbo);

    // Write weight buffer descriptors to init/training set (call before initWeights or training)
    void writeInitDescriptors(Device& d, NdgiResources& res);
    // One-shot: initialize MLP weights with Xavier
    void initWeights(VkCommandBuffer cmd);

    // Per-frame: trace probe rays to collect training samples
    void record(VkCommandBuffer cmd, NdgiResources& res, uint32_t frameIndex,
                glm::vec3 ddgiOrigin, glm::vec3 ddgiSpacing);
    // Per-frame: train MLP with collected samples (call after probe trace)
    void recordTraining(VkCommandBuffer cmd, NdgiResources& res, uint32_t frameIndex);

private:
    Device* m_device = nullptr;
    bool m_rtSupported = false;

    // Probe trace pipeline
    VkDescriptorSetLayout m_traceDsl = VK_NULL_HANDLE;
    VkPipelineLayout m_tracePipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_tracePipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_tracePool = VK_NULL_HANDLE;
    VkDescriptorSet m_traceSet = VK_NULL_HANDLE;

    // Weight init pipeline
    VkDescriptorSetLayout m_initDsl = VK_NULL_HANDLE;
    VkPipelineLayout m_initPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_initPipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_initPool = VK_NULL_HANDLE;
    VkDescriptorSet m_initSet = VK_NULL_HANDLE;

    // Training pipeline (reuses init descriptor layout + set)
    VkPipelineLayout m_trainPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_trainPipeline = VK_NULL_HANDLE;
};

} // namespace somegi
