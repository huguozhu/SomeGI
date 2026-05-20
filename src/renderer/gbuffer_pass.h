#pragma once
#include "core/buffer.h"
#include "core/shader.h"
#include "scene/scene.h"
#include "render_targets.h"
#include "forward_pass.h"   // for FrameUBO type

namespace somegi {
class Device;

// Geometry-only prepass: writes albedo+metallic, normal+roughness,
// emissive+AO into the GBuffer color attachments + depth. Lighting is
// evaluated in the LightingPass (compute) that consumes this output.
class GBufferPass {
public:
    void init(Device& d,
              VkFormat rt0Fmt, VkFormat rt1Fmt, VkFormat rt2Fmt,
              VkFormat depthFmt, uint32_t maxTextures,
              VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT);
    void destroy();

    void bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount);
    void updateFrame(const FrameUBO& ubo);

    // Expose the frame UBO buffer so LightingPass can bind the same memory.
    VkBuffer frameUboHandle() const { return m_frameUbo.handle(); }

    void record(VkCommandBuffer cmd, const RenderTargets& rt,
                const SceneCpu& cpu, const SceneGpu& gpu);

    void setMsaaSamples(VkSampleCountFlagBits samples);

private:
    void buildPipeline();
    void destroyPipeline();

    Device* m_device = nullptr;
    VkFormat m_rt0Fmt = VK_FORMAT_UNDEFINED;
    VkFormat m_rt1Fmt = VK_FORMAT_UNDEFINED;
    VkFormat m_rt2Fmt = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFmt = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits m_msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    Buffer m_frameUbo;
    uint32_t m_maxTextures = 0;
};

}
