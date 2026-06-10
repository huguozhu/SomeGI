#pragma once
#include "core/buffer.h"
#include "core/shader.h"
#include "scene/scene.h"
#include "renderer/core/render_targets.h"
#include "renderer/core/frame_ubo.h"   // for FrameUBO type
#include "scene/draw_list.h"          // for DrawEntry

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
    void bindDrawData(Device& d, VkBuffer drawDataBuf);
    void updateFrame(const FrameUBO& ubo);
    // 更新 Task Shader 的 CullUbo（Mesh Shader 路径每帧调用）
    // 构建 mesh workgroup 映射表（每 draw 可能拆成多个 group）
    void buildMeshGroups(const std::vector<DrawEntry>& entries);
    uint32_t meshGroupCount() const { return m_meshGroupCount; }

    void updateCullUbo(const glm::mat4& viewProj, const glm::vec4 frustum[6],
                       uint32_t drawCount, uint32_t hizMaxMip,
                       uint32_t screenW, uint32_t screenH);

    // Expose the frame UBO buffer so LightingPass can bind the same memory.
    VkBuffer frameUboHandle() const { return m_frameUbo.handle(); }

    void record(VkCommandBuffer cmd, const RenderTargets& rt,
                VkBuffer indirectBuf, uint32_t drawCount, const SceneGpu& gpu);

    void setMsaaSamples(VkSampleCountFlagBits samples);

    // Mesh Shader 双模式
    void setMeshShaderEnabled(bool v);
    bool meshShaderEnabled() const { return m_useMeshShader; }
    // 绑定 Hi-Z mip views 到 mesh descriptor set（每帧调用，在 Task Shader 前）
    void bindHiZViews(VkImageView mip1, VkImageView mip2, VkImageView mip3, VkImageView mip4);

private:
    void buildPipeline();
    void buildMeshPipeline();    // 构建 mesh pipeline + set=0 layout
    void destroyPipeline();

    Device* m_device = nullptr;
    VkFormat m_rt0Fmt = VK_FORMAT_UNDEFINED;
    VkFormat m_rt1Fmt = VK_FORMAT_UNDEFINED;
    VkFormat m_rt2Fmt = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFmt = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits m_msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    // VS 路径
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    // Mesh Shader 路径
    bool m_useMeshShader = false;
    VkPipelineLayout m_meshPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_meshPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_meshSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_meshPool = VK_NULL_HANDLE;
    VkDescriptorSet m_meshSet = VK_NULL_HANDLE;
    Buffer m_cullUbo;                  // Task Shader CullUniforms UBO
    Buffer m_meshGroupBuf;             // MeshGroup 映射 SSBO
    uint32_t m_meshGroupCount = 0;

    Buffer m_frameUbo;
    uint32_t m_maxTextures = 0;
};

}
