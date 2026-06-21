// GBufferPass — MRT prepass (albedo+metallic, normal+roughness, emissive+AO + depth).
// VS 路径已深度迁移到 RHI，Mesh Shader 路径保留 VK。
#pragma once
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/buffer.h"
#include "rhi/base/device.h"
#include "core/buffer.h"
#include "scene/scene.h"
#include "renderer/core/render_targets.h"
#include "renderer/core/frame_ubo.h"
#include "scene/draw_list.h"

namespace somegi {

class Device;
namespace rhi { class RHICommandBuffer; class RHIBuffer; }

class GBufferPass {
public:
    void init(Device& d, rhi::RHIDevice& rhiDevice,
              VkFormat rt0Fmt, VkFormat rt1Fmt, VkFormat rt2Fmt,
              VkFormat depthFmt, uint32_t maxTextures,
              VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT);
    void destroy();

    void bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount);
    void bindDrawData(Device& d, VkBuffer drawDataBuf);
    void updateFrame(const FrameUBO& ubo);
    void buildMeshGroups(const std::vector<DrawEntry>& entries);
    uint32_t meshGroupCount() const { return m_meshGroupCount; }

    void updateCullUbo(const glm::mat4& viewProj, const glm::vec4 frustum[6],
                       uint32_t drawCount, uint32_t hizMaxMip,
                       uint32_t screenW, uint32_t screenH);

    VkBuffer frameUboHandle() const { return m_frameUbo.handle(); }

    // RHI 路径
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt,
                const rhi::RHIBuffer& indirectBuf, uint32_t drawCount, const SceneGpu& gpu);


    void setMsaaSamples(VkSampleCountFlagBits samples);

    void setMeshShaderEnabled(bool v);
    bool meshShaderEnabled() const { return m_useMeshShader; }
    void bindHiZViews(VkImageView mip1, VkImageView mip2, VkImageView mip3, VkImageView mip4);

private:
    void buildPipeline();       // RHI VS 路径
    void buildMeshPipeline();   // VK Mesh Shader 路径（保留）
    void destroyPipeline();

    Device* m_device = nullptr;
    rhi::RHIDevice* m_rhiDevice = nullptr;
    VkFormat m_rt0Fmt = VK_FORMAT_UNDEFINED;
    VkFormat m_rt1Fmt = VK_FORMAT_UNDEFINED;
    VkFormat m_rt2Fmt = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFmt = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits m_msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    // ── VS 路径（RHI 管理）──
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;

    // ── Mesh Shader 路径（VK，保留）──
    bool m_useMeshShader = false;
    VkPipelineLayout m_meshPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_meshPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_meshSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_meshPool = VK_NULL_HANDLE;
    VkDescriptorSet m_meshSet = VK_NULL_HANDLE;
    Buffer m_cullUbo;
    Buffer m_meshGroupBuf;
    uint32_t m_meshGroupCount = 0;

    Buffer m_frameUbo;
    std::unique_ptr<rhi::RHIBuffer> m_rhiFrameUbo;  // 非拥有型 RHI 包装，init() 中创建
    uint32_t m_maxTextures = 0;
};

} // namespace somegi
