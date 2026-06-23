// ForwardPass — 前向渲染 (Graphics+IBL)，已深度迁移到 RHI。
#pragma once
#include "rhi/base/sampler.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/device.h"
#include "core/buffer.h"
#include "scene/scene.h"
#include "renderer/core/render_targets.h"
#include "renderer/core/frame_ubo.h"
#include "gi/ibl_baker.h"
#include "scene/draw_list.h"
#include <glm/glm.hpp>

namespace somegi {
class Device;
namespace rhi { class RHICommandBuffer; class RHIBuffer; }

class ForwardPass {
public:
    ~ForwardPass();
    void init(Device& d, rhi::RHIDevice& rhiDevice, VkFormat colorFmt, VkFormat depthFmt, uint32_t maxTextures);
    void destroy();
    void bindIblResources(Device& d, const IblResources& ibl);
    void bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount);
    void bindDrawData(Device& d, VkBuffer drawDataBuf);
    void updateFrame(const FrameUBO& ubo);
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt,
                const rhi::RHIBuffer& indirectBuf, uint32_t drawCount, const SceneGpu& gpu);

    void setNdgiWeights(Device& d,
        VkBuffer w1, VkBuffer b1, VkBuffer w2, VkBuffer b2,
        VkBuffer w3, VkBuffer b3);
    void setMeshShaderEnabled(bool v) { m_useMeshShader = v; }
    bool meshShaderEnabled() const { return m_useMeshShader; }
    void bindHiZViews(VkImageView mip1, VkImageView mip2, VkImageView mip3, VkImageView mip4);
    void buildMeshGroups(const std::vector<DrawEntry>& entries);
    uint32_t meshGroupCount() const { return m_meshGroupCount; }
    void updateCullUbo(const glm::mat4& viewProj, const glm::vec4 frustum[6],
                       uint32_t drawCount, uint32_t hizMaxMip, uint32_t screenW, uint32_t screenH);

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    VkFormat m_colorFmt = VK_FORMAT_UNDEFINED, m_depthFmt = VK_FORMAT_UNDEFINED;

    // set=0 (RHI, 11 bindings + texture array + PARTIALLY_BOUND)
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;

    // IBL set=1 (RHI, 5 bindings)
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_iblDsl;
    std::unique_ptr<rhi::RHIDescriptorSet> m_iblSet;

    // Mesh Shader (已迁移到 RHI)
    bool m_useMeshShader = false;
    std::unique_ptr<rhi::RHIPipelineState> m_meshPipelineRhi;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_meshSetLayoutRhi;
    std::unique_ptr<rhi::RHIDescriptorSet> m_meshSetRhi;
    Buffer m_cullUbo, m_meshGroupBuf;
    uint32_t m_meshGroupCount = 0;

    Buffer m_frameUbo;
    Buffer m_iblParamsUbo;  // IBL intensity UBO (set=1 binding 4)
    Buffer m_dummySBuf;     // 占位 STORAGE buffer（NDGI weights 初始值）
    std::unique_ptr<rhi::RHIBuffer> m_rhiFrameUbo;       // 非拥有型 RHI 包装
    std::unique_ptr<rhi::RHIBuffer> m_rhiIblParamsUbo;   // 非拥有型 RHI 包装
    std::unique_ptr<rhi::RHIBuffer> m_rhiDummySBuf;      // 非拥有型 RHI 包装
    uint32_t m_maxTextures = 0;
    std::vector<std::unique_ptr<rhi::RHITextureView>> m_texViews;
    std::vector<const rhi::RHITextureView*> m_texViewPtrs;
};

} // namespace somegi
