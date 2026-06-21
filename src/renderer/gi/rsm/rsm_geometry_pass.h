// RsmGeometryPass — Sun 视角 MRT 渲染 (Graphics PSO)，已迁移到纯 RHI。
#pragma once
#include "rhi/base/texture.h"
#include "rhi/base/buffer.h"
#include "scene/scene.h"
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; class RHIBuffer; }

class RsmGeometryPass {
public:
    void init(rhi::RHIDevice& d, uint32_t maxTextures);
    void destroy();

    void bindScene(const SceneGpu& gpu, uint32_t textureCount);
    void bindDrawData(VkBuffer drawDataBuf);
    void updateLight(const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                     const glm::vec3& sunDir, const glm::vec3& sunColor, float sunIntensity);

    void record(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& indirectBuf, uint32_t drawCount, const SceneGpu& gpu);

    const rhi::RHITexture& positionTex() const { return *m_positionTex; }
    const rhi::RHITexture& normalTex()   const { return *m_normalTex; }
    const rhi::RHITexture& fluxTex()     const { return *m_fluxTex; }
    const rhi::RHITexture& depthTex()    const { return *m_depthTex; }
    VkBuffer frameUboHandle() const { return (VkBuffer)(uintptr_t)m_rsmFrameUbo->nativeHandle(); }
    static constexpr uint32_t kRsmSize = 512;

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;

    std::unique_ptr<rhi::RHITexture> m_positionTex, m_normalTex, m_fluxTex, m_depthTex;
    std::unique_ptr<rhi::RHITextureView> m_positionView, m_normalView, m_fluxView, m_depthView;
    std::unique_ptr<rhi::RHIBuffer> m_rsmFrameUbo;

    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;

    uint32_t m_maxTextures = 0;
    std::vector<std::unique_ptr<rhi::RHITextureView>> m_texViews;
    std::vector<const rhi::RHITextureView*> m_texViewPtrs;
};

} // namespace somegi
