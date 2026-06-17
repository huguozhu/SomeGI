// RsmGeometryPass — Sun 视角 MRT 渲染 (Graphics PSO)，已迁移到 RHI。
#pragma once
#include "rhi/base/texture.h"
#include "core/buffer.h"
#include "core/image.h"
#include "scene/scene.h"
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {
class Device;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }

class RsmGeometryPass {
public:
    void init(Device& d, rhi::RHIDevice& rhiDevice, uint32_t maxTextures);
    void destroy();

    void bindScene(const SceneGpu& gpu, uint32_t textureCount);
    void bindDrawData(VkBuffer drawDataBuf);
    void updateLight(const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                     const glm::vec3& sunDir, const glm::vec3& sunColor, float sunIntensity);

    void record(rhi::RHICommandBuffer& cmd, VkBuffer indirectBuf, uint32_t drawCount, const SceneGpu& gpu);
    void record(VkCommandBuffer cmd, VkBuffer indirectBuf, uint32_t drawCount, const SceneGpu& gpu);

    const Image& position() const { return m_position; }
    const Image& normal()   const { return m_normal; }
    const Image& flux()     const { return m_flux; }
    const Image& depth()    const { return m_depth; }
    VkBuffer frameUboHandle() const { return m_rsmFrameUbo.handle(); }
    static constexpr uint32_t kRsmSize = 512;

private:
    Device* m_device = nullptr;
    rhi::RHIDevice* m_rhiDevice = nullptr;

    Image m_position, m_normal, m_flux, m_depth;
    Buffer m_rsmFrameUbo;

    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;

    uint32_t m_maxTextures = 0;
    std::vector<std::unique_ptr<rhi::RHITextureView>> m_texViews;
    std::vector<const rhi::RHITextureView*> m_texViewPtrs;
};

} // namespace somegi
