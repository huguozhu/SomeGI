// VxgiVoxelizePass — 场景三角形散射到体素 (Compute)，已迁移到 RHI。
#pragma once
#include "rhi/base/texture.h"  // RHITextureView 完整类型
#include "scene/scene.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>
namespace somegi {
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; class RHITextureView; }
class VxgiVoxelizePass {
public:
    ~VxgiVoxelizePass();
    void init(rhi::RHIDevice& d, uint32_t maxTextures);
    void destroy();
    void bindScene(const SceneGpu& gpu, uint32_t textureCount, const VxgiResources& vxgi);
    void record(rhi::RHICommandBuffer& cmd, const SceneCpu& cpu, const SceneGpu& gpu, const glm::vec3& gridMin, float cellSize, uint32_t gridRes);
    void record(VkCommandBuffer cmd, const SceneCpu& cpu, const SceneGpu& gpu, const glm::vec3& gridMin, float cellSize, uint32_t gridRes);
private:
    rhi::RHIDevice* m_rhiDevice = nullptr; uint32_t m_maxTextures = 0;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
    // 纹理数组视图缓存（每 scene 切时重建）
    std::vector<std::unique_ptr<rhi::RHITextureView>> m_texViews;
    std::vector<const rhi::RHITextureView*> m_texViewPtrs;
};
} // namespace somegi
