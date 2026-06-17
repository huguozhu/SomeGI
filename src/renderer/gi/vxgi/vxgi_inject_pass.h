// VxgiInjectPass — RSM→voxel 注入 (Compute), 已迁移到 RHI。
#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>
namespace somegi { class Image; class VxgiResources;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }
class VxgiInjectPass {
public:
    ~VxgiInjectPass();
    void init(rhi::RHIDevice& d, uint32_t rsmSize);
    void destroy();
    void bindResources(const Image& rsmPos, const Image& rsmFlux, const VxgiResources& vxgi);
    void record(rhi::RHICommandBuffer& cmd, uint32_t gridRes, const glm::vec3& gridMin, float cellSize);
    void record(VkCommandBuffer cmd, uint32_t gridRes, const glm::vec3& gridMin, float cellSize);
private:
    rhi::RHIDevice* m_rhiDevice = nullptr; uint32_t m_rsmSize=0;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
}; } // namespace somegi
