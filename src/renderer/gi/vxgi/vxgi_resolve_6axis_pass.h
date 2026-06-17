// VxgiResolve6AxisPass — 6-轴卷积 (Compute, sampler), 已迁移到 RHI。
#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>
namespace somegi { class VxgiResources;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }
class VxgiResolve6AxisPass {
public:
    ~VxgiResolve6AxisPass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindResources(const VxgiResources& vxgi);
    void record(rhi::RHICommandBuffer& cmd, uint32_t gridRes, uint32_t mipLevels, float cellSize, const glm::vec3& gridMin, float strength);
    void record(VkCommandBuffer cmd, uint32_t gridRes, uint32_t mipLevels, float cellSize, const glm::vec3& gridMin, float strength);
private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
    VkSampler m_linearClamp = VK_NULL_HANDLE;
}; } // namespace somegi
