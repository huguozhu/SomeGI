// DdgiPass — DDGI 4-pass pipeline (Compute)，已迁移到 RHI。
#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>
namespace somegi { class DdgiResources; class VxgiResources;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }
class DdgiPass {
public:
    ~DdgiPass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindResources(const DdgiResources& ddgi, const VxgiResources& vxgi);
    void record(rhi::RHICommandBuffer& cmd, const DdgiResources& ddgi, const glm::vec3& ddgiOrigin, const glm::vec3& ddgiSpacing, const glm::vec3& vxgiGridMin, float vxgiCellSize, uint32_t vxgiRes, float randomRotation, uint32_t frameIndex);
    void record(VkCommandBuffer cmd, const DdgiResources& ddgi, const glm::vec3& dO, const glm::vec3& dS, const glm::vec3& vM, float vC, uint32_t vR, float rR, uint32_t fI);
private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout, m_setLayoutBlend, m_setLayoutClassify;
    std::unique_ptr<rhi::RHIPipelineState> m_pipelineUpdate, m_pipelineClassify, m_pipelineBlendIrr, m_pipelineBlendDist;
    std::unique_ptr<rhi::RHIDescriptorSet> m_setUpdate, m_setBlend, m_setClassify;
    VkSampler m_linearClamp = VK_NULL_HANDLE;
}; } // namespace somegi
