// PrtBakePass — PRT 可见性烘焙 (Compute+sampler)，已迁移到 RHI。
#pragma once
#include "rhi/base/sampler.h"
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>
namespace somegi { class VxgiResources; class PrtResources;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }
class PrtBakePass {
public:
    ~PrtBakePass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindResources(const VxgiResources& vxgi, const PrtResources& prt);
    void record(rhi::RHICommandBuffer& cmd, const glm::vec3& prtGridMin, float prtCellSize, uint32_t prtRes, const glm::vec3& vxgiGridMin, float vxgiCellSize, uint32_t vxgiRes, uint32_t numSamples=64);
    void record(VkCommandBuffer cmd, const glm::vec3& pgm, float pcs, uint32_t pr, const glm::vec3& vgm, float vcs, uint32_t vr, uint32_t ns=64);
private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
    std::unique_ptr<rhi::RHISampler> m_linearClamp;
}; } // namespace somegi
