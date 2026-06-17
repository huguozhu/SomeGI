// TaaPass —— Temporal Anti-Aliasing (Compute)，已迁移到 RHI。
#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {
struct RenderTargets;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }

class TaaPass {
public:
    ~TaaPass();
    void init(rhi::RHIDevice& d);
    void destroy();
    void bindResources(const RenderTargets& rt, uint32_t frameIdx);
    void bindOutput(VkImageView outView, uint32_t frameIdx);
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt,
                const glm::vec2& jitter, const glm::vec2& prevJitter,
                const glm::mat4& invViewProj, const glm::mat4& prevViewProj,
                uint32_t frameIdx, float blendAlpha = 0.9f);
    void record(VkCommandBuffer cmd, const RenderTargets& rt,
                const glm::vec2& jitter, const glm::vec2& prevJitter,
                const glm::mat4& invViewProj, const glm::mat4& prevViewProj,
                uint32_t frameIdx, float blendAlpha = 0.9f);

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_sets[2];
};

} // namespace somegi
