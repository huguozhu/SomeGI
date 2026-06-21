// FrustumCullPass —— 视锥剔除 + Hi-Z 遮挡剔除（单 Compute Pass），已迁移到纯 RHI。
#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {

struct RenderTargets;

namespace rhi {
class RHIDevice;
class RHIDescriptorSetLayout;
class RHIPipelineState;
class RHIDescriptorSet;
class RHIBuffer;
class RHICommandBuffer;
}

class FrustumCullPass {
public:
    ~FrustumCullPass();
    void init(rhi::RHIDevice& d, uint32_t maxDraws);
    void destroy();

    // RHI 路径（无 Hi-Z）
    void record(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& drawBuf, uint32_t drawCount,
                const rhi::RHIBuffer& indirectOut, const rhi::RHIBuffer& countOut,
                const glm::mat4& vp, VkExtent2D screenSize, uint32_t flightIdx);

    // RHI 路径（含 Hi-Z）
    void record(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& drawBuf, uint32_t drawCount,
                const rhi::RHIBuffer& indirectOut, const rhi::RHIBuffer& countOut,
                const glm::mat4& vp, VkExtent2D screenSize, uint32_t flightIdx,
                VkImageView hizMip1, VkImageView hizMip2,
                VkImageView hizMip3, VkImageView hizMip4);

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;

    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_dsl;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_sets[2];  // double-buffered
    std::unique_ptr<rhi::RHIBuffer> m_ubo;  // CullUbo (纯 RHI，HostVisible)
    uint32_t m_maxDraws = 0;
};

void extractFrustumPlanes(const glm::mat4& vp, glm::vec4 f[6]);

} // namespace somegi
