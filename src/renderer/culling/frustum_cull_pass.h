// FrustumCullPass —— 视锥剔除 + Hi-Z 遮挡剔除（单 Compute Pass），已迁移到 RHI。
#pragma once
#include "core/buffer.h"
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {

class Device;
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
    void init(Device& d, rhi::RHIDevice& rhiDevice, uint32_t maxDraws);
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

    // 兼容 VkCommandBuffer（迁移期间）
    void record(VkCommandBuffer cmd, VkBuffer drawBuf, uint32_t drawCount,
                VkBuffer indirectOut, VkBuffer countOut,
                const glm::mat4& vp, VkExtent2D screenSize, uint32_t flightIdx);

    void record(VkCommandBuffer cmd, VkBuffer drawBuf, uint32_t drawCount,
                VkBuffer indirectOut, VkBuffer countOut,
                const glm::mat4& vp, VkExtent2D screenSize, uint32_t flightIdx,
                VkImageView hizMip1, VkImageView hizMip2,
                VkImageView hizMip3, VkImageView hizMip4);

private:
    Device* m_device = nullptr;
    rhi::RHIDevice* m_rhiDevice = nullptr;

    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_dsl;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_sets[2];  // double-buffered
    Buffer m_ubo;  // CullUbo (仍使用 core::Buffer，GPU 可见)
    uint32_t m_maxDraws = 0;
};

void extractFrustumPlanes(const glm::mat4& vp, glm::vec4 f[6]);

} // namespace somegi
