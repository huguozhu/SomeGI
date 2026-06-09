#pragma once
#include "core/vk_common.h"
#include "core/buffer.h"
#include <glm/glm.hpp>

namespace somegi {
class Device;
struct RenderTargets;

class FrustumCullPass {
public:
    void init(Device& d, uint32_t maxDraws);
    void destroy();

    // Without Hi-Z (Phase 2)
    void record(VkCommandBuffer cmd, VkBuffer drawBuf, uint32_t drawCount,
                VkBuffer indirectOut, VkBuffer countOut,
                const glm::mat4& vp, VkExtent2D screenSize, uint32_t flightIdx);

    // With Hi-Z (Phase 3): takes Hi-Z mip image views for occlusion test
    void record(VkCommandBuffer cmd, VkBuffer drawBuf, uint32_t drawCount,
                VkBuffer indirectOut, VkBuffer countOut,
                const glm::mat4& vp, VkExtent2D screenSize, uint32_t flightIdx,
                VkImageView hizMip1, VkImageView hizMip2,
                VkImageView hizMip3, VkImageView hizMip4);

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_dsl = VK_NULL_HANDLE;
    VkPipelineLayout m_pl = VK_NULL_HANDLE;
    VkPipeline m_pipe = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_sets[kFramesInFlight]{};
    Buffer m_ubo;
    uint32_t m_maxDraws = 0;
};

void extractFrustumPlanes(const glm::mat4& vp, glm::vec4 f[6]);
}
