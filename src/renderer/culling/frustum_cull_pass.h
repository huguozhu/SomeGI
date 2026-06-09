#pragma once
#include "core/vk_common.h"
#include "core/buffer.h"
#include <glm/glm.hpp>
namespace somegi {
class Device;
class FrustumCullPass {
public:
    void init(Device& d, uint32_t maxDraws);
    void destroy();
    void record(VkCommandBuffer cmd, VkBuffer drawBuf, uint32_t drawCount, VkBuffer indirectOut, VkBuffer countOut, const glm::mat4& vp, uint32_t flightIdx);
private:
    Device* m_device=nullptr;
    VkDescriptorSetLayout m_dsl=VK_NULL_HANDLE;
    VkPipelineLayout m_pl=VK_NULL_HANDLE;
    VkPipeline m_pipe=VK_NULL_HANDLE;
    VkDescriptorPool m_pool=VK_NULL_HANDLE;
    VkDescriptorSet m_sets[kFramesInFlight]{};
    Buffer m_ubo;
};
void extractFrustumPlanes(const glm::mat4& vp, glm::vec4 f[6]);
}
