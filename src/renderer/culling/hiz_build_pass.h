#pragma once
#include "core/vk_common.h"
#include "core/image.h"
#include "core/buffer.h"
#include <glm/glm.hpp>

namespace somegi {
class Device;
struct RenderTargets;

class HiZBuildPass {
public:
    void init(Device& d, VkExtent2D screenExtent);
    void destroy();

    // Build Hi-Z pyramid from rt.depth. Call BEFORE culling (uses depth from PREVIOUS frame).
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

    VkImageView mip1View() const { return m_mip1.view(); }
    VkImageView mip2View() const { return m_mip2.view(); }
    VkImageView mip3View() const { return m_mip3.view(); }
    VkImageView mip4View() const { return m_mip4.view(); }

    VkExtent2D mipExtent(uint32_t level) const;

private:
    Device* m_device = nullptr;
    VkExtent2D m_extent{};

    Image m_mip1, m_mip2, m_mip3, m_mip4;

    VkDescriptorSetLayout m_dsl = VK_NULL_HANDLE;
    VkPipelineLayout m_pl = VK_NULL_HANDLE;
    VkPipeline m_pipe = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    Buffer m_pcUbo;  // push constant data (src size)
};

} // namespace somegi
