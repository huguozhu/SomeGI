#pragma once
#include "core/vk_common.h"
#include "core/image.h"

namespace somegi {
class Device;
struct RenderTargets;

class SmaaPass {
public:
    void init(Device& d, VkExtent2D ext);
    void destroy();
    void bindResources(Device& d, const RenderTargets& rt);
    void bindOutput(Device& d, VkImageView outView);  // rebind blend output
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

private:
    Device* m_device = nullptr;

    // Edge detection
    VkDescriptorSetLayout m_edgeSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_edgePipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_edgePipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_edgePool = VK_NULL_HANDLE;
    VkDescriptorSet m_edgeSet = VK_NULL_HANDLE;

    // Blending
    VkDescriptorSetLayout m_blendSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_blendPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_blendPipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_blendPool = VK_NULL_HANDLE;
    VkDescriptorSet m_blendSet = VK_NULL_HANDLE;

    // Edge texture (R16G16_SFLOAT, screen-sized)
    Image m_edgeTex;
    VkImageLayout m_edgeLayout = VK_IMAGE_LAYOUT_UNDEFINED;
};

}
