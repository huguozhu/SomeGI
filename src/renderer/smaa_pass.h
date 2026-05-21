#pragma once
#include "core/vk_common.h"

namespace somegi {
class Device;
struct RenderTargets;

class SmaaPass {
public:
    void init(Device& d);
    void destroy();
    void bindResources(Device& d, const RenderTargets& rt);
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
};

}
