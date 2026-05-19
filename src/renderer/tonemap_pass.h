#pragma once
#include "core/shader.h"
#include "render_targets.h"

namespace somegi {
class Device;

class TonemapPass {
public:
    void init(Device& d, VkSampler linearSampler);
    void destroy();

    void bindTargets(Device& d, const RenderTargets& rt);
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkSampler m_sampler = VK_NULL_HANDLE;
};

}
