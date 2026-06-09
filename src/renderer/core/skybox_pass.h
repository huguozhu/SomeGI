#pragma once
#include "core/buffer.h"
#include "renderer/core/render_targets.h"
#include <glm/glm.hpp>

namespace somegi {
class Device;

class SkyboxPass {
public:
    void init(Device& d, VkFormat colorFmt, VkFormat depthFmt);
    void destroy();

    // Bind an env cubemap (view must be VK_IMAGE_VIEW_TYPE_CUBE) + linear sampler.
    void bindEnv(Device& d, VkImageView envCubeView, VkSampler linearSampler);

    // Update per-frame constants (invViewProj + camera position).
    void updateFrame(const glm::mat4& invViewProj, const glm::vec3& cameraPos);

    // Render after the forward pass (color/depth still in attachment layout).
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

private:
    Device* m_device = nullptr;
    VkFormat m_colorFmt = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFmt = VK_FORMAT_UNDEFINED;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    Buffer m_ubo;
};

}
