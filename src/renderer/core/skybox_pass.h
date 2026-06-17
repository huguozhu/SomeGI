// SkyboxPass —— Graphics PSO 天空盒，已迁移到 RHI（首个 Graphics PSO 验证）。
#pragma once
#include "core/buffer.h"
#include "renderer/core/render_targets.h"
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {

class Device;

namespace rhi {
class RHIDevice;
class RHIDescriptorSetLayout;
class RHIPipelineState;
class RHIDescriptorSet;
class RHICommandBuffer;
}

class SkyboxPass {
public:
    ~SkyboxPass();
    void init(Device& d, rhi::RHIDevice& rhiDevice, VkFormat colorFmt, VkFormat depthFmt);
    void destroy();

    void bindEnv(VkImageView envCubeView, VkSampler linearSampler);
    void updateFrame(const glm::mat4& invViewProj, const glm::vec3& cameraPos);

    // RHI 路径
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt);
    // 兼容 VkCommandBuffer
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

private:
    Device* m_device = nullptr;
    rhi::RHIDevice* m_rhiDevice = nullptr;

    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_set;

    Buffer m_ubo;
    VkSampler m_sampler = VK_NULL_HANDLE;  // 外部传入，RHI 不管理生命周期
    VkFormat m_colorFmt = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFmt = VK_FORMAT_UNDEFINED;
};

} // namespace somegi
