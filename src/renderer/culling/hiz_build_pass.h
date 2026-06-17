// HiZBuildPass —— 构建深度层级金字塔，已迁移到 RHI。
#pragma once
#include "core/image.h"
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
class RHICommandBuffer;
}

class HiZBuildPass {
public:
    ~HiZBuildPass();
    void init(Device& d, rhi::RHIDevice& rhiDevice, VkExtent2D screenExtent);
    void destroy();

    // RHI 路径
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt);
    // 兼容 VkCommandBuffer（迁移期间）
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

    VkImageView mip1View() const { return m_mip1.view(); }
    VkImageView mip2View() const { return m_mip2.view(); }
    VkImageView mip3View() const { return m_mip3.view(); }
    VkImageView mip4View() const { return m_mip4.view(); }

    VkExtent2D mipExtent(uint32_t level) const;

private:
    Device* m_device = nullptr;
    rhi::RHIDevice* m_rhiDevice = nullptr;
    VkExtent2D m_extent{};

    Image m_mip1, m_mip2, m_mip3, m_mip4;

    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_dsl;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_set;
};

} // namespace somegi
