// HiZBuildPass —— 构建深度层级金字塔，已迁移到纯 RHI。
#pragma once
#include "rhi/base/texture.h"
#include "core/buffer.h"
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {

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
    void init(rhi::RHIDevice& d, VkExtent2D screenExtent);
    void destroy();

    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt);

    VkImageView mip1View() const { return (VkImageView)(uintptr_t)m_mipView1->nativeHandle(); }
    VkImageView mip2View() const { return (VkImageView)(uintptr_t)m_mipView2->nativeHandle(); }
    VkImageView mip3View() const { return (VkImageView)(uintptr_t)m_mipView3->nativeHandle(); }
    VkImageView mip4View() const { return (VkImageView)(uintptr_t)m_mipView4->nativeHandle(); }

    VkExtent2D mipExtent(uint32_t level) const;

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    VkExtent2D m_extent{};

    std::unique_ptr<rhi::RHITexture> m_mipTex1, m_mipTex2, m_mipTex3, m_mipTex4;
    std::unique_ptr<rhi::RHITextureView> m_mipView1, m_mipView2, m_mipView3, m_mipView4;

    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_dsl;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_set;
};

} // namespace somegi
