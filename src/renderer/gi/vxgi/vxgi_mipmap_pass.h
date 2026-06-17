// VxgiMipmapPass — 体素 mipmap 降采样 (Compute)，已迁移到 RHI。
#pragma once
#include "renderer/gi/vxgi/vxgi_resources.h"
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

namespace somegi {
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }

class VxgiMipmapPass {
public:
    ~VxgiMipmapPass();
    void init(rhi::RHIDevice& d, uint32_t mipLevels);
    void destroy();
    void bindResources(const VxgiResources& vxgi);
    void record(rhi::RHICommandBuffer& cmd, const VxgiResources& vxgi);
    void record(VkCommandBuffer cmd, const VxgiResources& vxgi);

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    uint32_t m_mipLevels = 0;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::vector<std::unique_ptr<rhi::RHIDescriptorSet>> m_sets;
};

} // namespace somegi
