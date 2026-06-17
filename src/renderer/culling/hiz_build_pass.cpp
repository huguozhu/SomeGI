// HiZBuildPass RHI 实现 — 描述符 set=0:
//   binding 0: depth (sampled image)
//   binding 1-4: mip1-mip4 (storage images)
// push constant: { uint32_t w, h } (屏幕尺寸)

#include "renderer/culling/hiz_build_pass.h"
#include "renderer/core/render_targets.h"
#include "core/device.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_command.h"
#include "core/shader.h"
#include <array>
#include <algorithm>

namespace somegi {

static Image mkHiZMip(Device& d, uint32_t w, uint32_t h) {
    ImageDesc id{};
    id.format = VK_FORMAT_R32_SFLOAT;
    id.extent = {w, h, 1};
    id.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    return Image(d, id);
}

HiZBuildPass::~HiZBuildPass() = default;

void HiZBuildPass::init(Device& d, rhi::RHIDevice& rhiDevice, VkExtent2D extent) {
    m_device = &d;
    m_rhiDevice = &rhiDevice;
    m_extent = extent;

    // Hi-Z mip 图像（仍使用 core::Image 管理 GPU 内存）
    uint32_t w = extent.width, h = extent.height;
    m_mip1 = mkHiZMip(d, std::max(1u, w/2), std::max(1u, h/2));
    m_mip2 = mkHiZMip(d, std::max(1u, w/4), std::max(1u, h/4));
    m_mip3 = mkHiZMip(d, std::max(1u, w/8), std::max(1u, h/8));
    m_mip4 = mkHiZMip(d, std::max(1u, w/16), std::max(1u, h/16));

    // ── Descriptor Set Layout ──
    rhi::DescSetLayoutDesc layoutDesc;
    layoutDesc.debugName = "HiZBuild";
    layoutDesc.bindings = {
        {0, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},  // depth
        {1, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},  // mip1
        {2, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},  // mip2
        {3, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},  // mip3
        {4, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},  // mip4
    };
    m_dsl = rhiDevice.createDescriptorSetLayout(layoutDesc);

    // ── Descriptor Set ──
    m_set = rhiDevice.createDescriptorSet(*m_dsl);

    // ── Compute PSO ──
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(rhiDevice);
    rhi::ShaderDesc shaderDesc;
    shaderDesc.stage = rhi::ShaderStage::Compute;
    shaderDesc.entryPoint = "cs_main";
    auto shader = rhi::VkRHIShader::createFromFile(vkDevice, shaderDesc,
        shaderDir() / "culling" / "hiz_build.spv");

    rhi::ComputePSODesc psoDesc;
    psoDesc.debugName = "HiZBuild";
    psoDesc.computeShader = shader.get();
    psoDesc.descriptorSetLayouts = {m_dsl.get()};
    psoDesc.pushConstants = {
        {rhi::ShaderStage::Compute, 0, 8}  // {uint32_t w, h}
    };
    m_pipeline = rhiDevice.createComputePSO(psoDesc);
}

void HiZBuildPass::destroy() {
    m_set.reset();
    m_pipeline.reset();
    m_dsl.reset();
    m_mip1.reset(); m_mip2.reset(); m_mip3.reset(); m_mip4.reset();
    m_device = nullptr;
    m_rhiDevice = nullptr;
}

void HiZBuildPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt) {
    if (!m_set || !m_pipeline) return;
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    // ── 写描述符集（非拥有型视图包装） ──
    auto depthView = rhi::VkRHITextureView::createNonOwning(vkDevice, rt.depth.view());
    auto m1View    = rhi::VkRHITextureView::createNonOwning(vkDevice, m_mip1.view());
    auto m2View    = rhi::VkRHITextureView::createNonOwning(vkDevice, m_mip2.view());
    auto m3View    = rhi::VkRHITextureView::createNonOwning(vkDevice, m_mip3.view());
    auto m4View    = rhi::VkRHITextureView::createNonOwning(vkDevice, m_mip4.view());

    m_set->write({
        {0, rhi::DescriptorType::SampledImage, depthView.get()},
        {1, rhi::DescriptorType::StorageImage, m1View.get()},
        {2, rhi::DescriptorType::StorageImage, m2View.get()},
        {3, rhi::DescriptorType::StorageImage, m3View.get()},
        {4, rhi::DescriptorType::StorageImage, m4View.get()},
    });

    // ── Depth barrier（执行同步，无需 layout 转换） ──
    {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        b.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.image = rt.depth.image();
        b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2((VkCommandBuffer)(uintptr_t)cmd.nativeHandle(), &di);
    }

    // ── Dispatch ──
    cmd.bindPipelineState(*m_pipeline);
    cmd.bindDescriptorSet(0, *m_set);
    struct { uint32_t w, h; } pc{m_extent.width, m_extent.height};
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, 8);
    cmd.dispatch((m_extent.width + 15) / 16, (m_extent.height + 15) / 16, 1);
}

// 兼容 VkCommandBuffer 重载
void HiZBuildPass::record(VkCommandBuffer vkCmd, const RenderTargets& rt) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
    record(rhiCmd, rt);
}

} // namespace somegi
