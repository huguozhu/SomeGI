// HiZBuildPass RHI 实现 — 描述符 set=0:
//   binding 0: depth (sampled image)
//   binding 1-4: mip1-mip4 (storage images)
// push constant: { uint32_t w, h } (屏幕尺寸)

#include "renderer/culling/hiz_build_pass.h"
#include "renderer/core/render_targets.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_command.h"
#include "core/path_util.h"
#include <array>
#include <algorithm>

namespace somegi {

static std::unique_ptr<rhi::RHITexture> mkHiZMip(rhi::RHIDevice& d, uint32_t w, uint32_t h) {
    rhi::TextureDesc td{};
    td.format = rhi::Format::R32_SFLOAT;
    td.width = w; td.height = h; td.depth = 1;
    td.usage = static_cast<rhi::TextureUsage>(
        static_cast<uint32_t>(rhi::TextureUsage::Storage) |
        static_cast<uint32_t>(rhi::TextureUsage::Sampled));
    return d.createTexture(td);
}

HiZBuildPass::~HiZBuildPass() = default;

void HiZBuildPass::init(rhi::RHIDevice& d, VkExtent2D extent) {
    m_rhiDevice = &d;
    m_extent = extent;

    // Hi-Z mip 纹理（通过 RHI 创建）
    uint32_t w = extent.width, h = extent.height;
    auto createMip = [&](uint32_t mw, uint32_t mh,
                         std::unique_ptr<rhi::RHITexture>& tex,
                         std::unique_ptr<rhi::RHITextureView>& view) {
        tex = mkHiZMip(d, mw, mh);
        view = d.createTextureView(*tex, {});
    };
    createMip(std::max(1u, w/2),  std::max(1u, h/2),  m_mipTex1, m_mipView1);
    createMip(std::max(1u, w/4),  std::max(1u, h/4),  m_mipTex2, m_mipView2);
    createMip(std::max(1u, w/8),  std::max(1u, h/8),  m_mipTex3, m_mipView3);
    createMip(std::max(1u, w/16), std::max(1u, h/16), m_mipTex4, m_mipView4);

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
    m_dsl = d.createDescriptorSetLayout(layoutDesc);

    // ── Descriptor Set ──
    m_set = d.createDescriptorSet(*m_dsl);

    // ── Compute PSO ──
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(d);
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
    m_pipeline = d.createComputePSO(psoDesc);
}

void HiZBuildPass::destroy() {
    m_set.reset();
    m_pipeline.reset();
    m_dsl.reset();
    m_mipView1.reset(); m_mipView2.reset(); m_mipView3.reset(); m_mipView4.reset();
    m_mipTex1.reset(); m_mipTex2.reset(); m_mipTex3.reset(); m_mipTex4.reset();
    m_rhiDevice = nullptr;
}

void HiZBuildPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt) {
    if (!m_set || !m_pipeline) return;
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    // ── 写描述符集（深度需要非拥有型包装，mip 直接用成员 view） ──
    auto depthView = rhi::VkRHITextureView::createNonOwning(vkDevice, rt.depth.view());

    m_set->write({
        {0, rhi::DescriptorType::SampledImage, depthView.get()},
        {1, rhi::DescriptorType::StorageImage, m_mipView1.get()},
        {2, rhi::DescriptorType::StorageImage, m_mipView2.get()},
        {3, rhi::DescriptorType::StorageImage, m_mipView3.get()},
        {4, rhi::DescriptorType::StorageImage, m_mipView4.get()},
    });

    // ── Depth execution barrier（确保上一阶段的深度写入对 compute 可见）──
    {
        auto depthRHI = rhi::VkRHITexture::createNonOwning(vkDevice, rt.depth.image(),
            rhi::Format::D32_SFLOAT, m_extent.width, m_extent.height);
        cmd.textureBarrier(*depthRHI, rhi::TextureLayout::ShaderReadOnly, rhi::TextureLayout::ShaderReadOnly);
    }

    // ── Dispatch ──
    cmd.bindPipelineState(*m_pipeline);
    cmd.bindDescriptorSet(0, *m_set);
    struct { uint32_t w, h; } pc{m_extent.width, m_extent.height};
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, 8);
    cmd.dispatch((m_extent.width + 15) / 16, (m_extent.height + 15) / 16, 1);
}

} // namespace somegi
