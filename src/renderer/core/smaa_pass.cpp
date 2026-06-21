// SmaaPass RHI 实现 — 双 Compute Pass: edge detection → blending。

#include "renderer/core/smaa_pass.h"
#include "renderer/core/render_targets.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/base/texture.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_command.h"
#include "core/shader.h"
#include <array>

namespace somegi {

SmaaPass::~SmaaPass() = default;

void SmaaPass::init(rhi::RHIDevice& d, VkExtent2D ext) {
    m_rhiDevice = &d;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(d);

    // Edge texture (纯 RHI 创建)
    rhi::TextureDesc ed;
    ed.format = rhi::Format::R16G16_SFLOAT;
    ed.width = ext.width; ed.height = ext.height; ed.depth = 1;
    ed.usage = rhi::TextureUsage::Storage | rhi::TextureUsage::Sampled;
    ed.debugName = "SMAA_Edge";
    m_edgeTex = d.createTexture(ed);
    m_edgeView = d.createTextureView(*m_edgeTex, {});

    rhi::ShaderDesc sd; sd.stage = rhi::ShaderStage::Compute; sd.entryPoint = "cs_main";

    // Edge detection pipeline
    {
        rhi::DescSetLayoutDesc ld; ld.debugName = "SMAA_Edge";
        ld.bindings = {
            {0, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {1, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},
        };
        m_edgeSetLayout = d.createDescriptorSetLayout(ld);
        m_edgeSet = d.createDescriptorSet(*m_edgeSetLayout);

        auto sh = rhi::VkRHIShader::createFromFile(vkD, sd, shaderDir() / "aa" / "smaa_edge.spv");
        rhi::ComputePSODesc pd; pd.debugName="SMAA_Edge"; pd.computeShader=sh.get();
        pd.descriptorSetLayouts = {m_edgeSetLayout.get()};
        m_edgePipeline = d.createComputePSO(pd);
    }

    // Blending pipeline
    {
        rhi::DescSetLayoutDesc ld; ld.debugName = "SMAA_Blend";
        ld.bindings = {
            {0, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {1, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {2, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},
        };
        m_blendSetLayout = d.createDescriptorSetLayout(ld);
        m_blendSet = d.createDescriptorSet(*m_blendSetLayout);

        auto sh = rhi::VkRHIShader::createFromFile(vkD, sd, shaderDir() / "aa" / "smaa_blend.spv");
        rhi::ComputePSODesc pd; pd.debugName="SMAA_Blend"; pd.computeShader=sh.get();
        pd.descriptorSetLayouts = {m_blendSetLayout.get()};
        m_blendPipeline = d.createComputePSO(pd);
    }
}

void SmaaPass::destroy() {
    m_blendSet.reset(); m_blendPipeline.reset(); m_blendSetLayout.reset();
    m_edgeSet.reset(); m_edgePipeline.reset(); m_edgeSetLayout.reset();
    m_edgeView.reset(); m_edgeTex.reset(); m_rhiDevice = nullptr;
}

void SmaaPass::bindResources(const RenderTargets& rt) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    // Edge set
    {
        auto in = rhi::VkRHITextureView::createNonOwning(vkD, rt.aaHdr.view());
        m_edgeSet->write({
            {0, rhi::DescriptorType::SampledImage, in.get()},
            {1, rhi::DescriptorType::StorageImage, m_edgeView.get()},
        });
    }

    // Blend set
    {
        auto in = rhi::VkRHITextureView::createNonOwning(vkD, rt.aaHdr.view());
        auto out = rhi::VkRHITextureView::createNonOwning(vkD, rt.ldrTonemap.view());
        m_blendSet->write({
            {0, rhi::DescriptorType::SampledImage, in.get()},
            {1, rhi::DescriptorType::SampledImage, m_edgeView.get()},
            {2, rhi::DescriptorType::StorageImage, out.get()},
        });
    }
}

void SmaaPass::bindOutput(VkImageView outView) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto out = rhi::VkRHITextureView::createNonOwning(vkD, outView);
    m_blendSet->write({{2, rhi::DescriptorType::StorageImage, out.get()}});
}

void SmaaPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt) {
    uint32_t gx = (rt.extent.width+7)/8, gy = (rt.extent.height+7)/8;

    // Pass 1: Edge detection
    cmd.textureBarrier(*m_edgeTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::General);

    cmd.bindPipelineState(*m_edgePipeline);
    cmd.bindDescriptorSet(0, *m_edgeSet);
    cmd.dispatch(gx, gy, 1);

    // Barrier: edge → read-only
    cmd.textureBarrier(*m_edgeTex, rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);

    // Pass 2: Blending
    cmd.bindPipelineState(*m_blendPipeline);
    cmd.bindDescriptorSet(0, *m_blendSet);
    cmd.dispatch(gx, gy, 1);

    // Barrier: edge back to General
    cmd.textureBarrier(*m_edgeTex, rhi::TextureLayout::ShaderReadOnly, rhi::TextureLayout::General);
}

} // namespace somegi
