// SmaaPass RHI 实现 — 双 Compute Pass: edge detection → blending。

#include "renderer/core/smaa_pass.h"
#include "renderer/core/render_targets.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_command.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>

namespace somegi {

SmaaPass::~SmaaPass() = default;

void SmaaPass::init(Device& dev, rhi::RHIDevice& d, VkExtent2D ext) {
    m_rhiDevice = &d;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(d);

    // Edge texture (仍需 core::Device 构造)
    ImageDesc ed; ed.format = VK_FORMAT_R16G16_SFLOAT;
    ed.extent = {ext.width, ext.height, 1};
    ed.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    m_edgeTex = Image(dev, ed);

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
    m_edgeTex.reset(); m_rhiDevice = nullptr;
}

void SmaaPass::bindResources(const RenderTargets& rt) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    // Edge set
    {
        auto in = rhi::VkRHITextureView::createNonOwning(vkD, rt.aaHdr.view());
        auto eo = rhi::VkRHITextureView::createNonOwning(vkD, m_edgeTex.view());
        m_edgeSet->write({
            {0, rhi::DescriptorType::SampledImage, in.get()},
            {1, rhi::DescriptorType::StorageImage, eo.get()},
        });
    }

    // Blend set
    {
        auto in = rhi::VkRHITextureView::createNonOwning(vkD, rt.aaHdr.view());
        auto ei = rhi::VkRHITextureView::createNonOwning(vkD, m_edgeTex.view());
        auto out = rhi::VkRHITextureView::createNonOwning(vkD, rt.ldrTonemap.view());
        m_blendSet->write({
            {0, rhi::DescriptorType::SampledImage, in.get()},
            {1, rhi::DescriptorType::SampledImage, ei.get()},
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
    VkCommandBuffer vkCmd = (VkCommandBuffer)(uintptr_t)cmd.nativeHandle();
    uint32_t gx = (rt.extent.width+7)/8, gy = (rt.extent.height+7)/8;

    auto edgeBarrier = [&](VkImageLayout oldL, VkImageLayout newL,
                           VkAccessFlags2 sa, VkAccessFlags2 da,
                           VkPipelineStageFlags2 ss, VkPipelineStageFlags2 ds) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask=ss; b.srcAccessMask=sa; b.dstStageMask=ds; b.dstAccessMask=da;
        b.oldLayout=oldL; b.newLayout=newL; b.image=m_edgeTex.image();
        b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount=1; di.pImageMemoryBarriers=&b;
        vkCmdPipelineBarrier2(vkCmd, &di);
    };

    // Pass 1: Edge detection
    edgeBarrier(m_edgeLayout, VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    m_edgeLayout = VK_IMAGE_LAYOUT_GENERAL;

    cmd.bindPipelineState(*m_edgePipeline);
    cmd.bindDescriptorSet(0, *m_edgeSet);
    cmd.dispatch(gx, gy, 1);

    // Barrier: edge → read-only
    edgeBarrier(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    m_edgeLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Pass 2: Blending
    cmd.bindPipelineState(*m_blendPipeline);
    cmd.bindDescriptorSet(0, *m_blendSet);
    cmd.dispatch(gx, gy, 1);

    // Barrier: edge back to GENERAL
    edgeBarrier(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
    m_edgeLayout = VK_IMAGE_LAYOUT_GENERAL;
}

void SmaaPass::record(VkCommandBuffer vkCmd, const RenderTargets& rt) {
    rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice), vkCmd);
    record(rhiCmd, rt);
}

} // namespace somegi
