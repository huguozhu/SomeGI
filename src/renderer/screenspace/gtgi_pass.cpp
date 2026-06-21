// GtgiPass RHI 实现 — 描述符布局与 SsgiPass 一致 (7 bindings)。

#include "renderer/screenspace/gtgi_pass.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/base/sampler.h"
#include "core/device.h"
#include "core/shader.h"
#include <cstring>

namespace somegi {

namespace { struct GtgiPC {
    uint32_t outSizeX, outSizeY; float invOutSizeX, invOutSizeY;
    float radiusPixels, falloff; uint32_t sliceCount, samplesPerSlice;
}; static_assert(sizeof(GtgiPC) == 32); }

GtgiPass::~GtgiPass() = default;

void GtgiPass::init(rhi::RHIDevice& d) {
    m_rhiDevice = &d;
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(d);

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR; si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    m_linearClamp = d.createSampler({rhi::Filter::Linear,rhi::Filter::Linear,rhi::SamplerMipmapMode::Linear,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,0.f});

    rhi::DescSetLayoutDesc layoutDesc; layoutDesc.debugName = "GTGI";
    layoutDesc.bindings = {
        {0, rhi::DescriptorType::UniformBuffer, 1, rhi::ShaderStage::Compute},
        {1, rhi::DescriptorType::SampledImage,  1, rhi::ShaderStage::Compute},
        {2, rhi::DescriptorType::SampledImage,  1, rhi::ShaderStage::Compute},
        {3, rhi::DescriptorType::SampledImage,  1, rhi::ShaderStage::Compute},
        {4, rhi::DescriptorType::Sampler,        1, rhi::ShaderStage::Compute},
        {5, rhi::DescriptorType::StorageImage,   1, rhi::ShaderStage::Compute},
        {6, rhi::DescriptorType::SampledImage,   1, rhi::ShaderStage::Compute},
    };
    m_setLayout = d.createDescriptorSetLayout(layoutDesc);
    m_set = d.createDescriptorSet(*m_setLayout);

    rhi::ShaderDesc sd; sd.stage = rhi::ShaderStage::Compute; sd.entryPoint = "cs_main";
    auto shader = rhi::VkRHIShader::createFromFile(vkDevice, sd, shaderDir() / "gi" / "gtgi" / "gtgi.spv");

    rhi::ComputePSODesc psoDesc; psoDesc.debugName = "GTGI";
    psoDesc.computeShader = shader.get();
    psoDesc.descriptorSetLayouts = {m_setLayout.get()};
    psoDesc.pushConstants = {{rhi::ShaderStage::Compute, 0, sizeof(GtgiPC)}};
    m_pipeline = d.createComputePSO(psoDesc);
}

void GtgiPass::destroy() {
    m_linearClamp.reset();
    m_set.reset(); m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice = nullptr;
}

void GtgiPass::bindFrame(const RenderTargets& rt, VkBuffer frameUbo) {
    if (!m_set) return;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto ubo = rhi::VkRHIBuffer::createNonOwning(vkD, frameUbo, VK_WHOLE_SIZE);
    m_set->write({
        {0, rhi::DescriptorType::UniformBuffer, nullptr, ubo.get()},
        {1, rhi::DescriptorType::SampledImage,  rhi::VkRHITextureView::createNonOwning(vkD, rt.gNormalRough.view()).get()},
        {2, rhi::DescriptorType::SampledImage,  rhi::VkRHITextureView::createNonOwning(vkD, rt.depth.view()).get()},
        {3, rhi::DescriptorType::SampledImage,  rhi::VkRHITextureView::createNonOwning(vkD, rt.hdrPrev.view()).get()},
        {4, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, m_linearClamp.get()},
        {5, rhi::DescriptorType::StorageImage,  rhi::VkRHITextureView::createNonOwning(vkD, rt.ssgi.view()).get()},
        {6, rhi::DescriptorType::SampledImage,  rhi::VkRHITextureView::createNonOwning(vkD, rt.ssgiPrev.view()).get()},
    });
}

void GtgiPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt) {
    if (!m_pipeline || !m_set) return;
    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0, *m_set);
    GtgiPC pc{}; pc.outSizeX = rt.extent.width; pc.outSizeY = rt.extent.height;
    pc.invOutSizeX = 1.0f/(float)rt.extent.width; pc.invOutSizeY = 1.0f/(float)rt.extent.height;
    pc.radiusPixels = radiusPixels; pc.falloff = falloff;
    pc.sliceCount = (uint32_t)sliceCount; pc.samplesPerSlice = (uint32_t)samplesPerSlice;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
    cmd.dispatch((rt.extent.width+7)/8, (rt.extent.height+7)/8, 1);
}


} // namespace somegi
