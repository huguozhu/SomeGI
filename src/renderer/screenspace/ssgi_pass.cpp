// SsgiPass RHI 实现 — 描述符 set=0 (7 bindings):
//   0: FrameUBO     (uniform buffer)
//   1: gNormalRough (sampled image)
//   2: gDepth       (sampled image)
//   3: gPrevHdr     (sampled image)
//   4: linearClamp  (sampler)
//   5: gOutSsgi     (storage image)
//   6: gPrevSsgi    (sampled image, history)

#include "renderer/screenspace/ssgi_pass.h"
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
#include "core/path_util.h"
#include <array>
#include <cstring>

namespace somegi {

namespace { struct SsgiPC {
    uint32_t outSizeX, outSizeY; float invOutSizeX, invOutSizeY;
    uint32_t maxSteps; float maxDist, thickness; uint32_t sampleCount;
}; static_assert(sizeof(SsgiPC) == 32); }

SsgiPass::~SsgiPass() = default;

void SsgiPass::init(rhi::RHIDevice& d) {
    m_rhiDevice = &d;
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(d);

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    m_linearClamp = d.createSampler({rhi::Filter::Linear,rhi::Filter::Linear,rhi::SamplerMipmapMode::Linear,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,0.f});

    rhi::DescSetLayoutDesc layoutDesc; layoutDesc.debugName = "SSGI";
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
    auto shader = rhi::VkRHIShader::createFromFile(vkDevice, sd, shaderDir() / "ssgi" / "ssgi.spv");

    rhi::ComputePSODesc psoDesc; psoDesc.debugName = "SSGI";
    psoDesc.computeShader = shader.get();
    psoDesc.descriptorSetLayouts = {m_setLayout.get()};
    psoDesc.pushConstants = {{rhi::ShaderStage::Compute, 0, sizeof(SsgiPC)}};
    m_pipeline = d.createComputePSO(psoDesc);
}

void SsgiPass::destroy() {
    m_linearClamp.reset();
    m_set.reset(); m_pipeline.reset(); m_setLayout.reset();
    m_rhiDevice = nullptr;
}

void SsgiPass::bindFrame(const RenderTargets& rt, VkBuffer frameUbo) {
    if (!m_set) return;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto ubo = rhi::VkRHIBuffer::createNonOwning(vkD, frameUbo, VK_WHOLE_SIZE);
    m_set->write({
        {0, rhi::DescriptorType::UniformBuffer, nullptr, ubo.get()},
        {1, rhi::DescriptorType::SampledImage,  rt.rhiGNormalRoughView()},
        {2, rhi::DescriptorType::SampledImage,  rt.rhiDepthView()},
        {3, rhi::DescriptorType::SampledImage,  rt.rhiHdrPrevView()},
        {4, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, m_linearClamp.get()},
        {5, rhi::DescriptorType::StorageImage,  rt.rhiSsgiView()},
        {6, rhi::DescriptorType::SampledImage,  rt.rhiSsgiPrevView()},
    });
}

void SsgiPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt) {
    if (!m_pipeline || !m_set) return;
    cmd.bindPipelineState(*m_pipeline);
    cmd.bindDescriptorSet(0, *m_set);
    SsgiPC pc{}; pc.outSizeX = rt.extent.width; pc.outSizeY = rt.extent.height;
    pc.invOutSizeX = 1.0f/(float)rt.extent.width; pc.invOutSizeY = 1.0f/(float)rt.extent.height;
    pc.maxSteps = (uint32_t)maxSteps; pc.maxDist = maxDist; pc.thickness = thickness;
    pc.sampleCount = (uint32_t)sampleCount;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
    cmd.dispatch((rt.extent.width+7)/8, (rt.extent.height+7)/8, 1);
}


} // namespace somegi
