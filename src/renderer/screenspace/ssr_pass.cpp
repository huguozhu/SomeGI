// SsrPass RHI 实现 — 描述符 set=0 (6 bindings):
//   0: FrameUBO      (uniform buffer)
//   1: gNormalRough  (sampled image)
//   2: gDepth        (sampled image)
//   3: gPrevHdr      (sampled image)
//   4: gLinearClamp  (sampler)
//   5: gOutSsr       (storage image)
// push constant: SsrPC (32 bytes)

#include "renderer/screenspace/ssr_pass.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_command.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>
#include <cstring>

namespace somegi {

namespace {
struct SsrPC {
    uint32_t outSizeX, outSizeY;
    float invOutSizeX, invOutSizeY;
    uint32_t maxSteps;
    float maxDist, thickness, roughThreshold;
};
static_assert(sizeof(SsrPC) == 32);
}

SsrPass::~SsrPass() = default;

void SsrPass::init(rhi::RHIDevice& d) {
    m_rhiDevice = &d;
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(d);

    // Linear-clamp sampler
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    vkCreateSampler(vkDevice.vkDevice(), &si, nullptr, &m_linearClamp);

    rhi::DescSetLayoutDesc layoutDesc;
    layoutDesc.debugName = "SSR";
    layoutDesc.bindings = {
        {0, rhi::DescriptorType::UniformBuffer, 1, rhi::ShaderStage::Compute},
        {1, rhi::DescriptorType::SampledImage,  1, rhi::ShaderStage::Compute},
        {2, rhi::DescriptorType::SampledImage,  1, rhi::ShaderStage::Compute},
        {3, rhi::DescriptorType::SampledImage,  1, rhi::ShaderStage::Compute},
        {4, rhi::DescriptorType::Sampler,        1, rhi::ShaderStage::Compute},
        {5, rhi::DescriptorType::StorageImage,   1, rhi::ShaderStage::Compute},
    };
    m_setLayout = d.createDescriptorSetLayout(layoutDesc);
    m_set = d.createDescriptorSet(*m_setLayout);

    rhi::ShaderDesc shaderDesc;
    shaderDesc.stage = rhi::ShaderStage::Compute;
    shaderDesc.entryPoint = "cs_main";
    auto shader = rhi::VkRHIShader::createFromFile(vkDevice, shaderDesc,
        shaderDir() / "ssr" / "ssr.spv");

    rhi::ComputePSODesc psoDesc;
    psoDesc.debugName = "SSR";
    psoDesc.computeShader = shader.get();
    psoDesc.descriptorSetLayouts = {m_setLayout.get()};
    psoDesc.pushConstants = {{rhi::ShaderStage::Compute, 0, sizeof(SsrPC)}};
    m_pipeline = d.createComputePSO(psoDesc);
}

void SsrPass::destroy() {
    if (m_linearClamp) {
        auto& vkDevice = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
        vkDestroySampler(vkDevice.vkDevice(), m_linearClamp, nullptr);
        m_linearClamp = VK_NULL_HANDLE;
    }
    m_set.reset();
    m_pipeline.reset();
    m_setLayout.reset();
    m_rhiDevice = nullptr;
}

void SsrPass::bindFrame(const RenderTargets& rt, VkBuffer frameUbo) {
    if (!m_set) return;
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    auto uboRHI = rhi::VkRHIBuffer::createNonOwning(vkDevice, frameUbo, VK_WHOLE_SIZE);
    auto nrView = rhi::VkRHITextureView::createNonOwning(vkDevice, rt.gNormalRough.view());
    auto dpView = rhi::VkRHITextureView::createNonOwning(vkDevice, rt.depth.view());
    auto hpView = rhi::VkRHITextureView::createNonOwning(vkDevice, rt.hdrPrev.view());
    auto srView = rhi::VkRHITextureView::createNonOwning(vkDevice, rt.ssr.view());

    m_set->write({
        {0, rhi::DescriptorType::UniformBuffer, nullptr, uboRHI.get()},
        {1, rhi::DescriptorType::SampledImage,  nrView.get()},
        {2, rhi::DescriptorType::SampledImage,  dpView.get()},
        {3, rhi::DescriptorType::SampledImage,  hpView.get()},
        {4, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, (const void*)(uintptr_t)m_linearClamp},
        {5, rhi::DescriptorType::StorageImage,  srView.get()},
    });
}

void SsrPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt) {
    if (!m_pipeline || !m_set) return;

    cmd.bindPipelineState(*m_pipeline);
    cmd.bindDescriptorSet(0, *m_set);

    SsrPC pc{};
    pc.outSizeX = rt.extent.width;
    pc.outSizeY = rt.extent.height;
    pc.invOutSizeX = 1.0f / (float)rt.extent.width;
    pc.invOutSizeY = 1.0f / (float)rt.extent.height;
    pc.maxSteps = (uint32_t)maxSteps;
    pc.maxDist = maxDist;
    pc.thickness = thickness;
    pc.roughThreshold = roughThreshold;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));

    uint32_t gx = (rt.extent.width  + 7) / 8;
    uint32_t gy = (rt.extent.height + 7) / 8;
    cmd.dispatch(gx, gy, 1);
}

void SsrPass::record(VkCommandBuffer vkCmd, const RenderTargets& rt) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
    record(rhiCmd, rt);
}

} // namespace somegi
