// TonemapPass RHI 实现 — 描述符 set=0 (3 bindings):
//   0: hdrColor (sampled image)
//   1: sampler  (linear clamp)
//   2: outLdr   (storage image)
// push constant: TonemapPC (16 bytes, hdrMode + exposure)

#include "renderer/core/tonemap_pass.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/vulkan/vk_sampler.h"
#include "core/shader.h"
#include <array>

namespace somegi {

namespace { struct TonemapPC { uint32_t hdrMode; float exposure; uint32_t pad0, pad1; };
static_assert(sizeof(TonemapPC) == 16); }

TonemapPass::~TonemapPass() = default;

void TonemapPass::init(rhi::RHIDevice& d, VkSampler linearSampler) {
    m_rhiDevice = &d;
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(d);
    m_sampler = rhi::VkRHISampler::createNonOwning(vkDevice, linearSampler);

    rhi::DescSetLayoutDesc layoutDesc; layoutDesc.debugName = "Tonemap";
    layoutDesc.bindings = {
        {0, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
        {1, rhi::DescriptorType::Sampler,       1, rhi::ShaderStage::Compute},
        {2, rhi::DescriptorType::StorageImage,  1, rhi::ShaderStage::Compute},
    };
    m_setLayout = d.createDescriptorSetLayout(layoutDesc);
    for (auto& s : m_sets) s = d.createDescriptorSet(*m_setLayout);

    rhi::ShaderDesc sd; sd.stage = rhi::ShaderStage::Compute; sd.entryPoint = "cs_main";
    auto shader = rhi::VkRHIShader::createFromFile(vkDevice, sd, shaderDir() / "tonemap" / "tonemap.spv");

    rhi::ComputePSODesc psoDesc; psoDesc.debugName = "Tonemap";
    psoDesc.computeShader = shader.get();
    psoDesc.descriptorSetLayouts = {m_setLayout.get()};
    psoDesc.pushConstants = {{rhi::ShaderStage::Compute, 0, sizeof(TonemapPC)}};
    m_pipeline = d.createComputePSO(psoDesc);
}

void TonemapPass::init(rhi::RHIDevice& d, rhi::RHISampler& linearSampler) {
    m_rhiDevice = &d;
    // D3D12 路径：直接使用已创建的 RHI sampler
    m_sampler.reset(static_cast<rhi::RHISampler*>(&linearSampler));
    m_sampler.release(); // 不拥有所有权

    rhi::DescSetLayoutDesc layoutDesc; layoutDesc.debugName = "Tonemap";
    layoutDesc.bindings = {
        {0, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
        {1, rhi::DescriptorType::Sampler,       1, rhi::ShaderStage::Compute},
        {2, rhi::DescriptorType::StorageImage,  1, rhi::ShaderStage::Compute},
    };
    m_setLayout = d.createDescriptorSetLayout(layoutDesc);
    for (auto& s : m_sets) s = d.createDescriptorSet(*m_setLayout);

    // D3D12 shader 加载
    rhi::ShaderDesc sd; sd.stage = rhi::ShaderStage::Compute; sd.entryPoint = "cs_main";
    // Phase 5: 加载 DXIL shader（当前 PSO 创建需要 DXIL bytecode）
    // auto shader = d.createShader(sd, ...);
    // m_pipeline = d.createComputePSO(...);
}

void TonemapPass::destroy() {
    for (auto& s : m_sets) s.reset();
    m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice = nullptr;
}

void TonemapPass::bindOutput(VkImageView outView, uint32_t frameIdx) {
    if (!m_sets[frameIdx]) return;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto ao = rhi::VkRHITextureView::createNonOwning(vkD, outView);
    m_sets[frameIdx]->write({{2, rhi::DescriptorType::StorageImage, ao.get()}});
}

void TonemapPass::bindTargets(const RenderTargets& rt) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto hdrView = rhi::VkRHITextureView::createNonOwning(vkD, rt.hdrColor.view());
    auto ldrView = rhi::VkRHITextureView::createNonOwning(vkD, rt.ldrTonemap.view());
    for (uint32_t fi = 0; fi < 2; ++fi)
        m_sets[fi]->write({
            {0, rhi::DescriptorType::SampledImage, hdrView.get()},
            {1, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, m_sampler.get()},
            {2, rhi::DescriptorType::StorageImage, ldrView.get()},
        });
}

void TonemapPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt, uint32_t fi,
                          bool hdrMode, float exposure) {
    if (!m_pipeline || !m_sets[fi]) return;
    cmd.bindPipelineState(*m_pipeline);
    cmd.bindDescriptorSet(0, *m_sets[fi]);
    TonemapPC pc{};
    pc.hdrMode = hdrMode ? 1u : 0u; pc.exposure = exposure;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
    cmd.dispatch((rt.extent.width+7)/8, (rt.extent.height+7)/8, 1);
}

void TonemapPass::record(VkCommandBuffer vkCmd, const RenderTargets& rt, uint32_t fi,
                          bool hdrMode, float exposure) {
    rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice), vkCmd);
    record(rhiCmd, rt, fi, hdrMode, exposure);
}

} // namespace somegi
