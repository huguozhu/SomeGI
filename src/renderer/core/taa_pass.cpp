// TaaPass RHI 实现 — 4 bindings: curr/prev/depth (sampled) + output (storage).

#include "renderer/core/taa_pass.h"
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

namespace somegi {

namespace { struct TaaPC {
    float jitterX, jitterY, prevJitterX, prevJitterY;
    glm::mat4 invViewProj, prevViewProj;
    float blendAlpha, _pad, invResX, invResY;
}; static_assert(sizeof(TaaPC) == 160); }

TaaPass::~TaaPass() = default;

void TaaPass::init(rhi::RHIDevice& d) {
    m_rhiDevice = &d;
    rhi::DescSetLayoutDesc ld; ld.debugName = "TAA";
    ld.bindings = {
        {0, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
        {1, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
        {2, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
        {3, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},
    };
    m_setLayout = d.createDescriptorSetLayout(ld);
    for (auto& s : m_sets) s = d.createDescriptorSet(*m_setLayout);

    auto& vkD = static_cast<rhi::VkRHIDevice&>(d);
    rhi::ShaderDesc sd; sd.stage = rhi::ShaderStage::Compute; sd.entryPoint = "cs_main";
    auto sh = rhi::VkRHIShader::createFromFile(vkD, sd, shaderDir() / "aa" / "taa.spv");
    rhi::ComputePSODesc pd; pd.debugName = "TAA"; pd.computeShader = sh.get();
    pd.descriptorSetLayouts = {m_setLayout.get()};
    pd.pushConstants = {{rhi::ShaderStage::Compute, 0, sizeof(TaaPC)}};
    m_pipeline = d.createComputePSO(pd);
}

void TaaPass::destroy() {
    for (auto& s : m_sets) s.reset();
    m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice = nullptr;
}

void TaaPass::bindResources(const RenderTargets& rt, uint32_t fi) {
    if (!m_sets[fi]) return;
    m_sets[fi]->write({
        {0, rhi::DescriptorType::SampledImage, rt.rhiAaHdrView()},
        {1, rhi::DescriptorType::SampledImage, rt.rhiAaHistoryView()},
        {2, rhi::DescriptorType::SampledImage, rt.rhiDepthView()},
        {3, rhi::DescriptorType::StorageImage, rt.rhiLdrTonemapView()},
    });
}

void TaaPass::bindOutput(VkImageView outView, uint32_t fi) {
    if (!m_sets[fi]) return;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto ov = rhi::VkRHITextureView::createNonOwning(vkD, outView);
    m_sets[fi]->write({{3, rhi::DescriptorType::StorageImage, ov.get()}});
}

void TaaPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt,
                      const glm::vec2& j, const glm::vec2& pj,
                      const glm::mat4& ivp, const glm::mat4& pvp,
                      uint32_t fi, float ba) {
    if (!m_pipeline || !m_sets[fi]) return;
    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0, *m_sets[fi]);
    TaaPC pc{};
    pc.jitterX=j.x; pc.jitterY=j.y; pc.prevJitterX=pj.x; pc.prevJitterY=pj.y;
    pc.invViewProj=ivp; pc.prevViewProj=pvp; pc.blendAlpha=ba;
    pc.invResX=1.0f/rt.extent.width; pc.invResY=1.0f/rt.extent.height;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
    cmd.dispatch((rt.extent.width+7)/8, (rt.extent.height+7)/8, 1);
}


} // namespace somegi
