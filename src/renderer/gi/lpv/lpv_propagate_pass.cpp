// LpvPropagatePass RHI — 7 bindings: 3 src + 3 dst + GV, 2 ping-pong sets.

#include "renderer/gi/lpv/lpv_propagate_pass.h"
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

namespace somegi {
namespace { struct PropagatePC { uint32_t gridResolution; float occAmp, gvOccStr, _p1; };
static_assert(sizeof(PropagatePC) == 16); }

LpvPropagatePass::~LpvPropagatePass() = default;

void LpvPropagatePass::init(rhi::RHIDevice& d) {
    m_rhiDevice = &d;
    rhi::DescSetLayoutDesc ld; ld.debugName = "LpvPropagate";
    ld.bindings = {
        {0, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute}, {1, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
        {2, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute}, {3, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},
        {4, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},   {5, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},
        {6, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
    };
    m_setLayout = d.createDescriptorSetLayout(ld);
    for (auto& s : m_sets) s = d.createDescriptorSet(*m_setLayout);

    auto& vkD = static_cast<rhi::VkRHIDevice&>(d);
    rhi::ShaderDesc sd; sd.stage = rhi::ShaderStage::Compute; sd.entryPoint = "cs_main";
    auto sh = rhi::VkRHIShader::createFromFile(vkD, sd, shaderDir()/"gi"/"lpv"/"lpv_propagate.spv");
    rhi::ComputePSODesc pd; pd.debugName="LpvPropagate"; pd.computeShader=sh.get(); pd.descriptorSetLayouts={m_setLayout.get()};
    pd.pushConstants = {{rhi::ShaderStage::Compute, 0, sizeof(PropagatePC)}};
    m_pipeline = d.createComputePSO(pd);
}

void LpvPropagatePass::destroy() { for(auto& s:m_sets)s.reset(); m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice=nullptr; }

void LpvPropagatePass::bindResources(const LpvGrid& g0, const LpvGrid& g1, const Image& gv) {
    auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto writeOne = [&](int si, const LpvGrid& src, const LpvGrid& dst) {
        m_sets[si]->write({
            {0, rhi::DescriptorType::SampledImage, rhi::VkRHITextureView::createNonOwning(vkD,src.lpvR.view()).get()},
            {1, rhi::DescriptorType::SampledImage, rhi::VkRHITextureView::createNonOwning(vkD,src.lpvG.view()).get()},
            {2, rhi::DescriptorType::SampledImage, rhi::VkRHITextureView::createNonOwning(vkD,src.lpvB.view()).get()},
            {3, rhi::DescriptorType::StorageImage, rhi::VkRHITextureView::createNonOwning(vkD,dst.lpvR.view()).get()},
            {4, rhi::DescriptorType::StorageImage, rhi::VkRHITextureView::createNonOwning(vkD,dst.lpvG.view()).get()},
            {5, rhi::DescriptorType::StorageImage, rhi::VkRHITextureView::createNonOwning(vkD,dst.lpvB.view()).get()},
            {6, rhi::DescriptorType::SampledImage, rhi::VkRHITextureView::createNonOwning(vkD,gv.view()).get()},
        });
    };
    writeOne(0, g0, g1); writeOne(1, g1, g0);
}

void LpvPropagatePass::record(rhi::RHICommandBuffer& cmd, int si, uint32_t gr, float oa, float gs) {
    if(!m_pipeline||!m_sets[si&1])return;
    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0,*m_sets[si&1]);
    PropagatePC pc{gr,oa,gs,0};
    cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc));
    cmd.dispatch((gr+3)/4,(gr+3)/4,(gr+3)/4);
}
void LpvPropagatePass::record(VkCommandBuffer vkCmd, int si, uint32_t gr, float oa, float gs) {
    rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice),vkCmd); record(rhiCmd,si,gr,oa,gs);
}
} // namespace somegi
