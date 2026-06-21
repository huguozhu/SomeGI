// LpvInjectPass RHI — 7 bindings: 3 RSM + 3 LPV + GV storage。
// barrier/clear 已迁移到 RHI textureBarrier + clearColor；VkCommandBuffer 重载仅作桥接。

#include "renderer/gi/lpv/lpv_inject_pass.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "core/path_util.h"
#include <array>

namespace somegi {
namespace { struct InjectPC { uint32_t rsmSizeX,rsmSizeY,gridRes,_pad; float gridMinX,gridMinY,gridMinZ,cellSize; };
static_assert(sizeof(InjectPC)==32); }

LpvInjectPass::~LpvInjectPass() = default;

void LpvInjectPass::init(rhi::RHIDevice& d, uint32_t rsmSize) {
    m_rhiDevice=&d; m_rsmSize=rsmSize;
    rhi::DescSetLayoutDesc ld; ld.debugName="LpvInject";
    ld.bindings={{0,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute},{5,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute},{6,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_setLayout=d.createDescriptorSetLayout(ld); m_set=d.createDescriptorSet(*m_setLayout);
    auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
    rhi::ShaderDesc sd; sd.stage=rhi::ShaderStage::Compute; sd.entryPoint="cs_main";
    auto sh=rhi::VkRHIShader::createFromFile(vkD,sd,shaderDir()/"gi"/"lpv"/"lpv_inject.spv");
    rhi::ComputePSODesc pd; pd.debugName="LpvInject"; pd.computeShader=sh.get(); pd.descriptorSetLayouts={m_setLayout.get()}; pd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(InjectPC)}};
    m_pipeline=d.createComputePSO(pd);
}

void LpvInjectPass::destroy() { m_set.reset(); m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice=nullptr; }

void LpvInjectPass::bindResources(VkImageView rsmPosView, VkImageView rsmNView, VkImageView rsmFluxView, const LpvGrid& grid, const Image& gv) {
    if(!m_set)return; auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_set->write({
        {0,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rsmPosView).get()},
        {1,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rsmNView).get()},
        {2,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rsmFluxView).get()},
        {3,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,grid.lpvR.view()).get()},
        {4,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,grid.lpvG.view()).get()},
        {5,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,grid.lpvB.view()).get()},
        {6,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,gv.view()).get()},
    });
    m_lpvR=&grid.lpvR; m_lpvG=&grid.lpvG; m_lpvB=&grid.lpvB; m_gv=&gv;
}

void LpvInjectPass::record(rhi::RHICommandBuffer& cmd, uint32_t gr, const glm::vec3& gMin, float cs) {
    if(!m_pipeline||!m_set)return;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::Format fmt = rhi::toRhiFormat(m_lpvR->format());
    uint32_t res = m_lpvR->extent().width;

    const Image* imgs[4] = {m_lpvR, m_lpvG, m_lpvB, m_gv};
    for(auto img : imgs) {
        auto tex = rhi::VkRHITexture::createNonOwning(vkD, img->image(), fmt, res, res, 1);
        cmd.textureBarrier(*tex, rhi::TextureLayout::Undefined, rhi::TextureLayout::TransferDst);
        cmd.clearColor(*tex, 0.0f, 0.0f, 0.0f, 0.0f);
        cmd.textureBarrier(*tex, rhi::TextureLayout::TransferDst, rhi::TextureLayout::General);
    }

    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0,*m_set);
    InjectPC pc{(uint32_t)m_rsmSize,(uint32_t)m_rsmSize,gr,0,gMin.x,gMin.y,gMin.z,cs};
    cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc));
    cmd.dispatch((m_rsmSize+7)/8,(m_rsmSize+7)/8,1);
}

} // namespace somegi
