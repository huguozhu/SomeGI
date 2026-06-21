// LpvInjectPass RHI — 7 bindings: 3 RSM + 3 LPV + GV storage。
// barrier/clear 通过 nativeHandle 桥接（操作 VkImage，非 RHI 管理）。

#include "renderer/gi/lpv/lpv_inject_pass.h"
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
    VkCommandBuffer vkCmd=(VkCommandBuffer)(uintptr_t)cmd.nativeHandle();
    auto barr=[&](VkImage img,VkImageLayout oldL,VkImageLayout newL,VkPipelineStageFlags2 ss,VkAccessFlags2 sa,VkPipelineStageFlags2 ds,VkAccessFlags2 da){
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2}; b.srcStageMask=ss; b.srcAccessMask=sa; b.dstStageMask=ds; b.dstAccessMask=da; b.oldLayout=oldL; b.newLayout=newL; b.image=img; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO}; di.imageMemoryBarrierCount=1; di.pImageMemoryBarriers=&b; vkCmdPipelineBarrier2(vkCmd,&di);
    };
    VkImage imgs[4]={m_lpvR->image(),m_lpvG->image(),m_lpvB->image(),m_gv->image()};
    for(auto img:imgs){
        barr(img,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,0,VK_PIPELINE_STAGE_2_CLEAR_BIT,VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkClearColorValue zero{}; VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        vkCmdClearColorImage(vkCmd,img,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&zero,1,&r);
        barr(img,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_GENERAL,VK_PIPELINE_STAGE_2_CLEAR_BIT,VK_ACCESS_2_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0,*m_set);
    InjectPC pc{(uint32_t)m_rsmSize,(uint32_t)m_rsmSize,gr,0,gMin.x,gMin.y,gMin.z,cs};
    cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc));
    cmd.dispatch((m_rsmSize+7)/8,(m_rsmSize+7)/8,1);
}

void LpvInjectPass::record(VkCommandBuffer vkCmd, uint32_t gr, const glm::vec3& gMin, float cs) {
    rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice),vkCmd); record(rhiCmd,gr,gMin,cs);
}
} // namespace somegi
