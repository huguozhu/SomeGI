// VxgiAnisoPass RHI — 3 bindings: voxelGrid(src)+aniso(prev)+aniso(dst).

#include "renderer/gi/vxgi/vxgi_aniso_pass.h"
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
namespace { struct AnisoPC { uint32_t dstSize,useAnisoSrc,srcLevel,_pad; }; static_assert(sizeof(AnisoPC)==16); }
VxgiAnisoPass::~VxgiAnisoPass()=default;
void VxgiAnisoPass::init(rhi::RHIDevice& d,uint32_t ml){ m_rhiDevice=&d; m_mipLevels=ml; if(ml<2)return;
    rhi::DescSetLayoutDesc ld; ld.debugName="VxgiAniso"; ld.bindings={{0,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_setLayout=d.createDescriptorSetLayout(ld); uint32_t n=ml-1; m_sets.resize(n); for(auto& s:m_sets)s=d.createDescriptorSet(*m_setLayout);
    auto& vkD=static_cast<rhi::VkRHIDevice&>(d); rhi::ShaderDesc sd; sd.stage=rhi::ShaderStage::Compute; sd.entryPoint="cs_main";
    auto sh=rhi::VkRHIShader::createFromFile(vkD,sd,shaderDir()/"gi"/"vxgi"/"vxgi_aniso.spv");
    rhi::ComputePSODesc pd; pd.debugName="VxgiAniso"; pd.computeShader=sh.get(); pd.descriptorSetLayouts={m_setLayout.get()}; pd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(AnisoPC)}};
    m_pipeline=d.createComputePSO(pd);
}
void VxgiAnisoPass::destroy(){m_sets.clear();m_pipeline.reset();m_setLayout.reset();m_rhiDevice=nullptr;}
void VxgiAnisoPass::bindResources(const VxgiResources& vxgi){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_sets[0]->write({{0,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.mipView(0)).get()},{1,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.mipView(0)).get()},{2,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.anisoMipView(1)).get()}});
    for(uint32_t lv=2;lv<m_mipLevels;++lv) m_sets[lv-1]->write({{0,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.mipView(0)).get()},{1,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.anisoMipView(lv-1)).get()},{2,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.anisoMipView(lv)).get()}});
}
void VxgiAnisoPass::record(rhi::RHICommandBuffer& cmd,const VxgiResources& vxgi){ if(!m_pipeline)return; VkCommandBuffer vkCmd=(VkCommandBuffer)(uintptr_t)cmd.nativeHandle();
    auto bm=[&](VkImage img,uint32_t mip,VkImageLayout ol,VkImageLayout nl,VkPipelineStageFlags2 ss,VkAccessFlags2 sa,VkPipelineStageFlags2 ds,VkAccessFlags2 da){ VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};b.srcStageMask=ss;b.srcAccessMask=sa;b.dstStageMask=ds;b.dstAccessMask=da;b.oldLayout=ol;b.newLayout=nl;b.image=img;b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,mip,1,0,1};VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};di.imageMemoryBarrierCount=1;di.pImageMemoryBarriers=&b;vkCmdPipelineBarrier2(vkCmd,&di);};
    VkImage anisoImg=vxgi.aniso().image(); uint32_t res=vxgi.resolution(); cmd.bindPipelineState(*m_pipeline);
    { bm(anisoImg,1,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_GENERAL,VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,0,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        cmd.bindDescriptorSet(0,*m_sets[0]); AnisoPC pc{res>>1,0,0}; cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc)); cmd.dispatch(((res>>1)+3)/4,((res>>1)+3)/4,((res>>1)+3)/4);
        bm(anisoImg,1,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT); }
    for(uint32_t lv=2;lv<m_mipLevels;++lv){ uint32_t ds=res>>lv; if(!ds)ds=1; bm(anisoImg,lv,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_GENERAL,VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,0,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        cmd.bindDescriptorSet(0,*m_sets[lv-1]); AnisoPC pc{ds,1,lv-1}; cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc)); cmd.dispatch((ds+3)/4,(ds+3)/4,(ds+3)/4);
        bm(anisoImg,lv,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT); }
}
void VxgiAnisoPass::record(VkCommandBuffer vkCmd,const VxgiResources& vxgi){ rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice),vkCmd); record(rhiCmd,vxgi); }
} // namespace somegi
