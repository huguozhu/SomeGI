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
    auto sh=rhi::VkRHIShader::createFromFile(vkD,sd,shaderDir()/"gi"/"vxgi"/"vxgi_aniso_build.spv");
    rhi::ComputePSODesc pd; pd.debugName="VxgiAniso"; pd.computeShader=sh.get(); pd.descriptorSetLayouts={m_setLayout.get()}; pd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(AnisoPC)}};
    m_pipeline=d.createComputePSO(pd);
}
void VxgiAnisoPass::destroy(){m_sets.clear();m_pipeline.reset();m_setLayout.reset();m_rhiDevice=nullptr;}
void VxgiAnisoPass::bindResources(const VxgiResources& vxgi){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_sets[0]->write({{0,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.mipView(0)).get()},{1,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.mipView(0)).get()},{2,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.anisoMipView(1)).get()}});
    for(uint32_t lv=2;lv<m_mipLevels;++lv) m_sets[lv-1]->write({{0,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.mipView(0)).get()},{1,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.anisoMipView(lv-1)).get()},{2,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.anisoMipView(lv)).get()}});
}
void VxgiAnisoPass::record(rhi::RHICommandBuffer& cmd,const VxgiResources& vxgi){
    if(!m_pipeline)return;
    uint32_t res=vxgi.resolution();
    cmd.bindPipelineState(*m_pipeline);
    cmd.globalBarrier();
    cmd.bindDescriptorSet(0,*m_sets[0]);
    AnisoPC pc{res>>1,0,0};
    cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc));
    cmd.dispatch(((res>>1)+3)/4,((res>>1)+3)/4,((res>>1)+3)/4);
    cmd.globalBarrier();
    for(uint32_t lv=2;lv<m_mipLevels;++lv){
        uint32_t ds=res>>lv; if(!ds)ds=1;
        cmd.globalBarrier();
        cmd.bindDescriptorSet(0,*m_sets[lv-1]);
        AnisoPC pc2{ds,1,lv-1};
        cmd.pushConstants(rhi::ShaderStage::Compute,&pc2,sizeof(pc2));
        cmd.dispatch((ds+3)/4,(ds+3)/4,(ds+3)/4);
        cmd.globalBarrier();
    }
}
} // namespace somegi
