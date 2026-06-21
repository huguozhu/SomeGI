// VxgiVoxelizePass RHI — 6 bindings: 3 SSBO + sampler + texture[] + voxel storage.

#include "renderer/gi/vxgi/vxgi_voxelize_pass.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/vulkan/vk_sampler.h"
#include "core/shader.h"
#include <array>
#include <cstring>
namespace somegi {
namespace { struct VoxelizePC { float model[16]; uint32_t firstIndex,indexCount; int32_t vertexOffset,materialIndex; float gridMinX,gridMinY,gridMinZ,cellSize; uint32_t gridResolution,_p0,_p1,_p2; };
static_assert(sizeof(VoxelizePC)==112); }
VxgiVoxelizePass::~VxgiVoxelizePass()=default;
void VxgiVoxelizePass::init(rhi::RHIDevice& d,uint32_t mt){ m_rhiDevice=&d; m_maxTextures=mt;
    rhi::DescSetLayoutDesc ld; ld.debugName="VxgiVoxelize";
    ld.bindings={{0,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::SampledImage,mt,rhi::ShaderStage::Compute,true},{5,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_setLayout=d.createDescriptorSetLayout(ld); m_set=d.createDescriptorSet(*m_setLayout);
    auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
    rhi::ShaderDesc sd; sd.stage=rhi::ShaderStage::Compute; sd.entryPoint="cs_main";
    auto sh=rhi::VkRHIShader::createFromFile(vkD,sd,shaderDir()/"gi"/"vxgi"/"vxgi_voxelize.spv");
    rhi::ComputePSODesc pd; pd.debugName="VxgiVoxelize"; pd.computeShader=sh.get(); pd.descriptorSetLayouts={m_setLayout.get()}; pd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(VoxelizePC)}};
    m_pipeline=d.createComputePSO(pd);
}
void VxgiVoxelizePass::destroy(){ m_texViews.clear(); m_texViewPtrs.clear(); m_set.reset(); m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice=nullptr; }
void VxgiVoxelizePass::bindScene(const SceneGpu& gpu,uint32_t tc,const VxgiResources& vxgi){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto vb=rhi::VkRHIBuffer::createNonOwning(vkD,gpu.vertexBuffer.handle(),VK_WHOLE_SIZE);
    auto ib=rhi::VkRHIBuffer::createNonOwning(vkD,gpu.indexBuffer.handle(),VK_WHOLE_SIZE);
    auto mb=rhi::VkRHIBuffer::createNonOwning(vkD,gpu.materialBuffer.handle(),VK_WHOLE_SIZE);
    auto vox=vxgi.mipView(0);
    m_texViews.clear(); m_texViewPtrs.clear(); m_texViews.reserve(m_maxTextures); m_texViewPtrs.reserve(m_maxTextures);
    for(uint32_t i=0;i<m_maxTextures;++i){ VkImageView v=(i<tc&&i<gpu.images.size())?gpu.images[i].view():gpu.whiteTex.view(); m_texViews.push_back(rhi::VkRHITextureView::createNonOwning(vkD,v)); m_texViewPtrs.push_back(m_texViews.back().get()); }
    auto vxSampler = rhi::VkRHISampler::createNonOwning(vkD, gpu.linearSampler);
    rhi::DescriptorWrite w[6]={};
    w[0]={0,rhi::DescriptorType::StorageBuffer,nullptr,vb.get()}; w[1]={1,rhi::DescriptorType::StorageBuffer,nullptr,ib.get()}; w[2]={2,rhi::DescriptorType::StorageBuffer,nullptr,mb.get()}; w[3]={3,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,vxSampler.get()}; w[4]={4,rhi::DescriptorType::SampledImage,nullptr,nullptr,0,0,nullptr,nullptr,m_maxTextures,m_texViewPtrs.data()}; w[5]={5,rhi::DescriptorType::StorageImage,vox};
    m_set->write({w,w+6});
}
void VxgiVoxelizePass::record(rhi::RHICommandBuffer& cmd,const SceneCpu& cpu,const SceneGpu&,const glm::vec3& gm,float cs,uint32_t gr){ if(!m_pipeline||!m_set)return;
    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0,*m_set);
    VoxelizePC pc{}; pc.gridMinX=gm.x; pc.gridMinY=gm.y; pc.gridMinZ=gm.z; pc.cellSize=cs; pc.gridResolution=gr;
    for(auto& n:cpu.nodes){if(n.meshIndex<0)continue; const Mesh& M=cpu.meshes[n.meshIndex]; std::memcpy(pc.model,&n.worldTransform[0][0],sizeof(pc.model));
        for(auto& p:M.primitives){ pc.firstIndex=p.firstIndex; pc.indexCount=p.indexCount; pc.vertexOffset=p.vertexOffset; pc.materialIndex=p.materialIndex;
            cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc)); uint32_t gx=(p.indexCount/3u+63u)/64u; if(gx)cmd.dispatch(gx,1,1); }}}

} // namespace somegi
