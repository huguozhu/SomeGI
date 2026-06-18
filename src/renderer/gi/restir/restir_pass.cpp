// RestirPass RHI — 3-4管线 (init/spatial/shade/shadeRt)，bind/record VkCompat。
#include "renderer/gi/restir/restir_pass.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_pso.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>
namespace somegi {
namespace { struct InitPC{uint32_t outX,outY;float invX,invY;uint32_t numLights,numCandidates,frameIndex,pad0;};static_assert(sizeof(InitPC)==32);
struct SpatialPC{uint32_t outX,outY;float invX,invY;uint32_t numLights,numNeighbors;float radiusPixels;uint32_t frameIndex;};static_assert(sizeof(SpatialPC)==32);
struct ShadePC{uint32_t outX,outY;float invX,invY;uint32_t numLights,pad0;float shadowSteps,intensityScale;};static_assert(sizeof(ShadePC)==32); }
RestirPass::~RestirPass()=default;
void RestirPass::init(rhi::RHIDevice& d,bool hwRt){ m_rhiDevice=&d; auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};si.magFilter=si.minFilter=VK_FILTER_LINEAR;si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR;si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;si.maxLod=0.f;
    vkCreateSampler(vkD.vkDevice(),&si,nullptr,&m_linearClamp); auto sp=shaderDir()/"gi"/"restir";
    auto mk=[&](const char* fn,const char* ep,rhi::RHIDescriptorSetLayout* lo,uint32_t pcs){
        auto sh=rhi::VkRHIShader::createFromFile(vkD,{rhi::ShaderStage::Compute,rhi::ShaderFormat::SPIRV,ep},sp/fn);
        rhi::ComputePSODesc pd; pd.debugName=fn; pd.computeShader=sh.get(); pd.descriptorSetLayouts={lo}; if(pcs)pd.pushConstants={{rhi::ShaderStage::Compute,0,pcs}}; return d.createComputePSO(pd); };
    // Init: 6 bindings (UBO+3images+lights SSBO+reservoirA storage)
    rhi::DescSetLayoutDesc ild; ild.debugName="RST_Init"; ild.bindings={{0,rhi::DescriptorType::UniformBuffer,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{5,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_initDsl=d.createDescriptorSetLayout(ild); m_initSet=d.createDescriptorSet(*m_initDsl); m_initPipe=mk("restir_init.spv","cs_main",m_initDsl.get(),sizeof(InitPC));
    // Spatial: 7 bindings (UBO+3images+lights+reservoirA sampled+reservoirB storage)
    rhi::DescSetLayoutDesc sld; sld.debugName="RST_Spatial"; sld.bindings={{0,rhi::DescriptorType::UniformBuffer,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{5,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{6,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_spatialDsl=d.createDescriptorSetLayout(sld); m_spatialSet=d.createDescriptorSet(*m_spatialDsl); m_spatialPipe=mk("restir_spatial.spv","cs_main",m_spatialDsl.get(),sizeof(SpatialPC));
    // Shade: 9 bindings (UBO+3images+lights+reservoirB+voxel+sampler+output)
    rhi::DescSetLayoutDesc hld; hld.debugName="RST_Shade"; hld.bindings={{0,rhi::DescriptorType::UniformBuffer,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{5,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{6,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{7,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute},{8,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Compute}};
    m_shadeDsl=d.createDescriptorSetLayout(hld); m_shadeSet=d.createDescriptorSet(*m_shadeDsl); m_shadePipe=mk("restir_shade.spv","cs_main",m_shadeDsl.get(),sizeof(ShadePC));
    if(hwRt){ m_rtShadeReady=true;
        rhi::DescSetLayoutDesc rld; rld.debugName="RST_ShadeRt"; rld.bindings={{0,rhi::DescriptorType::UniformBuffer,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{5,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{6,rhi::DescriptorType::AccelerationStructure,1,rhi::ShaderStage::Compute},{7,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Compute},{8,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
        m_shadeRtDsl=d.createDescriptorSetLayout(rld); m_shadeRtSet=d.createDescriptorSet(*m_shadeRtDsl); m_shadeRtPipe=mk("restir_shade_rt.spv","cs_main",m_shadeRtDsl.get(),sizeof(ShadePC));
    }
}
void RestirPass::destroy(){ if(m_linearClamp)vkDestroySampler(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice).vkDevice(),m_linearClamp,nullptr);
    m_shadeRtSet.reset();m_shadeRtPipe.reset();m_shadeRtDsl.reset(); m_shadeSet.reset();m_shadePipe.reset();m_shadeDsl.reset(); m_spatialSet.reset();m_spatialPipe.reset();m_spatialDsl.reset(); m_initSet.reset();m_initPipe.reset();m_initDsl.reset(); m_rhiDevice=nullptr; }
void RestirPass::bindResources(const RestirResources& res,const VxgiResources& vxgi,const RenderTargets& rt,VkBuffer ubo){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto ab=rhi::VkRHITextureView::createNonOwning(vkD,rt.gAlbedoMetal.view()); auto nr=rhi::VkRHITextureView::createNonOwning(vkD,rt.gNormalRough.view()); auto dp=rhi::VkRHITextureView::createNonOwning(vkD,rt.depth.view());
    auto ub=rhi::VkRHIBuffer::createNonOwning(vkD,ubo,VK_WHOLE_SIZE); auto lb=rhi::VkRHIBuffer::createNonOwning(vkD,res.lightBuffer(),VK_WHOLE_SIZE);
    auto ra=rhi::VkRHITextureView::createNonOwning(vkD,res.reservoirA().view()); auto rb=rhi::VkRHITextureView::createNonOwning(vkD,res.reservoirB().view());
    auto out=rhi::VkRHITextureView::createNonOwning(vkD,rt.restir.view()); auto vox=rhi::VkRHITextureView::createNonOwning(vkD,vxgi.fullView());
    const void* sp=(const void*)(uintptr_t)m_linearClamp;
    m_initSet->write({{0,rhi::DescriptorType::UniformBuffer,nullptr,ub.get()},{1,rhi::DescriptorType::SampledImage,ab.get()},{2,rhi::DescriptorType::SampledImage,nr.get()},{3,rhi::DescriptorType::SampledImage,dp.get()},{4,rhi::DescriptorType::StorageBuffer,nullptr,lb.get()},{5,rhi::DescriptorType::StorageImage,ra.get()}});
    m_spatialSet->write({{0,rhi::DescriptorType::UniformBuffer,nullptr,ub.get()},{1,rhi::DescriptorType::SampledImage,ab.get()},{2,rhi::DescriptorType::SampledImage,nr.get()},{3,rhi::DescriptorType::SampledImage,dp.get()},{4,rhi::DescriptorType::StorageBuffer,nullptr,lb.get()},{5,rhi::DescriptorType::SampledImage,ra.get()},{6,rhi::DescriptorType::StorageImage,rb.get()}});
    m_shadeSet->write({{0,rhi::DescriptorType::UniformBuffer,nullptr,ub.get()},{1,rhi::DescriptorType::SampledImage,ab.get()},{2,rhi::DescriptorType::SampledImage,nr.get()},{3,rhi::DescriptorType::SampledImage,dp.get()},{4,rhi::DescriptorType::StorageBuffer,nullptr,lb.get()},{5,rhi::DescriptorType::SampledImage,rb.get()},{6,rhi::DescriptorType::SampledImage,vox.get()},{7,rhi::DescriptorType::StorageImage,out.get()},{8,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,sp}});
}
void RestirPass::bindResourcesRt(const RestirResources& res,const RenderTargets& rt,VkBuffer ubo,VkAccelerationStructureKHR tlas){ if(!m_rtShadeReady)return; auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto ab=rhi::VkRHITextureView::createNonOwning(vkD,rt.gAlbedoMetal.view()); auto nr=rhi::VkRHITextureView::createNonOwning(vkD,rt.gNormalRough.view()); auto dp=rhi::VkRHITextureView::createNonOwning(vkD,rt.depth.view());
    auto ub=rhi::VkRHIBuffer::createNonOwning(vkD,ubo,VK_WHOLE_SIZE); auto lb=rhi::VkRHIBuffer::createNonOwning(vkD,res.lightBuffer(),VK_WHOLE_SIZE);
    auto rb=rhi::VkRHITextureView::createNonOwning(vkD,res.reservoirB().view()); auto out=rhi::VkRHITextureView::createNonOwning(vkD,rt.restir.view());
    const void* sp=(const void*)(uintptr_t)m_linearClamp;
    m_shadeRtSet->write({{0,rhi::DescriptorType::UniformBuffer,nullptr,ub.get()},{1,rhi::DescriptorType::SampledImage,ab.get()},{2,rhi::DescriptorType::SampledImage,nr.get()},{3,rhi::DescriptorType::SampledImage,dp.get()},{4,rhi::DescriptorType::StorageBuffer,nullptr,lb.get()},{5,rhi::DescriptorType::SampledImage,rb.get()},{6,rhi::DescriptorType::AccelerationStructure,nullptr,nullptr,0,0,nullptr,&tlas},{7,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,sp},{8,rhi::DescriptorType::StorageImage,out.get()}});
}
void RestirPass::record(VkCommandBuffer vkCmd,const RestirResources&,const RenderTargets& rt,uint32_t nl,uint32_t nc,uint32_t nn,float sr,uint32_t ss,float is,uint32_t fi,bool useRt){
    auto h=[&](auto& p){return (VkPipeline)(uintptr_t)p->nativeHandle();}; auto l=[&](auto& p){return static_cast<rhi::VkRHIPipelineState&>(*p).layout();}; auto s=[&](auto& x){return (VkDescriptorSet)(uintptr_t)x->nativeHandle();};
    auto b=[&](VkPipelineStageFlags2 s1,VkAccessFlags2 a1,VkPipelineStageFlags2 s2,VkAccessFlags2 a2){VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};mb.srcStageMask=s1;mb.srcAccessMask=a1;mb.dstStageMask=s2;mb.dstAccessMask=a2;VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};di.memoryBarrierCount=1;di.pMemoryBarriers=&mb;vkCmdPipelineBarrier2(vkCmd,&di);};
    uint32_t gx=(rt.extent.width+7)/8,gy=(rt.extent.height+7)/8; float ix=1.f/rt.extent.width,iy=1.f/rt.extent.height;
    VkDescriptorSet iset=s(m_initSet),sset=s(m_spatialSet),hset=s(m_shadeSet),rset=m_rtShadeReady?s(m_shadeRtSet):VK_NULL_HANDLE;
    // Init
    vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,h(m_initPipe));vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,l(m_initPipe),0,1,&iset,0,nullptr);
    InitPC ipc{gx*8u,gy*8u,ix,iy,nl,(uint32_t)nc,fi};vkCmdPushConstants(vkCmd,l(m_initPipe),VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(ipc),&ipc);vkCmdDispatch(vkCmd,gx,gy,1);
    b(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    // Spatial
    vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,h(m_spatialPipe));vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,l(m_spatialPipe),0,1,&sset,0,nullptr);
    SpatialPC spc{gx*8u,gy*8u,ix,iy,nl,(uint32_t)nn,sr,fi};vkCmdPushConstants(vkCmd,l(m_spatialPipe),VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(spc),&spc);vkCmdDispatch(vkCmd,gx,gy,1);
    b(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_READ_BIT|VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    // Shade
    if(useRt&&m_rtShadeReady){ vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,h(m_shadeRtPipe));vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,l(m_shadeRtPipe),0,1,&rset,0,nullptr); }
    else{ vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,h(m_shadePipe));vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,l(m_shadePipe),0,1,&hset,0,nullptr); }
    auto& spso=useRt&&m_rtShadeReady?m_shadeRtPipe:m_shadePipe; ShadePC hpc{gx*8u,gy*8u,ix,iy,nl,0,(float)ss,is};
    vkCmdPushConstants(vkCmd,l(spso),VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(hpc),&hpc);vkCmdDispatch(vkCmd,gx,gy,1);
}
} // namespace somegi
