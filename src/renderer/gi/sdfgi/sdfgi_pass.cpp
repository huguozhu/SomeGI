// SdfgiPass RHI — 4 pipelines (seed/jfa/finalize/trace)，record VkCompat。
#include "renderer/gi/sdfgi/sdfgi_pass.h"
#include "renderer/gi/sdfgi/sdfgi_resources.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include "renderer/core/render_targets.h"
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
namespace { struct SeedPC{uint32_t res,_p0,_p1,_p2;}; struct JfaPC{uint32_t res,step,_p0,_p1;}; struct FinalizePC{uint32_t res;float maxDistCells;}; struct TracePC{uint32_t outSizeX,outSizeY,numRays,maxSteps;float invOutSizeX,invOutSizeY,rayMaxCells,hitEpsCells;uint32_t frameIndex,_p0,_p1,_p2;}; }
SdfgiPass::~SdfgiPass()=default;
void SdfgiPass::init(rhi::RHIDevice& d){ m_rhiDevice=&d; auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};si.magFilter=si.minFilter=VK_FILTER_LINEAR;si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;si.maxLod=16.f;
    vkCreateSampler(vkD.vkDevice(),&si,nullptr,&m_linearClamp); auto sp=shaderDir()/"gi"/"sdfgi";
    rhi::DescSetLayoutDesc sld; sld.debugName="SDFGI_Seed"; sld.bindings={{0,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_seedDsl=d.createDescriptorSetLayout(sld); m_seedSet=d.createDescriptorSet(*m_seedDsl);
    auto ms=rhi::VkRHIShader::createFromFile(vkD,{rhi::ShaderStage::Compute,rhi::ShaderFormat::SPIRV,"cs_main"},sp/"sdfgi_seed.spv");
    rhi::ComputePSODesc spd;spd.computeShader=ms.get();spd.descriptorSetLayouts={m_seedDsl.get()};spd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(SeedPC)}};m_seedPipe=d.createComputePSO(spd);
    rhi::DescSetLayoutDesc jld; jld.debugName="SDFGI_JFA"; jld.bindings={{0,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_jfaDsl=d.createDescriptorSetLayout(jld); m_jfaAB=d.createDescriptorSet(*m_jfaDsl); m_jfaBA=d.createDescriptorSet(*m_jfaDsl);
    auto mj=rhi::VkRHIShader::createFromFile(vkD,{rhi::ShaderStage::Compute,rhi::ShaderFormat::SPIRV,"cs_main"},sp/"sdfgi_jfa.spv");
    rhi::ComputePSODesc jpd;jpd.computeShader=mj.get();jpd.descriptorSetLayouts={m_jfaDsl.get()};jpd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(JfaPC)}};m_jfaPipe=d.createComputePSO(jpd);
    rhi::DescSetLayoutDesc fld; fld.debugName="SDFGI_Fin"; fld.bindings={{0,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_finDsl=d.createDescriptorSetLayout(fld); m_finA=d.createDescriptorSet(*m_finDsl); m_finB=d.createDescriptorSet(*m_finDsl);
    auto mf=rhi::VkRHIShader::createFromFile(vkD,{rhi::ShaderStage::Compute,rhi::ShaderFormat::SPIRV,"cs_main"},sp/"sdfgi_finalize.spv");
    rhi::ComputePSODesc fpd;fpd.computeShader=mf.get();fpd.descriptorSetLayouts={m_finDsl.get()};fpd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(FinalizePC)}};m_finPipe=d.createComputePSO(fpd);
    rhi::DescSetLayoutDesc tld; tld.debugName="SDFGI_Trace"; tld.bindings={{0,rhi::DescriptorType::UniformBuffer,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{5,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Compute},{6,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_traceDsl=d.createDescriptorSetLayout(tld); m_traceSet=d.createDescriptorSet(*m_traceDsl);
    auto mt=rhi::VkRHIShader::createFromFile(vkD,{rhi::ShaderStage::Compute,rhi::ShaderFormat::SPIRV,"cs_main"},sp/"sdfgi_trace.spv");
    rhi::ComputePSODesc tpd;tpd.computeShader=mt.get();tpd.descriptorSetLayouts={m_traceDsl.get()};tpd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(TracePC)}};m_tracePipe=d.createComputePSO(tpd);
}
void SdfgiPass::destroy(){ if(m_linearClamp)vkDestroySampler(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice).vkDevice(),m_linearClamp,nullptr);
    m_traceSet.reset();m_finB.reset();m_finA.reset();m_jfaBA.reset();m_jfaAB.reset();m_seedSet.reset(); m_tracePipe.reset();m_finPipe.reset();m_jfaPipe.reset();m_seedPipe.reset(); m_traceDsl.reset();m_finDsl.reset();m_jfaDsl.reset();m_seedDsl.reset(); m_rhiDevice=nullptr; }
void SdfgiPass::bindResources(const SdfgiResources& sf,const VxgiResources& vx,const RenderTargets& rt,VkBuffer fb){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto vox=rhi::VkRHITextureView::createNonOwning(vkD,vx.fullView()); auto aniso=rhi::VkRHITextureView::createNonOwning(vkD,vx.anisoFullView());
    auto nr=rhi::VkRHITextureView::createNonOwning(vkD,rt.gNormalRough.view()); auto dp=rhi::VkRHITextureView::createNonOwning(vkD,rt.depth.view());
    auto out=rhi::VkRHITextureView::createNonOwning(vkD,rt.ssgi.view()); auto ubo=rhi::VkRHIBuffer::createNonOwning(vkD,fb,VK_WHOLE_SIZE);
    auto sa=rhi::VkRHITextureView::createNonOwning(vkD,sf.seedA().view()); auto sb=rhi::VkRHITextureView::createNonOwning(vkD,sf.seedB().view());
    auto udf=rhi::VkRHITextureView::createNonOwning(vkD,sf.udf().view());
    m_seedSet->write({{0,rhi::DescriptorType::SampledImage,vox.get()},{1,rhi::DescriptorType::StorageImage,sa.get()}});
    m_jfaAB->write({{0,rhi::DescriptorType::SampledImage,sa.get()},{1,rhi::DescriptorType::StorageImage,sb.get()}});
    m_jfaBA->write({{0,rhi::DescriptorType::SampledImage,sb.get()},{1,rhi::DescriptorType::StorageImage,sa.get()}});
    m_finA->write({{0,rhi::DescriptorType::SampledImage,sa.get()},{1,rhi::DescriptorType::StorageImage,udf.get()}}); m_finB->write({{0,rhi::DescriptorType::SampledImage,sb.get()},{1,rhi::DescriptorType::StorageImage,udf.get()}});
    m_traceSet->write({{0,rhi::DescriptorType::UniformBuffer,nullptr,ubo.get()},{1,rhi::DescriptorType::SampledImage,nr.get()},{2,rhi::DescriptorType::SampledImage,dp.get()},{3,rhi::DescriptorType::SampledImage,vox.get()},{4,rhi::DescriptorType::SampledImage,aniso.get()},{5,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,(const void*)(uintptr_t)m_linearClamp},{6,rhi::DescriptorType::StorageImage,out.get()}});
}
void SdfgiPass::record(VkCommandBuffer vkCmd,const SdfgiResources& sf,const RenderTargets& rt,uint32_t fi,float st,float md,uint32_t nr,uint32_t ms,float rm,float he){
    auto h=[&](auto& p){return (VkPipeline)(uintptr_t)p->nativeHandle();}; auto l=[&](auto& p){return static_cast<rhi::VkRHIPipelineState&>(*p).layout();}; auto s=[&](auto& x){return (VkDescriptorSet)(uintptr_t)x->nativeHandle();};
    auto b=[&](VkPipelineStageFlags2 ss,VkAccessFlags2 sa,VkPipelineStageFlags2 ds,VkAccessFlags2 da){VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};mb.srcStageMask=ss;mb.srcAccessMask=sa;mb.dstStageMask=ds;mb.dstAccessMask=da;VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};di.memoryBarrierCount=1;di.pMemoryBarriers=&mb;vkCmdPipelineBarrier2(vkCmd,&di);};
    uint32_t res=sf.resolution(); VkDescriptorSet sds=s(m_seedSet),jab=s(m_jfaAB),jba=s(m_jfaBA),fas=s(m_finA),fbs=s(m_finB),tcs=s(m_traceSet);
    vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,h(m_seedPipe));vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,l(m_seedPipe),0,1,&sds,0,nullptr);
    SeedPC spc{res};vkCmdPushConstants(vkCmd,l(m_seedPipe),VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(spc),&spc);vkCmdDispatch(vkCmd,(res+3)/4,(res+3)/4,(res+3)/4);
    b(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_READ_BIT|VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,h(m_jfaPipe));
    for(int k=64;k>=1;k>>=1){VkDescriptorSet sj=(k&1)?jba:jab;vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,l(m_jfaPipe),0,1,&sj,0,nullptr);
        JfaPC jpc{res,(uint32_t)k};vkCmdPushConstants(vkCmd,l(m_jfaPipe),VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(jpc),&jpc);vkCmdDispatch(vkCmd,(res+3)/4,(res+3)/4,(res+3)/4);
        b(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_READ_BIT|VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);}
    VkDescriptorSet fs=(sf.resolution()%2==0)?fas:fbs;vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,h(m_finPipe));vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,l(m_finPipe),0,1,&fs,0,nullptr);
    FinalizePC fpc{res,md};vkCmdPushConstants(vkCmd,l(m_finPipe),VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(fpc),&fpc);vkCmdDispatch(vkCmd,(res+3)/4,(res+3)/4,(res+3)/4);
    b(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,h(m_tracePipe));vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,l(m_tracePipe),0,1,&tcs,0,nullptr);
    TracePC tpc{(uint32_t)rt.extent.width,(uint32_t)rt.extent.height,nr,ms,1.f/rt.extent.width,1.f/rt.extent.height,rm,he,fi};vkCmdPushConstants(vkCmd,l(m_tracePipe),VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(tpc),&tpc);
    vkCmdDispatch(vkCmd,(rt.extent.width+7)/8,(rt.extent.height+7)/8,1);
}
} // namespace somegi
