// NdgiPass — RHI 管理 layouts/sets/pipelines，record 保留 VkCompat。
#include "renderer/gi/ndgi/ndgi_pass.h"
#include "renderer/gi/ndgi/ndgi_resources.h"
#include "core/device.h"
#include "renderer/gi/rt/scene_rt_as.h"
#include "renderer/core/render_targets.h"
#include "scene/scene.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_pso.h"
#include "core/shader.h"
#include <array>
namespace somegi {
NdgiPass::~NdgiPass()=default;
void NdgiPass::init(rhi::RHIDevice& d,bool rtSupported){ m_rhiDevice=&d; m_rtSupported=rtSupported; if(!rtSupported)return;
    auto& vkD=static_cast<rhi::VkRHIDevice&>(d); auto sd=shaderDir()/"gi"/"ndgi";
    // Trace: 10 bindings (TLAS+6SSBO+sampler+UBO+128textures)
    rhi::DescSetLayoutDesc tld; tld.debugName="NDGI_Trace"; tld.bindings={{0,rhi::DescriptorType::AccelerationStructure,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{5,rhi::DescriptorType::SampledImage,128,rhi::ShaderStage::Compute},{6,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Compute},{7,rhi::DescriptorType::UniformBuffer,1,rhi::ShaderStage::Compute},{8,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{9,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute}};
    m_traceDsl=d.createDescriptorSetLayout(tld); m_traceSet=d.createDescriptorSet(*m_traceDsl);
    rhi::ShaderDesc tsd; tsd.stage=rhi::ShaderStage::Compute; tsd.entryPoint="cs_main";
    auto tsh=rhi::VkRHIShader::createFromFile(vkD,tsd,sd/"ndgi_probe_trace.spv");
    rhi::ComputePSODesc tpd; tpd.debugName="NDGI_Trace"; tpd.computeShader=tsh.get(); tpd.descriptorSetLayouts={m_traceDsl.get()}; tpd.pushConstants={{rhi::ShaderStage::Compute,0,64}};
    m_tracePipeline=d.createComputePSO(tpd);
    // Init/Train: shared 8 SSBO bindings
    rhi::DescSetLayoutDesc ild; ild.debugName="NDGI_Init"; for(uint32_t i=0;i<8;++i)ild.bindings.push_back({i,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute});
    m_initDsl=d.createDescriptorSetLayout(ild); m_initSet=d.createDescriptorSet(*m_initDsl);
    rhi::ShaderDesc isd; isd.stage=rhi::ShaderStage::Compute; isd.entryPoint="cs_main";
    auto ish=rhi::VkRHIShader::createFromFile(vkD,isd,sd/"ndgi_init.spv");
    rhi::ComputePSODesc ipd; ipd.debugName="NDGI_Init"; ipd.computeShader=ish.get(); ipd.descriptorSetLayouts={m_initDsl.get()}; ipd.pushConstants={{rhi::ShaderStage::Compute,0,16}};
    m_initPipeline=d.createComputePSO(ipd);
    auto tsh2=rhi::VkRHIShader::createFromFile(vkD,isd,sd/"ndgi_train.spv");
    rhi::ComputePSODesc trpd; trpd.debugName="NDGI_Train"; trpd.computeShader=tsh2.get(); trpd.descriptorSetLayouts={m_initDsl.get()}; trpd.pushConstants={{rhi::ShaderStage::Compute,0,64}};
    m_trainPipeline=d.createComputePSO(trpd);
}
void NdgiPass::destroy(){ m_initSet.reset(); m_traceSet.reset(); m_initPipeline.reset(); m_trainPipeline.reset(); m_tracePipeline.reset(); m_initDsl.reset(); m_traceDsl.reset(); m_rhiDevice=nullptr; }
void NdgiPass::bindResources(const NdgiResources& res,SceneRtAS& rtAS,const SceneGpu& scene,const RenderTargets&,VkBuffer frameUbo){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice); if(!m_rtSupported||!m_traceSet)return;
    auto tasInfo=rtAS.tlasWriteInfo();
    auto inst=rhi::VkRHIBuffer::createNonOwning(vkD,rtAS.instanceDataBuffer(),VK_WHOLE_SIZE);
    auto vert=rhi::VkRHIBuffer::createNonOwning(vkD,scene.vertexBuffer.handle(),VK_WHOLE_SIZE);
    auto ind=rhi::VkRHIBuffer::createNonOwning(vkD,scene.indexBuffer.handle(),VK_WHOLE_SIZE);
    auto mat=rhi::VkRHIBuffer::createNonOwning(vkD,scene.materialBuffer.handle(),VK_WHOLE_SIZE);
    auto ubo=rhi::VkRHIBuffer::createNonOwning(vkD,frameUbo,VK_WHOLE_SIZE);
    auto sampB=rhi::VkRHIBuffer::createNonOwning(vkD,res.sampleBuf().handle(),VK_WHOLE_SIZE);
    auto cntB=rhi::VkRHIBuffer::createNonOwning(vkD,res.sampleCount().handle(),VK_WHOLE_SIZE);
    std::vector<std::unique_ptr<rhi::RHITextureView>> tvs; std::vector<const rhi::RHITextureView*> tvp;
    for(uint32_t i=0;i<128;++i){ VkImageView v=(i<(uint32_t)scene.images.size())?scene.images[i].view():scene.whiteTex.view(); tvs.push_back(rhi::VkRHITextureView::createNonOwning(vkD,v)); tvp.push_back(tvs.back().get()); }
    m_traceSet->write({{0,rhi::DescriptorType::AccelerationStructure,nullptr,nullptr,0,0,nullptr,tasInfo.pAccelerationStructures},{1,rhi::DescriptorType::StorageBuffer,nullptr,inst.get()},{2,rhi::DescriptorType::StorageBuffer,nullptr,vert.get()},{3,rhi::DescriptorType::StorageBuffer,nullptr,ind.get()},{4,rhi::DescriptorType::StorageBuffer,nullptr,mat.get()},{5,rhi::DescriptorType::SampledImage,nullptr,nullptr,0,0,nullptr,nullptr,128,tvp.data()},{6,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,(const void*)(uintptr_t)scene.linearSampler},{7,rhi::DescriptorType::UniformBuffer,nullptr,ubo.get()},{8,rhi::DescriptorType::StorageBuffer,nullptr,sampB.get()},{9,rhi::DescriptorType::StorageBuffer,nullptr,cntB.get()}});
    writeInitDescriptors(res);
}
void NdgiPass::writeInitDescriptors(const NdgiResources& res){ if(!m_initSet)return; auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto w1=rhi::VkRHIBuffer::createNonOwning(vkD,res.weights1().handle(),VK_WHOLE_SIZE); auto b1=rhi::VkRHIBuffer::createNonOwning(vkD,res.bias1().handle(),VK_WHOLE_SIZE);
    auto w2=rhi::VkRHIBuffer::createNonOwning(vkD,res.weights2().handle(),VK_WHOLE_SIZE); auto b2=rhi::VkRHIBuffer::createNonOwning(vkD,res.bias2().handle(),VK_WHOLE_SIZE);
    auto w3=rhi::VkRHIBuffer::createNonOwning(vkD,res.weights3().handle(),VK_WHOLE_SIZE); auto b3=rhi::VkRHIBuffer::createNonOwning(vkD,res.bias3().handle(),VK_WHOLE_SIZE);
    auto sb=rhi::VkRHIBuffer::createNonOwning(vkD,res.sampleBuf().handle(),VK_WHOLE_SIZE); auto sc=rhi::VkRHIBuffer::createNonOwning(vkD,res.sampleCount().handle(),VK_WHOLE_SIZE);
    m_initSet->write({{0,rhi::DescriptorType::StorageBuffer,nullptr,w1.get()},{1,rhi::DescriptorType::StorageBuffer,nullptr,b1.get()},{2,rhi::DescriptorType::StorageBuffer,nullptr,w2.get()},{3,rhi::DescriptorType::StorageBuffer,nullptr,b2.get()},{4,rhi::DescriptorType::StorageBuffer,nullptr,w3.get()},{5,rhi::DescriptorType::StorageBuffer,nullptr,b3.get()},{6,rhi::DescriptorType::StorageBuffer,nullptr,sb.get()},{7,rhi::DescriptorType::StorageBuffer,nullptr,sc.get()}});
}
// VkCompat record — 使用 RHI PSO 的 nativeHandle + layout
void NdgiPass::initWeights(VkCommandBuffer vkCmd){ if(!m_rtSupported||!m_initPipeline)return;
    auto& p=static_cast<rhi::VkRHIPipelineState&>(*m_initPipeline); VkDescriptorSet ds=(VkDescriptorSet)(uintptr_t)m_initSet->nativeHandle();
    vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,(VkPipeline)(uintptr_t)p.nativeHandle()); vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,p.layout(),0,1,&ds,0,nullptr);
    struct{uint32_t seed;float scale;uint32_t p0,p1;}pc{42,1.f,0,0}; vkCmdPushConstants(vkCmd,p.layout(),VK_SHADER_STAGE_COMPUTE_BIT,0,16,&pc); vkCmdDispatch(vkCmd,1,1,1);
}
void NdgiPass::record(VkCommandBuffer vkCmd,NdgiResources& res,uint32_t fi,glm::vec3 o,glm::vec3 s){ if(!m_rtSupported||!m_tracePipeline)return;
    auto& p=static_cast<rhi::VkRHIPipelineState&>(*m_tracePipeline); VkDescriptorSet ds=(VkDescriptorSet)(uintptr_t)m_traceSet->nativeHandle();
    vkCmdFillBuffer(vkCmd,res.sampleCount().handle(),0,4,0);
    vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,(VkPipeline)(uintptr_t)p.nativeHandle()); vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,p.layout(),0,1,&ds,0,nullptr);
    struct{float origin[3],pad0,spacing[3],pad1; uint32_t px,py,pz,rpp; float rotation,_pad2; uint32_t _pad3;}pc;
    pc.origin[0]=o.x;pc.origin[1]=o.y;pc.origin[2]=o.z;pc.spacing[0]=s.x;pc.spacing[1]=s.y;pc.spacing[2]=s.z;
    pc.px=NdgiResources::kProbesX;pc.py=NdgiResources::kProbesY;pc.pz=NdgiResources::kProbesZ;pc.rpp=NdgiResources::kRaysPerProbe;pc.rotation=float((fi%360)*0.0174532925);
    vkCmdPushConstants(vkCmd,p.layout(),VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc),&pc); vkCmdDispatch(vkCmd,(pc.px*pc.py*pc.pz*pc.rpp+63)/64,1,1);
}
void NdgiPass::recordTraining(VkCommandBuffer vkCmd,NdgiResources& res,uint32_t){ if(!m_rtSupported||!m_trainPipeline||!m_initSet)return;
    auto* cnt=static_cast<uint32_t*>(res.sampleCount().mapped()); uint32_t total=cnt?*cnt:0; if(!total)return;
    auto& p=static_cast<rhi::VkRHIPipelineState&>(*m_trainPipeline); VkDescriptorSet ds=(VkDescriptorSet)(uintptr_t)m_initSet->nativeHandle();
    vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,(VkPipeline)(uintptr_t)p.nativeHandle()); vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,p.layout(),0,1,&ds,0,nullptr);
    struct{float lr,ema;uint32_t batch,iters,samples,p0,p1,p2;}pc{0.01f,0.95f,256,4,total};
    vkCmdPushConstants(vkCmd,p.layout(),VK_SHADER_STAGE_COMPUTE_BIT,0,32,&pc); vkCmdDispatch(vkCmd,1,1,1);
}
} // namespace somegi
