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
#include "rhi/vulkan/vk_acceleration_structure.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_pso.h"
#include "rhi/vulkan/vk_sampler.h"
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
    auto tlasRHI=rtAS.tlas();
    auto inst=rhi::VkRHIBuffer::createNonOwning(vkD,rtAS.instanceDataBuffer(),VK_WHOLE_SIZE);
    auto vert=rhi::VkRHIBuffer::createNonOwning(vkD,scene.vertexBuffer.handle(),VK_WHOLE_SIZE);
    auto ind=rhi::VkRHIBuffer::createNonOwning(vkD,scene.indexBuffer.handle(),VK_WHOLE_SIZE);
    auto mat=rhi::VkRHIBuffer::createNonOwning(vkD,scene.materialBuffer.handle(),VK_WHOLE_SIZE);
    auto ubo=rhi::VkRHIBuffer::createNonOwning(vkD,frameUbo,VK_WHOLE_SIZE);
    auto sampB=rhi::VkRHIBuffer::createNonOwning(vkD,res.sampleBuf().handle(),VK_WHOLE_SIZE);
    auto cntB=rhi::VkRHIBuffer::createNonOwning(vkD,res.sampleCount().handle(),VK_WHOLE_SIZE);
    std::vector<std::unique_ptr<rhi::RHITextureView>> tvs; std::vector<const rhi::RHITextureView*> tvp;
    for(uint32_t i=0;i<128;++i){ VkImageView v=(i<(uint32_t)scene.images.size())?scene.images[i].view():scene.whiteTex.view(); tvs.push_back(rhi::VkRHITextureView::createNonOwning(vkD,v)); tvp.push_back(tvs.back().get()); }
    auto ndgiSampler = rhi::VkRHISampler::createNonOwning(vkD, scene.linearSampler);
    m_traceSet->write({{0,rhi::DescriptorType::AccelerationStructure,nullptr,nullptr,0,0,nullptr,tlasRHI},{1,rhi::DescriptorType::StorageBuffer,nullptr,inst.get()},{2,rhi::DescriptorType::StorageBuffer,nullptr,vert.get()},{3,rhi::DescriptorType::StorageBuffer,nullptr,ind.get()},{4,rhi::DescriptorType::StorageBuffer,nullptr,mat.get()},{5,rhi::DescriptorType::SampledImage,nullptr,nullptr,0,0,nullptr,nullptr,128,tvp.data()},{6,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,ndgiSampler.get()},{7,rhi::DescriptorType::UniformBuffer,nullptr,ubo.get()},{8,rhi::DescriptorType::StorageBuffer,nullptr,sampB.get()},{9,rhi::DescriptorType::StorageBuffer,nullptr,cntB.get()}});
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
// RHI 路径：初始化 NDGI 神经网络权重
void NdgiPass::initWeights(rhi::RHICommandBuffer& cmd){ if(!m_rtSupported||!m_initPipeline)return;
    cmd.bindPipelineState(*m_initPipeline);
    cmd.bindDescriptorSet(0, *m_initSet);
    struct{uint32_t seed;float scale;uint32_t p0,p1;}pc{42,1.f,0,0};
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, 16);
    cmd.dispatch(1, 1, 1);
}

// RHI 路径：NDGI 光线追踪 probe
void NdgiPass::record(rhi::RHICommandBuffer& cmd,NdgiResources& res,uint32_t fi,glm::vec3 o,glm::vec3 s){ if(!m_rtSupported||!m_tracePipeline)return;
    // 清零采样计数缓冲
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto cntBuf = rhi::VkRHIBuffer::createNonOwning(vkDev, res.sampleCount().handle(), VK_WHOLE_SIZE);
    cmd.fillBuffer(*cntBuf, 0, sizeof(uint32_t), 0);

    cmd.bindPipelineState(*m_tracePipeline);
    cmd.bindDescriptorSet(0, *m_traceSet);
    struct{float origin[3],pad0,spacing[3],pad1; uint32_t px,py,pz,rpp; float rotation,_pad2; uint32_t _pad3;}pc;
    pc.origin[0]=o.x;pc.origin[1]=o.y;pc.origin[2]=o.z;pc.spacing[0]=s.x;pc.spacing[1]=s.y;pc.spacing[2]=s.z;
    pc.px=NdgiResources::kProbesX;pc.py=NdgiResources::kProbesY;pc.pz=NdgiResources::kProbesZ;pc.rpp=NdgiResources::kRaysPerProbe;pc.rotation=float((fi%360)*0.0174532925);
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
    cmd.dispatch((pc.px*pc.py*pc.pz*pc.rpp+63)/64, 1, 1);
}

// RHI 路径：NDGI 神经网络训练
void NdgiPass::recordTraining(rhi::RHICommandBuffer& cmd,NdgiResources& res,uint32_t){ if(!m_rtSupported||!m_trainPipeline||!m_initSet)return;
    auto* cnt=static_cast<uint32_t*>(res.sampleCount().mapped()); uint32_t total=cnt?*cnt:0; if(!total)return;
    cmd.bindPipelineState(*m_trainPipeline);
    cmd.bindDescriptorSet(0, *m_initSet);
    struct{float lr,ema;uint32_t batch,iters,samples,p0,p1,p2;}pc{0.01f,0.95f,256,4,total};
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, 32);
    cmd.dispatch(1, 1, 1);
}

} // namespace somegi
