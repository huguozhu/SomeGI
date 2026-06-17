// RtGiPass RHI — 11 bindings: UBO+2images+TLAS+4SSBO+sampler+texture[]+storage。

#include "renderer/gi/rt/rt_gi_pass.h"
#include "renderer/gi/rt/scene_rt_as.h"
#include "scene/scene_gpu.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_command.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>
namespace somegi {
static constexpr uint32_t kMaxTextures = 128;
namespace { struct RtPC { uint32_t outSizeX,outSizeY; float invOutSizeX,invOutSizeY; };
static_assert(sizeof(RtPC)==16); }
RtGiPass::~RtGiPass()=default;
void RtGiPass::init(rhi::RHIDevice& d){ m_rhiDevice=&d; auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter=si.minFilter=VK_FILTER_LINEAR; si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR; si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod=0.f;
    vkCreateSampler(vkD.vkDevice(),&si,nullptr,&m_linearClamp);
    rhi::DescSetLayoutDesc ld; ld.debugName="RtGI";
    ld.bindings={{0,rhi::DescriptorType::UniformBuffer,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::AccelerationStructure,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{5,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{6,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{7,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{8,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Compute},{9,rhi::DescriptorType::SampledImage,kMaxTextures,rhi::ShaderStage::Compute},{10,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_setLayout=d.createDescriptorSetLayout(ld); m_set=d.createDescriptorSet(*m_setLayout);
    rhi::ShaderDesc sd; sd.stage=rhi::ShaderStage::Compute; sd.entryPoint="cs_main";
    auto sh=rhi::VkRHIShader::createFromFile(vkD,sd,shaderDir()/"gi"/"rt"/"rt_gi.spv");
    rhi::ComputePSODesc pd; pd.debugName="RtGI"; pd.computeShader=sh.get(); pd.descriptorSetLayouts={m_setLayout.get()}; pd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(RtPC)}};
    m_pipeline=d.createComputePSO(pd);
    m_texViews.reserve(kMaxTextures); m_texViewPtrs.reserve(kMaxTextures);
}
void RtGiPass::destroy(){ if(m_linearClamp)vkDestroySampler(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice).vkDevice(),m_linearClamp,nullptr); m_texViews.clear(); m_texViewPtrs.clear(); m_set.reset(); m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice=nullptr; }
void RtGiPass::bindFrame(const RenderTargets& rt,VkBuffer frameUbo,const SceneRtAS& rtAS,const SceneGpu& gpu){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto ubo=rhi::VkRHIBuffer::createNonOwning(vkD,frameUbo,VK_WHOLE_SIZE);
    auto inst=rhi::VkRHIBuffer::createNonOwning(vkD,rtAS.instanceDataBuffer(),VK_WHOLE_SIZE);
    auto vert=rhi::VkRHIBuffer::createNonOwning(vkD,gpu.vertexBuffer.handle(),VK_WHOLE_SIZE);
    auto idx=rhi::VkRHIBuffer::createNonOwning(vkD,gpu.indexBuffer.handle(),VK_WHOLE_SIZE);
    auto mat=rhi::VkRHIBuffer::createNonOwning(vkD,gpu.materialBuffer.handle(),VK_WHOLE_SIZE);
    auto nr=rhi::VkRHITextureView::createNonOwning(vkD,rt.gNormalRough.view());
    auto dp=rhi::VkRHITextureView::createNonOwning(vkD,rt.depth.view());
    auto ao=rhi::VkRHITextureView::createNonOwning(vkD,rt.rtGI.view());
    m_texViews.clear(); m_texViewPtrs.clear();
    for(uint32_t i=0;i<kMaxTextures;++i){ VkImageView v=(i<gpu.images.size())?gpu.images[i].view():gpu.whiteTex.view(); m_texViews.push_back(rhi::VkRHITextureView::createNonOwning(vkD,v)); m_texViewPtrs.push_back(m_texViews.back().get()); }
    auto tasInfo=rtAS.tlasWriteInfo();
    rhi::DescriptorWrite w[11]={};
    w[0]={0,rhi::DescriptorType::UniformBuffer,nullptr,ubo.get()}; w[1]={1,rhi::DescriptorType::SampledImage,nr.get()}; w[2]={2,rhi::DescriptorType::SampledImage,dp.get()}; w[3]={3,rhi::DescriptorType::AccelerationStructure,nullptr,nullptr,0,0,nullptr,tasInfo.pAccelerationStructures}; w[4]={4,rhi::DescriptorType::StorageBuffer,nullptr,inst.get()}; w[5]={5,rhi::DescriptorType::StorageBuffer,nullptr,vert.get()}; w[6]={6,rhi::DescriptorType::StorageBuffer,nullptr,idx.get()}; w[7]={7,rhi::DescriptorType::StorageBuffer,nullptr,mat.get()}; w[8]={8,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,(const void*)(uintptr_t)m_linearClamp}; w[9]={9,rhi::DescriptorType::SampledImage,nullptr,nullptr,0,0,nullptr,nullptr,kMaxTextures,m_texViewPtrs.data()}; w[10]={10,rhi::DescriptorType::StorageImage,ao.get()};
    m_set->write({w,w+11});
}
void RtGiPass::record(rhi::RHICommandBuffer& cmd,const RenderTargets& rt){ if(!m_pipeline)return;
    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0,*m_set);
    RtPC pc{(uint32_t)rt.extent.width,(uint32_t)rt.extent.height,1.f/rt.extent.width,1.f/rt.extent.height};
    cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc)); cmd.dispatch((rt.extent.width+7)/8,(rt.extent.height+7)/8,1);
}
void RtGiPass::record(VkCommandBuffer vkCmd,const RenderTargets& rt){ rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice),vkCmd); record(rhiCmd,rt); }
} // namespace somegi
