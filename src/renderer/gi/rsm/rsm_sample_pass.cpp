// RsmSamplePass RHI — 9 bindings: 2 UBO + 5 sampled + 1 sampler + 1 storage.

#include "renderer/gi/rsm/rsm_sample_pass.h"
#include "core/device.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/base/sampler.h"
#include "core/shader.h"
#include <array>

namespace somegi {
namespace { struct RsmSamplePC { uint32_t outSizeX,outSizeY; float invOutSizeX,invOutSizeY,radius; uint32_t sampleCount; float intensity,_pad; };
static_assert(sizeof(RsmSamplePC)==32); }

RsmSamplePass::~RsmSamplePass() = default;

void RsmSamplePass::init(rhi::RHIDevice& d) {
    m_rhiDevice = &d; auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter=si.minFilter=VK_FILTER_LINEAR; si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod=0.f;
    m_linearClamp = d.createSampler({rhi::Filter::Linear,rhi::Filter::Linear,rhi::SamplerMipmapMode::Linear,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,0.f});

    rhi::DescSetLayoutDesc ld; ld.debugName="RsmSample";
    ld.bindings={{0,rhi::DescriptorType::UniformBuffer,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{5,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{6,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Compute},{7,rhi::DescriptorType::UniformBuffer,1,rhi::ShaderStage::Compute},{8,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_setLayout=d.createDescriptorSetLayout(ld); m_set=d.createDescriptorSet(*m_setLayout);

    rhi::ShaderDesc sd; sd.stage=rhi::ShaderStage::Compute; sd.entryPoint="cs_main";
    auto sh=rhi::VkRHIShader::createFromFile(vkD,sd,shaderDir()/"gi"/"rsm"/"rsm_sample.spv");
    rhi::ComputePSODesc pd; pd.debugName="RsmSample"; pd.computeShader=sh.get(); pd.descriptorSetLayouts={m_setLayout.get()}; pd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(RsmSamplePC)}};
    m_pipeline=d.createComputePSO(pd);
}

void RsmSamplePass::destroy() {
    m_linearClamp.reset();
    m_set.reset(); m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice=nullptr;
}

void RsmSamplePass::bindFrame(const RenderTargets& rt, VkBuffer frameUbo, VkBuffer rsmUbo,
                               const Image& rsmPos, const Image& rsmN, const Image& rsmFlux) {
    if(!m_set)return; auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_set->write({
        {0,rhi::DescriptorType::UniformBuffer,nullptr,rhi::VkRHIBuffer::createNonOwning(vkD,frameUbo,VK_WHOLE_SIZE).get()},
        {1,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rt.gNormalRough.view()).get()},
        {2,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rt.depth.view()).get()},
        {3,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rsmPos.view()).get()},
        {4,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rsmN.view()).get()},
        {5,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rsmFlux.view()).get()},
        {6,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,m_linearClamp->nativeHandle()},
        {7,rhi::DescriptorType::UniformBuffer,nullptr,rhi::VkRHIBuffer::createNonOwning(vkD,rsmUbo,VK_WHOLE_SIZE).get()},
        {8,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,rt.rsmGI.view()).get()},
    });
}

void RsmSamplePass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt) {
    if(!m_pipeline||!m_set)return; cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0,*m_set);
    RsmSamplePC pc{}; pc.outSizeX=rt.extent.width; pc.outSizeY=rt.extent.height; pc.invOutSizeX=1.f/rt.extent.width; pc.invOutSizeY=1.f/rt.extent.height; pc.radius=radius; pc.sampleCount=(uint32_t)sampleCount; pc.intensity=intensity;
    cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc)); cmd.dispatch((rt.extent.width+7)/8,(rt.extent.height+7)/8,1);
}
void RsmSamplePass::record(VkCommandBuffer vkCmd, const RenderTargets& rt) {
    rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice),vkCmd); record(rhiCmd,rt);
}
} // namespace somegi
