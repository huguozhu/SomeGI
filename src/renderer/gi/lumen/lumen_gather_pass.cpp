// LumenGatherPass RHI — cs_gather
#include "renderer/gi/lumen/lumen_gather_pass.h"
#include "renderer/gi/lumen/lumen_resources.h"
#include "renderer/core/render_targets.h"
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
#include "core/device.h"
#include "core/shader.h"
namespace somegi {
namespace { struct GatherPC { float invScreenSizeX,invScreenSizeY,probeTileSize; uint32_t probeGridW,probeGridH,debugMode; };
static_assert(sizeof(GatherPC)==24); }
LumenGatherPass::~LumenGatherPass()=default;
void LumenGatherPass::init(rhi::RHIDevice& d){ m_rhiDevice=&d; auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter=si.minFilter=VK_FILTER_NEAREST; si.addressModeU=si.addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    m_pointClamp = d.createSampler({rhi::Filter::Nearest,rhi::Filter::Nearest,rhi::SamplerMipmapMode::Nearest,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,0.f});
    rhi::DescSetLayoutDesc ld; ld.debugName="LumenGather"; ld.bindings={{0,rhi::DescriptorType::UniformBuffer,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{5,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{6,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_setLayout=d.createDescriptorSetLayout(ld); m_set=d.createDescriptorSet(*m_setLayout);
    rhi::ShaderDesc sd; sd.stage=rhi::ShaderStage::Compute; sd.entryPoint="cs_gather";
    auto sh=rhi::VkRHIShader::createFromFile(vkD,sd,shaderDir()/"gi"/"lumen"/"lumen_gather.spv");
    rhi::ComputePSODesc pd; pd.debugName="LumenGather"; pd.computeShader=sh.get(); pd.descriptorSetLayouts={m_setLayout.get()}; pd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(GatherPC)}};
    m_pipeline=d.createComputePSO(pd);
}
void LumenGatherPass::destroy(){ m_pointClamp.reset(); m_set.reset(); m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice=nullptr; }
void LumenGatherPass::bindResources(const LumenResources& res,const RenderTargets& rt,VkBuffer frameUbo,bool useFiltered){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    VkImageView av=useFiltered?res.filteredAtlas().view():res.probeAtlas().view();
    m_set->write({{0,rhi::DescriptorType::UniformBuffer,nullptr,rhi::VkRHIBuffer::createNonOwning(vkD,frameUbo,VK_WHOLE_SIZE).get()},{1,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,av).get()},{2,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,m_pointClamp.get()},{3,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rt.gAlbedoMetal.view()).get()},{4,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rt.gNormalRough.view()).get()},{5,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rt.depth.view()).get()},{6,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,rt.lumenGI.view()).get()}});
}
void LumenGatherPass::record(rhi::RHICommandBuffer& cmd,const LumenResources& res,const RenderTargets& rt,uint32_t dm){ if(!m_pipeline||!m_set)return;
    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0,*m_set);
    GatherPC pc{1.f/rt.extent.width,1.f/rt.extent.height,(float)LumenResources::kProbeTileSize,res.probeGridW(),res.probeGridH(),dm};
    cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc)); cmd.dispatch((rt.extent.width+7)/8,(rt.extent.height+7)/8,1);
}
void LumenGatherPass::record(VkCommandBuffer vkCmd,const LumenResources& res,const RenderTargets& rt,uint32_t dm){ rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice),vkCmd); record(rhiCmd,res,rt,dm); }
} // namespace somegi
