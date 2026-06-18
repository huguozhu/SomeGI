// PrtBakePass RHI — 7 bindings: voxel+sampler+5×PRT transfer storage。

#include "renderer/gi/prt/prt_bake_pass.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include "renderer/gi/prt/prt_resources.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/base/sampler.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>
namespace somegi {
namespace { struct BakePC { float prtGridMinX,prtGridMinY,prtGridMinZ,prtCellSize; uint32_t prtResolution,numSamples; float voxelGridDim; uint32_t _pad0; float vxgiGridMinX,vxgiGridMinY,vxgiGridMinZ,vxgiCellSize; uint32_t vxgiResolution,_p1,_p2,_p3; };
static_assert(sizeof(BakePC)==64); }
PrtBakePass::~PrtBakePass()=default;
void PrtBakePass::init(rhi::RHIDevice& d){ m_rhiDevice=&d; auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter=si.minFilter=VK_FILTER_LINEAR; si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR; si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod=0.f;
    m_linearClamp = d.createSampler({rhi::Filter::Linear,rhi::Filter::Linear,rhi::SamplerMipmapMode::Linear,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,0.f});
    rhi::DescSetLayoutDesc ld; ld.debugName="PrtBake"; ld.bindings={{0,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute},{5,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute},{6,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_setLayout=d.createDescriptorSetLayout(ld); m_set=d.createDescriptorSet(*m_setLayout);
    rhi::ShaderDesc sd; sd.stage=rhi::ShaderStage::Compute; sd.entryPoint="cs_main";
    auto sh=rhi::VkRHIShader::createFromFile(vkD,sd,shaderDir()/"gi"/"prt"/"prt_bake.spv");
    rhi::ComputePSODesc pd; pd.debugName="PrtBake"; pd.computeShader=sh.get(); pd.descriptorSetLayouts={m_setLayout.get()}; pd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(BakePC)}};
    m_pipeline=d.createComputePSO(pd);
}
void PrtBakePass::destroy(){ m_linearClamp.reset(); m_set.reset(); m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice=nullptr; }
void PrtBakePass::bindResources(const VxgiResources& vxgi,const PrtResources& prt){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_set->write({{0,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.fullView()).get()},{1,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,m_linearClamp.get()},{2,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,prt.view()).get()},{3,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,prt.viewB()).get()},{4,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,prt.viewC()).get()},{5,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,prt.viewD()).get()},{6,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,prt.viewE()).get()}});
}
void PrtBakePass::record(rhi::RHICommandBuffer& cmd,const glm::vec3& pgm,float pcs,uint32_t pr,const glm::vec3& vgm,float vcs,uint32_t vr,uint32_t ns){ if(!m_pipeline||!m_set)return;
    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0,*m_set);
    BakePC pc{pgm.x,pgm.y,pgm.z,pcs,pr,ns,vcs*(float)vr,0,vgm.x,vgm.y,vgm.z,vcs,vr};
    cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc)); cmd.dispatch((pr+3)/4,(pr+3)/4,(pr+3)/4);
}
void PrtBakePass::record(VkCommandBuffer vkCmd,const glm::vec3& pgm,float pcs,uint32_t pr,const glm::vec3& vgm,float vcs,uint32_t vr,uint32_t ns){ rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice),vkCmd); record(rhiCmd,pgm,pcs,pr,vgm,vcs,vr,ns); }
} // namespace somegi
