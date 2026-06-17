// VxgiResolve6AxisPass RHI — 6 bindings: voxel+aniso+sampler+axisXYZ.

#include "renderer/gi/vxgi/vxgi_resolve_6axis_pass.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_command.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>
namespace somegi {
namespace { struct SixAxisPC { uint32_t gridRes,mipLevels,_p0; float cellSize,strength,gridMinX,gridMinY,gridMinZ; };
static_assert(sizeof(SixAxisPC)==32); }
VxgiResolve6AxisPass::~VxgiResolve6AxisPass()=default;
void VxgiResolve6AxisPass::init(rhi::RHIDevice& d){ m_rhiDevice=&d; auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter=si.minFilter=VK_FILTER_LINEAR; si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR; si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod=16.f;
    vkCreateSampler(vkD.vkDevice(),&si,nullptr,&m_linearClamp);
    rhi::DescSetLayoutDesc ld; ld.debugName="Vxgi6Axis";
    ld.bindings={{0,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute},{4,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute},{5,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_setLayout=d.createDescriptorSetLayout(ld); m_set=d.createDescriptorSet(*m_setLayout);
    rhi::ShaderDesc sd; sd.stage=rhi::ShaderStage::Compute; sd.entryPoint="cs_resolve6Axis";
    auto sh=rhi::VkRHIShader::createFromFile(vkD,sd,shaderDir()/"gi"/"vxgi"/"vxgi_resolve_6axis.spv");
    rhi::ComputePSODesc pd; pd.debugName="Vxgi6Axis"; pd.computeShader=sh.get(); pd.descriptorSetLayouts={m_setLayout.get()}; pd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(SixAxisPC)}};
    m_pipeline=d.createComputePSO(pd);
}
void VxgiResolve6AxisPass::destroy(){ if(m_linearClamp)vkDestroySampler(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice).vkDevice(),m_linearClamp,nullptr); m_set.reset(); m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice=nullptr; }
void VxgiResolve6AxisPass::bindResources(const VxgiResources& vxgi){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_set->write({{0,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.fullView()).get()},{1,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.anisoFullView()).get()},{2,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,(const void*)(uintptr_t)m_linearClamp},{3,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.sixAxisX().view()).get()},{4,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.sixAxisY().view()).get()},{5,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.sixAxisZ().view()).get()}});
}
void VxgiResolve6AxisPass::record(rhi::RHICommandBuffer& cmd,uint32_t gr,uint32_t ml,float cs,const glm::vec3& gm,float st){ if(!m_pipeline||!m_set)return;
    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0,*m_set);
    SixAxisPC pc{gr,ml,0,cs,st,gm.x,gm.y,gm.z}; cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc));
    cmd.dispatch((gr+3)/4,(gr+3)/4,(gr+3)/4);
}
void VxgiResolve6AxisPass::record(VkCommandBuffer vkCmd,uint32_t gr,uint32_t ml,float cs,const glm::vec3& gm,float st){ rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice),vkCmd); record(rhiCmd,gr,ml,cs,gm,st); }
} // namespace somegi
