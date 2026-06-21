// VxgiRelightPass RHI — 4 bindings: voxel+aniso+sampler+dst。3 descriptor sets。

#include "renderer/gi/vxgi/vxgi_relight_pass.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/base/sampler.h"
#include "rhi/vulkan/vk_pso.h"
#include "core/shader.h"
#include <array>
namespace somegi {
namespace { struct RelightPC { uint32_t gridRes,mipLevels; float cellSize,bounceStrength,gridMinX,gridMinY,gridMinZ,_p0; };
static_assert(sizeof(RelightPC)==32); }
VxgiRelightPass::~VxgiRelightPass()=default;
void VxgiRelightPass::init(rhi::RHIDevice& d){ m_rhiDevice=&d; auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter=si.minFilter=VK_FILTER_LINEAR; si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR; si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod=16.f;
    m_linearClamp = d.createSampler({rhi::Filter::Linear,rhi::Filter::Linear,rhi::SamplerMipmapMode::Linear,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,16.f});
    rhi::DescSetLayoutDesc ld; ld.debugName="VxgiRelight"; ld.bindings={{0,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Compute},{3,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_setLayout=d.createDescriptorSetLayout(ld); m_set=d.createDescriptorSet(*m_setLayout); m_setPP0=d.createDescriptorSet(*m_setLayout); m_setPP1=d.createDescriptorSet(*m_setLayout);
    rhi::ShaderDesc sd; sd.stage=rhi::ShaderStage::Compute; sd.entryPoint="cs_main";
    auto sh=rhi::VkRHIShader::createFromFile(vkD,sd,shaderDir()/"gi"/"vxgi"/"vxgi_relight.spv");
    rhi::ComputePSODesc pd; pd.debugName="VxgiRelight"; pd.computeShader=sh.get(); pd.descriptorSetLayouts={m_setLayout.get()}; pd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(RelightPC)}};
    m_pipeline=d.createComputePSO(pd);
}
void VxgiRelightPass::destroy(){ m_linearClamp.reset(); m_setPP1.reset(); m_setPP0.reset(); m_set.reset(); m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice=nullptr; }
void VxgiRelightPass::bindResources(const VxgiResources& vxgi,VkImageView dstView){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_set->write({{0,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.fullView()).get()},{1,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.anisoFullView()).get()},{2,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,m_linearClamp.get()},{3,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,dstView).get()}});
}
void VxgiRelightPass::bindResourcesPingPong(const VxgiResources& vxgi,bool sw){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto& target=sw?*m_setPP1:*m_setPP0;
    target.write({{0,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,sw?vxgi.relightScratch2View():vxgi.relightScratchView()).get()},{1,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,vxgi.anisoFullView()).get()},{2,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,m_linearClamp.get()},{3,rhi::DescriptorType::StorageImage,rhi::VkRHITextureView::createNonOwning(vkD,sw?vxgi.relightScratchView():vxgi.relightScratch2View()).get()}});
}
// RHI 路径：体素 relight compute dispatch
void VxgiRelightPass::record(rhi::RHICommandBuffer& cmd,const rhi::RHIDescriptorSet& set,uint32_t gr,uint32_t ml,float cs,const glm::vec3& gm,float bs){
    cmd.bindPipelineState(*m_pipeline);
    cmd.bindDescriptorSet(0, set);
    RelightPC pc{gr,ml,cs,bs,gm.x,gm.y,gm.z};
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
    cmd.dispatch((gr+3)/4, (gr+3)/4, (gr+3)/4);
}
} // namespace somegi
