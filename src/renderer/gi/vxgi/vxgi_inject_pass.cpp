// VxgiInjectPass RHI — 3 bindings: RSMpos+RSMflux+voxelGrid.

#include "renderer/gi/vxgi/vxgi_inject_pass.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_command.h"
#include "core/shader.h"
#include <array>
namespace somegi {
namespace { struct InjectPC { uint32_t rsmSizeX,rsmSizeY,gridRes,_pad; float gridMinX,gridMinY,gridMinZ,cellSize; };
static_assert(sizeof(InjectPC)==32); }
VxgiInjectPass::~VxgiInjectPass()=default;
void VxgiInjectPass::init(rhi::RHIDevice& d,uint32_t rs){ m_rhiDevice=&d; m_rsmSize=rs;
    rhi::DescSetLayoutDesc ld; ld.debugName="VxgiInject"; ld.bindings={{0,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_setLayout=d.createDescriptorSetLayout(ld); m_set=d.createDescriptorSet(*m_setLayout);
    auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
    rhi::ShaderDesc sd; sd.stage=rhi::ShaderStage::Compute; sd.entryPoint="cs_main";
    auto sh=rhi::VkRHIShader::createFromFile(vkD,sd,shaderDir()/"gi"/"vxgi"/"vxgi_inject.spv");
    rhi::ComputePSODesc pd; pd.debugName="VxgiInject"; pd.computeShader=sh.get(); pd.descriptorSetLayouts={m_setLayout.get()}; pd.pushConstants={{rhi::ShaderStage::Compute,0,sizeof(InjectPC)}};
    m_pipeline=d.createComputePSO(pd);
}
void VxgiInjectPass::destroy(){ m_set.reset(); m_pipeline.reset(); m_setLayout.reset(); m_rhiDevice=nullptr; }
void VxgiInjectPass::bindResources(VkImageView rsmPosView,VkImageView rsmFluxView,const VxgiResources& vxgi){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_set->write({{0,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rsmPosView).get()},{1,rhi::DescriptorType::SampledImage,rhi::VkRHITextureView::createNonOwning(vkD,rsmFluxView).get()},{2,rhi::DescriptorType::StorageImage,vxgi.mipView(0)}});
}
void VxgiInjectPass::record(rhi::RHICommandBuffer& cmd,uint32_t gr,const glm::vec3& gm,float cs){ if(!m_pipeline||!m_set)return;
    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0,*m_set);
    InjectPC pc{(uint32_t)m_rsmSize,(uint32_t)m_rsmSize,gr,0,gm.x,gm.y,gm.z,cs}; cmd.pushConstants(rhi::ShaderStage::Compute,&pc,sizeof(pc));
    cmd.dispatch((m_rsmSize+7)/8,(m_rsmSize+7)/8,1);
}

} // namespace somegi
