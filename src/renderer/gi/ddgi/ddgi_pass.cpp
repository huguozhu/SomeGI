// DdgiPass RHI — 4 pipelines, 3 layouts, record 通过 VkCompat 保留复杂 barrier。
#include "renderer/gi/ddgi/ddgi_pass.h"
#include "renderer/gi/ddgi/ddgi_resources.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
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
namespace { struct UpdatePC { float ddgiOriginX,ddgiOriginY,ddgiOriginZ,_p0,ddgiSpacingX,ddgiSpacingY,ddgiSpacingZ,_p1; uint32_t probesX,probesY,probesZ,raysPerProbe; float randomRotation,voxelGridDim,vxgiCellSize; uint32_t vxgiResolution; float vxgiGridMinX,vxgiGridMinY,vxgiGridMinZ,_p2; };
struct BlendPC { uint32_t probesX,probesY,probesZ,raysPerProbe,octaIrr,octaDist; float hysteresis,maxRayDistance; };
struct ClassifyPC { uint32_t probeCount,raysPerProbe; float closeHitDist,closeHitFrac; }; }

DdgiPass::~DdgiPass()=default;
void DdgiPass::init(rhi::RHIDevice& d){ m_rhiDevice=&d; auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter=si.minFilter=VK_FILTER_LINEAR; si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    m_linearClamp = d.createSampler({rhi::Filter::Linear,rhi::Filter::Linear,rhi::SamplerMipmapMode::Linear,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,rhi::SamplerAddressMode::ClampToEdge,0.f});
    // Update: 0=voxel, 1=sampler, 2=rayBuf
    rhi::DescSetLayoutDesc uld; uld.debugName="DDGI_Update"; uld.bindings={{0,rhi::DescriptorType::SampledImage,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute}};
    m_setLayout=d.createDescriptorSetLayout(uld); m_setUpdate=d.createDescriptorSet(*m_setLayout);
    // Blend: 0=rayBuf, 1=irrAtlas, 2=distAtlas
    rhi::DescSetLayoutDesc bld; bld.debugName="DDGI_Blend"; bld.bindings={{0,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute},{2,rhi::DescriptorType::StorageImage,1,rhi::ShaderStage::Compute}};
    m_setLayoutBlend=d.createDescriptorSetLayout(bld); m_setBlend=d.createDescriptorSet(*m_setLayoutBlend);
    // Classify: 0=rayBuf, 1=probeStates
    rhi::DescSetLayoutDesc cld; cld.debugName="DDGI_Classify"; cld.bindings={{0,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute},{1,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Compute}};
    m_setLayoutClassify=d.createDescriptorSetLayout(cld); m_setClassify=d.createDescriptorSet(*m_setLayoutClassify);
    // Pipelines
    auto mk=[&](auto spv,const char* ep,rhi::RHIDescriptorSetLayout* l,uint32_t pcSize){
        rhi::ShaderDesc sd; sd.stage=rhi::ShaderStage::Compute; sd.entryPoint=ep;
        auto sh=rhi::VkRHIShader::createFromFile(vkD,sd,shaderDir()/spv);
        rhi::ComputePSODesc pd; pd.computeShader=sh.get(); pd.descriptorSetLayouts={l}; if(pcSize)pd.pushConstants={{rhi::ShaderStage::Compute,0,pcSize}};
        return d.createComputePSO(pd);
    };
    m_pipelineUpdate=mk("gi/ddgi/ddgi_update.spv","cs_main",m_setLayout.get(),sizeof(UpdatePC));
    m_pipelineClassify=mk("gi/ddgi/ddgi_classify.spv","cs_main",m_setLayoutClassify.get(),sizeof(ClassifyPC));
    m_pipelineBlendIrr=mk("gi/ddgi/ddgi_blend.spv","cs_irradiance",m_setLayoutBlend.get(),sizeof(BlendPC));
    m_pipelineBlendDist=mk("gi/ddgi/ddgi_blend.spv","cs_distance",m_setLayoutBlend.get(),sizeof(BlendPC));
}
void DdgiPass::destroy(){ m_linearClamp.reset();
    m_setClassify.reset(); m_setBlend.reset(); m_setUpdate.reset(); m_pipelineBlendDist.reset(); m_pipelineBlendIrr.reset(); m_pipelineClassify.reset(); m_pipelineUpdate.reset(); m_setLayoutClassify.reset(); m_setLayoutBlend.reset(); m_setLayout.reset(); m_rhiDevice=nullptr; }
void DdgiPass::bindResources(const DdgiResources& ddgi,const VxgiResources& vxgi){ auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto vox=rhi::VkRHITextureView::createNonOwning(vkD,vxgi.fullView());
    auto irr=rhi::VkRHITextureView::createNonOwning(vkD,ddgi.irradiance().view());
    auto dist=rhi::VkRHITextureView::createNonOwning(vkD,ddgi.distance().view());
    auto rb=rhi::VkRHIBuffer::createNonOwning(vkD,ddgi.rayBuffer().handle(),VK_WHOLE_SIZE);
    auto ps=rhi::VkRHIBuffer::createNonOwning(vkD,ddgi.probeStates().handle(),VK_WHOLE_SIZE);
    m_setUpdate->write({{0,rhi::DescriptorType::SampledImage,vox.get()},{1,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,m_linearClamp.get()},{2,rhi::DescriptorType::StorageBuffer,nullptr,rb.get()}});
    m_setBlend->write({{0,rhi::DescriptorType::StorageBuffer,nullptr,rb.get()},{1,rhi::DescriptorType::StorageImage,irr.get()},{2,rhi::DescriptorType::StorageImage,dist.get()}});
    m_setClassify->write({{0,rhi::DescriptorType::StorageBuffer,nullptr,rb.get()},{1,rhi::DescriptorType::StorageBuffer,nullptr,ps.get()}});
}
// record 保留 VkCompat（复杂 barrier + 多 dispatch）
void DdgiPass::record(VkCommandBuffer vkCmd,const DdgiResources&,const glm::vec3& dO,const glm::vec3& dS,const glm::vec3& vM,float vC,uint32_t vR,float rR,uint32_t){
    auto hnd=[&](auto& pso)->VkPipeline{return (VkPipeline)(uintptr_t)pso->nativeHandle();};
    auto lay=[&](auto& pso)->VkPipelineLayout{return static_cast<rhi::VkRHIPipelineState&>(*pso).layout();};
    VkDescriptorSet setU=(VkDescriptorSet)(uintptr_t)m_setUpdate->nativeHandle();
    VkDescriptorSet setB=(VkDescriptorSet)(uintptr_t)m_setBlend->nativeHandle();
    VkDescriptorSet setC=(VkDescriptorSet)(uintptr_t)m_setClassify->nativeHandle();
    vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,hnd(m_pipelineUpdate));
    vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,lay(m_pipelineUpdate),0,1,&setU,0,nullptr);
    UpdatePC upc{};upc.ddgiOriginX=dO.x;upc.ddgiOriginY=dO.y;upc.ddgiOriginZ=dO.z;upc.ddgiSpacingX=dS.x;upc.ddgiSpacingY=dS.y;upc.ddgiSpacingZ=dS.z;upc.probesX=DdgiResources::kProbesX;upc.probesY=DdgiResources::kProbesY;upc.probesZ=DdgiResources::kProbesZ;upc.raysPerProbe=DdgiResources::kRaysPerProbe;upc.randomRotation=rR;upc.vxgiCellSize=vC;upc.vxgiResolution=vR;upc.vxgiGridMinX=vM.x;upc.vxgiGridMinY=vM.y;upc.vxgiGridMinZ=vM.z;upc.voxelGridDim=vC*(float)vR;
    vkCmdPushConstants(vkCmd,lay(m_pipelineUpdate),VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(upc),&upc);
    vkCmdDispatch(vkCmd,(DdgiResources::kProbeCount*DdgiResources::kRaysPerProbe+63)/64,1,1);
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};mb.srcStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;mb.srcAccessMask=VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;mb.dstStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;mb.dstAccessMask=VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};di.memoryBarrierCount=1;di.pMemoryBarriers=&mb;vkCmdPipelineBarrier2(vkCmd,&di);
    {ClassifyPC cpc{DdgiResources::kProbeCount,DdgiResources::kRaysPerProbe,vC*0.5f,0.7f}; vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,hnd(m_pipelineClassify)); vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,lay(m_pipelineClassify),0,1,&setC,0,nullptr);
        vkCmdPushConstants(vkCmd,lay(m_pipelineClassify),VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cpc),&cpc); vkCmdDispatch(vkCmd,(DdgiResources::kProbeCount+63)/64,1,1); vkCmdPipelineBarrier2(vkCmd,&di);}
    BlendPC bpc{DdgiResources::kProbesX,DdgiResources::kProbesY,DdgiResources::kProbesZ,DdgiResources::kRaysPerProbe,DdgiResources::kOctaIrr,DdgiResources::kOctaDist,0.92f,vC*(float)vR};
    vkCmdBindDescriptorSets(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,lay(m_pipelineBlendIrr),0,1,&setB,0,nullptr);
    vkCmdPushConstants(vkCmd,lay(m_pipelineBlendIrr),VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(bpc),&bpc);
    vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,hnd(m_pipelineBlendIrr));
    {uint32_t aw=DdgiResources::irradianceAtlasW(),ah=DdgiResources::irradianceAtlasH(); vkCmdDispatch(vkCmd,(aw+7)/8,(ah+7)/8,1);}
    vkCmdBindPipeline(vkCmd,VK_PIPELINE_BIND_POINT_COMPUTE,hnd(m_pipelineBlendDist));
    {uint32_t aw=DdgiResources::distanceAtlasW(),ah=DdgiResources::distanceAtlasH(); vkCmdDispatch(vkCmd,(aw+7)/8,(ah+7)/8,1);}
}
} // namespace somegi
