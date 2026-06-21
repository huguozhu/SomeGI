// DdgiPass RHI — 4 pipelines, 3 layouts, 纯 RHI record()。
#include "renderer/gi/ddgi/ddgi_pass.h"
#include "renderer/gi/ddgi/ddgi_resources.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_pso.h"
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
// record — 纯 RHI，不依赖 VkCompat
void DdgiPass::record(rhi::RHICommandBuffer& cmd, const DdgiResources&, const glm::vec3& dO, const glm::vec3& dS, const glm::vec3& vM, float vC, uint32_t vR, float rR, uint32_t) {
    // 1. Update dispatch
    cmd.bindPipelineState(*m_pipelineUpdate);
    cmd.bindDescriptorSet(0, *m_setUpdate);
    UpdatePC upc{};
    upc.ddgiOriginX = dO.x; upc.ddgiOriginY = dO.y; upc.ddgiOriginZ = dO.z;
    upc.ddgiSpacingX = dS.x; upc.ddgiSpacingY = dS.y; upc.ddgiSpacingZ = dS.z;
    upc.probesX = DdgiResources::kProbesX; upc.probesY = DdgiResources::kProbesY; upc.probesZ = DdgiResources::kProbesZ;
    upc.raysPerProbe = DdgiResources::kRaysPerProbe; upc.randomRotation = rR;
    upc.vxgiCellSize = vC; upc.vxgiResolution = vR;
    upc.vxgiGridMinX = vM.x; upc.vxgiGridMinY = vM.y; upc.vxgiGridMinZ = vM.z;
    upc.voxelGridDim = vC * (float)vR;
    cmd.pushConstants(rhi::ShaderStage::Compute, &upc, sizeof(upc));
    cmd.dispatch((DdgiResources::kProbeCount * DdgiResources::kRaysPerProbe + 63) / 64, 1, 1);
    cmd.globalBarrier();

    // 2. Classify dispatch
    cmd.bindPipelineState(*m_pipelineClassify);
    cmd.bindDescriptorSet(0, *m_setClassify);
    {ClassifyPC cpc{DdgiResources::kProbeCount, DdgiResources::kRaysPerProbe, vC * 0.5f, 0.7f};
        cmd.pushConstants(rhi::ShaderStage::Compute, &cpc, sizeof(cpc));
        cmd.dispatch((DdgiResources::kProbeCount + 63) / 64, 1, 1);}
    cmd.globalBarrier();

    // 3. BlendIrr dispatch
    BlendPC bpc{DdgiResources::kProbesX, DdgiResources::kProbesY, DdgiResources::kProbesZ, DdgiResources::kRaysPerProbe, DdgiResources::kOctaIrr, DdgiResources::kOctaDist, 0.92f, vC * (float)vR};
    cmd.bindDescriptorSet(0, *m_setBlend);
    cmd.pushConstants(rhi::ShaderStage::Compute, &bpc, sizeof(bpc));
    cmd.bindPipelineState(*m_pipelineBlendIrr);
    {uint32_t aw = DdgiResources::irradianceAtlasW(), ah = DdgiResources::irradianceAtlasH(); cmd.dispatch((aw + 7) / 8, (ah + 7) / 8, 1);}

    // 4. BlendDist dispatch
    cmd.bindPipelineState(*m_pipelineBlendDist);
    {uint32_t aw = DdgiResources::distanceAtlasW(), ah = DdgiResources::distanceAtlasH(); cmd.dispatch((aw + 7) / 8, (ah + 7) / 8, 1);}
}
} // namespace somegi
