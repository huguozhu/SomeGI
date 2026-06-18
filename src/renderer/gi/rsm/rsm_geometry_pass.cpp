// RsmGeometryPass RHI — Graphics MRT (3 color + depth) with vertex input.
// barrier/transition 通过 nativeHandle 桥接。

#include "renderer/gi/rsm/rsm_geometry_pass.h"
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
#include "rhi/vulkan/vk_sampler.h"
#include "core/shader.h"
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <cstring>
#include <limits>

namespace somegi {

namespace {
struct RsmFrameUbo { glm::mat4 sunViewProj, sunView; glm::vec4 sunDir, sunColor_intensity; };
struct PC { glm::mat4 model; int materialIndex; int p0,p1,p2; };
static_assert(sizeof(PC)==80);
}

// 过渡函数（仍用原生 Vk —— RHI 不管理子资源 barrier）
static void transition(VkCommandBuffer cmd, VkImage img, VkImageAspectFlags aspect,
                       VkImageLayout oldL, VkImageLayout newL,
                       VkPipelineStageFlags2 ss, VkAccessFlags2 sa,
                       VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask=ss; b.srcAccessMask=sa; b.dstStageMask=ds; b.dstAccessMask=da;
    b.oldLayout=oldL; b.newLayout=newL; b.image=img;
    b.subresourceRange={aspect,0,1,0,1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount=1; di.pImageMemoryBarriers=&b;
    vkCmdPipelineBarrier2(cmd,&di);
}

void RsmGeometryPass::init(Device& d, rhi::RHIDevice& rhiDevice, uint32_t maxTextures) {
    m_device=&d; m_rhiDevice=&rhiDevice; m_maxTextures=maxTextures;
    auto mkImg=[&](VkFormat fmt, Image& img, VkImageUsageFlags usage, VkImageAspectFlags aspect=0){
        ImageDesc id; id.format=fmt; id.extent={kRsmSize,kRsmSize,1}; id.usage=usage;
        if(aspect)id.aspect=aspect; img=Image(d,id);
    };
    mkImg(VK_FORMAT_R16G16B16A16_SFLOAT,m_position,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
    mkImg(VK_FORMAT_R16G16B16A16_SFLOAT,m_normal,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
    mkImg(VK_FORMAT_R16G16B16A16_SFLOAT,m_flux,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
    mkImg(VK_FORMAT_D32_SFLOAT,m_depth,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,VK_IMAGE_ASPECT_DEPTH_BIT);
    m_rsmFrameUbo=Buffer(d,sizeof(RsmFrameUbo),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    rhi::DescSetLayoutDesc ld; ld.debugName="RsmGeom";
    ld.bindings={{0,rhi::DescriptorType::UniformBuffer,1,rhi::ShaderStage::Vertex|rhi::ShaderStage::Fragment},{1,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Fragment},{2,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Fragment},{3,rhi::DescriptorType::SampledImage,maxTextures,rhi::ShaderStage::Fragment,true},{10,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Vertex}};
    m_setLayout=rhiDevice.createDescriptorSetLayout(ld); m_set=rhiDevice.createDescriptorSet(*m_setLayout);

    // 初始 UBO 绑定(binding 0)
    auto ubo=rhi::VkRHIBuffer::createNonOwning(static_cast<rhi::VkRHIDevice&>(rhiDevice),m_rsmFrameUbo.handle(),sizeof(RsmFrameUbo));
    m_set->write({{0,rhi::DescriptorType::UniformBuffer,nullptr,ubo.get()}});

    // Graphics PSO
    auto& vkD=static_cast<rhi::VkRHIDevice&>(rhiDevice);
    auto spv=shaderDir()/"gi"/"rsm"/"rsm_geometry.spv";
    rhi::ShaderDesc vsd,fsd; vsd.stage=rhi::ShaderStage::Vertex; vsd.entryPoint="vs_main"; fsd.stage=rhi::ShaderStage::Fragment; fsd.entryPoint="ps_main";
    auto vs=rhi::VkRHIShader::createFromFile(vkD,vsd,spv); auto fs=rhi::VkRHIShader::createFromFile(vkD,fsd,spv);

    rhi::GraphicsPSODesc pd; pd.debugName="RsmGeom"; pd.vertexShader=vs.get(); pd.fragmentShader=fs.get();
    pd.vertexInput.bindings={{0,sizeof(Vertex),false}};
    pd.vertexInput.attributes={{0,rhi::VertexFormat::Float3,offsetof(Vertex,position),0},{1,rhi::VertexFormat::Float3,offsetof(Vertex,normal),0},{3,rhi::VertexFormat::Float2,offsetof(Vertex,uv0),0}};
    pd.topology=rhi::PrimitiveTopology::TriangleList;
    pd.rasterization.cull=rhi::CullMode::Back; pd.rasterization.frontCCW=true;
    pd.depthStencil.depthTest=true; pd.depthStencil.depthWrite=true; pd.depthStencil.depthCompare=rhi::CompareFunc::LessEqual;
    pd.blend.attachments={{false},{false},{false}};
    pd.renderTargets.colorFormats={rhi::Format::R16G16B16A16_SFLOAT,rhi::Format::R16G16B16A16_SFLOAT,rhi::Format::R16G16B16A16_SFLOAT};
    pd.renderTargets.depthFormat=rhi::Format::D32_SFLOAT; pd.renderTargets.sampleCount=1;
    pd.descriptorSetLayouts={m_setLayout.get()};
    m_pipeline=rhiDevice.createGraphicsPSO(pd);
}

void RsmGeometryPass::destroy() {
    m_set.reset(); m_pipeline.reset(); m_setLayout.reset();
    m_rsmFrameUbo.reset(); m_position.reset(); m_normal.reset(); m_flux.reset(); m_depth.reset();
    m_device=nullptr; m_rhiDevice=nullptr;
}

void RsmGeometryPass::bindScene(const SceneGpu& gpu, uint32_t tc) {
    auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto matBuf=rhi::VkRHIBuffer::createNonOwning(vkD,gpu.materialBuffer.handle(),VK_WHOLE_SIZE);
    m_texViews.clear(); m_texViewPtrs.clear(); m_texViews.reserve(m_maxTextures); m_texViewPtrs.reserve(m_maxTextures);
    for(uint32_t i=0;i<m_maxTextures;++i){ VkImageView v=(i<tc&&i<gpu.images.size())?gpu.images[i].view():gpu.whiteTex.view(); m_texViews.push_back(rhi::VkRHITextureView::createNonOwning(vkD,v)); m_texViewPtrs.push_back(m_texViews.back().get()); }
    auto rsmSampler = rhi::VkRHISampler::createNonOwning(vkD, gpu.linearSampler);
    rhi::DescriptorWrite w[3]={};
    w[0]={1,rhi::DescriptorType::StorageBuffer,nullptr,matBuf.get()};
    w[1]={2,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,rsmSampler.get()};
    w[2]={3,rhi::DescriptorType::SampledImage,nullptr,nullptr,0,0,nullptr,nullptr,m_maxTextures,m_texViewPtrs.data()};
    m_set->write({w,w+3});
}

void RsmGeometryPass::bindDrawData(VkBuffer drawBuf) {
    auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto dd=rhi::VkRHIBuffer::createNonOwning(vkD,drawBuf,VK_WHOLE_SIZE);
    m_set->write({{10,rhi::DescriptorType::StorageBuffer,nullptr,dd.get()}});
}

void RsmGeometryPass::updateLight(const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                                   const glm::vec3& sunDir, const glm::vec3& sunColor, float sunIntensity) {
    glm::vec3 ld=glm::normalize(sunDir), toSun=-ld;
    glm::vec3 sc=(aabbMin+aabbMax)*0.5f, sz=aabbMax-aabbMin; float diag=glm::length(sz);
    glm::vec3 sp=sc+toSun*diag;
    glm::vec3 up=(std::abs(toSun.y)<0.999f)?glm::vec3(0,1,0):glm::vec3(1,0,0);
    glm::mat4 view=glm::lookAt(sp,sc,up);
    glm::vec3 mn{FLT_MAX},mx{-FLT_MAX};
    for(int i=0;i<8;++i){ glm::vec3 c((i&1)?aabbMax.x:aabbMin.x,(i&2)?aabbMax.y:aabbMin.y,(i&4)?aabbMax.z:aabbMin.z); glm::vec3 v=glm::vec3(view*glm::vec4(c,1.f)); mn=glm::min(mn,v); mx=glm::max(mx,v); }
    float pad=diag*0.05f; mn-=glm::vec3(pad); mx+=glm::vec3(pad);
    glm::mat4 proj=glm::ortho(mn.x,mx.x,mn.y,mx.y,-mx.z,-mn.z); proj[1][1]*=-1.f;
    RsmFrameUbo u{proj*view,view,glm::vec4(ld,0),glm::vec4(sunColor,sunIntensity)};
    std::memcpy(m_rsmFrameUbo.mapped(),&u,sizeof(u));
}

void RsmGeometryPass::record(rhi::RHICommandBuffer& cmd, VkBuffer indirectBuf, uint32_t drawCount, const SceneGpu& gpu) {
    if(!drawCount||!m_pipeline)return;
    VkCommandBuffer vkCmd=(VkCommandBuffer)(uintptr_t)cmd.nativeHandle();
    auto t=[&](VkImage img,VkImageAspectFlags a,VkImageLayout ol,VkImageLayout nl,VkPipelineStageFlags2 ss,VkAccessFlags2 sa,VkPipelineStageFlags2 ds,VkAccessFlags2 da){ transition(vkCmd,img,a,ol,nl,ss,sa,ds,da); };
    t(m_position.image(),VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,0,VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    t(m_normal.image(),VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,0,VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    t(m_flux.image(),VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,0,VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    t(m_depth.image(),VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,0,VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto cv0=rhi::VkRHITextureView::createNonOwning(vkD,m_position.view());
    auto cv1=rhi::VkRHITextureView::createNonOwning(vkD,m_normal.view());
    auto cv2=rhi::VkRHITextureView::createNonOwning(vkD,m_flux.view());
    auto dv=rhi::VkRHITextureView::createNonOwning(vkD,m_depth.view());
    rhi::RenderingAttachmentInfo cAttach[3]{};
    for(int i=0;i<3;++i){cAttach[i].view=(i==0?cv0:i==1?cv1:cv2).get();cAttach[i].loadOp=rhi::AttachmentLoadOp::Clear;}
    rhi::RenderingAttachmentInfo dAttach{}; dAttach.view=dv.get(); dAttach.loadOp=rhi::AttachmentLoadOp::Clear; dAttach.clearDepth=1.f;
    cmd.beginRendering(cAttach,3,&dAttach,kRsmSize,kRsmSize);
    cmd.setViewport(0,0,(float)kRsmSize,(float)kRsmSize);
    cmd.setScissor(0,0,kRsmSize,kRsmSize);
    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0,*m_set);
    auto vb=rhi::VkRHIBuffer::createNonOwning(vkD,gpu.vertexBuffer.handle(),VK_WHOLE_SIZE);
    auto ib=rhi::VkRHIBuffer::createNonOwning(vkD,gpu.indexBuffer.handle(),VK_WHOLE_SIZE);
    auto cb=rhi::VkRHIBuffer::createNonOwning(vkD,indirectBuf,VK_WHOLE_SIZE);
    cmd.bindVertexBuffer(0,*vb); cmd.bindIndexBuffer(*ib,0,false);
    cmd.drawIndexedIndirectCount(*cb,0,*cb,0,drawCount,sizeof(VkDrawIndexedIndirectCommand));
    cmd.endRendering();

    t(m_position.image(),VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    t(m_normal.image(),VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    t(m_flux.image(),VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    t(m_depth.image(),VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void RsmGeometryPass::record(VkCommandBuffer vkCmd, VkBuffer ib, uint32_t dc, const SceneGpu& gpu) {
    rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice),vkCmd); record(rhiCmd,ib,dc,gpu);
}

} // namespace somegi
