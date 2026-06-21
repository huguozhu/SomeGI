// RsmGeometryPass RHI — Graphics MRT (3 color + depth) with vertex input.
// barrier 通过 cmd.textureBarrier() 替代原生 vkCmdPipelineBarrier2。

#include "renderer/gi/rsm/rsm_geometry_pass.h"
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

void RsmGeometryPass::init(rhi::RHIDevice& d, uint32_t maxTextures) {
    m_rhiDevice=&d; m_maxTextures=maxTextures;

    // ── 创建渲染目标纹理（通过 RHI） ──
    auto mkImg=[&](rhi::Format fmt,
                    std::unique_ptr<rhi::RHITexture>& tex,
                    std::unique_ptr<rhi::RHITextureView>& view,
                    rhi::TextureUsage usage){
        rhi::TextureDesc td{};
        td.format=fmt; td.width=kRsmSize; td.height=kRsmSize; td.depth=1;
        td.usage=usage;
        tex=d.createTexture(td);
        view=d.createTextureView(*tex, {});
    };
    auto colorUsage=static_cast<rhi::TextureUsage>(
        static_cast<uint32_t>(rhi::TextureUsage::ColorAttachment) |
        static_cast<uint32_t>(rhi::TextureUsage::Sampled));
    auto depthUsage=static_cast<rhi::TextureUsage>(
        static_cast<uint32_t>(rhi::TextureUsage::DepthStencil) |
        static_cast<uint32_t>(rhi::TextureUsage::Sampled));
    mkImg(rhi::Format::R16G16B16A16_SFLOAT, m_positionTex, m_positionView, colorUsage);
    mkImg(rhi::Format::R16G16B16A16_SFLOAT, m_normalTex,   m_normalView,   colorUsage);
    mkImg(rhi::Format::R16G16B16A16_SFLOAT, m_fluxTex,     m_fluxView,     colorUsage);
    mkImg(rhi::Format::D32_SFLOAT,           m_depthTex,    m_depthView,    depthUsage);

    // ── 帧 UBO（RHI buffer） ──
    rhi::BufferDesc ubd{};
    ubd.size=sizeof(RsmFrameUbo);
    ubd.usage=rhi::BufferUsage::Uniform;
    ubd.memory=rhi::MemoryType::HostVisible;
    m_rsmFrameUbo=d.createBuffer(ubd);

    // ── Descriptor Set Layout ──
    rhi::DescSetLayoutDesc ld; ld.debugName="RsmGeom";
    ld.bindings={{0,rhi::DescriptorType::UniformBuffer,1,rhi::ShaderStage::Vertex|rhi::ShaderStage::Fragment},{1,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Fragment},{2,rhi::DescriptorType::Sampler,1,rhi::ShaderStage::Fragment},{3,rhi::DescriptorType::SampledImage,maxTextures,rhi::ShaderStage::Fragment,true},{10,rhi::DescriptorType::StorageBuffer,1,rhi::ShaderStage::Vertex}};
    m_setLayout=d.createDescriptorSetLayout(ld); m_set=d.createDescriptorSet(*m_setLayout);

    // 初始 UBO 绑定（binding 0）
    m_set->write({{0, rhi::DescriptorType::UniformBuffer, nullptr, m_rsmFrameUbo.get()}});

    // ── Graphics PSO ──
    auto& vkD=static_cast<rhi::VkRHIDevice&>(d);
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
    m_pipeline=d.createGraphicsPSO(pd);
}

void RsmGeometryPass::destroy() {
    m_set.reset(); m_pipeline.reset(); m_setLayout.reset();
    m_rsmFrameUbo.reset();
    m_positionView.reset(); m_normalView.reset(); m_fluxView.reset(); m_depthView.reset();
    m_positionTex.reset(); m_normalTex.reset(); m_fluxTex.reset(); m_depthTex.reset();
    m_rhiDevice=nullptr;
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
    void* ptr = m_rsmFrameUbo->map();
    std::memcpy(ptr, &u, sizeof(u));
    m_rsmFrameUbo->unmap();
}

void RsmGeometryPass::record(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& indirectBuf, uint32_t drawCount, const SceneGpu& gpu) {
    if(!drawCount||!m_pipeline)return;
    auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    // Pre-render barriers: Undefined → Color/Depth attachment
    cmd.textureBarrier(*m_positionTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::ColorAttachment);
    cmd.textureBarrier(*m_normalTex,   rhi::TextureLayout::Undefined, rhi::TextureLayout::ColorAttachment);
    cmd.textureBarrier(*m_fluxTex,     rhi::TextureLayout::Undefined, rhi::TextureLayout::ColorAttachment);
    cmd.textureBarrier(*m_depthTex,    rhi::TextureLayout::Undefined, rhi::TextureLayout::DepthAttachment);

    // ── Begin rendering（用成员 RHI view） ──
    rhi::RenderingAttachmentInfo cAttach[3]{};
    cAttach[0].view=m_positionView.get(); cAttach[0].loadOp=rhi::AttachmentLoadOp::Clear;
    cAttach[1].view=m_normalView.get();   cAttach[1].loadOp=rhi::AttachmentLoadOp::Clear;
    cAttach[2].view=m_fluxView.get();     cAttach[2].loadOp=rhi::AttachmentLoadOp::Clear;
    rhi::RenderingAttachmentInfo dAttach{};
    dAttach.view=m_depthView.get(); dAttach.loadOp=rhi::AttachmentLoadOp::Clear; dAttach.clearDepth=1.f;
    cmd.beginRendering(cAttach,3,&dAttach,kRsmSize,kRsmSize);
    cmd.setViewport(0,0,(float)kRsmSize,(float)kRsmSize);
    cmd.setScissor(0,0,kRsmSize,kRsmSize);
    cmd.bindPipelineState(*m_pipeline); cmd.bindDescriptorSet(0,*m_set);
    auto vb=rhi::VkRHIBuffer::createNonOwning(vkD,gpu.vertexBuffer.handle(),VK_WHOLE_SIZE);
    auto ib=rhi::VkRHIBuffer::createNonOwning(vkD,gpu.indexBuffer.handle(),VK_WHOLE_SIZE);
    cmd.bindVertexBuffer(0,*vb); cmd.bindIndexBuffer(*ib,0,false);
    cmd.drawIndexedIndirectCount(indirectBuf,0,indirectBuf,0,drawCount,sizeof(VkDrawIndexedIndirectCommand));
    cmd.endRendering();

    // Post-render barriers: Color/Depth attachment → ShaderReadOnly
    cmd.textureBarrier(*m_positionTex, rhi::TextureLayout::ColorAttachment, rhi::TextureLayout::ShaderReadOnly);
    cmd.textureBarrier(*m_normalTex,   rhi::TextureLayout::ColorAttachment, rhi::TextureLayout::ShaderReadOnly);
    cmd.textureBarrier(*m_fluxTex,     rhi::TextureLayout::ColorAttachment, rhi::TextureLayout::ShaderReadOnly);
    cmd.textureBarrier(*m_depthTex,    rhi::TextureLayout::DepthAttachment, rhi::TextureLayout::ShaderReadOnly);
}

} // namespace somegi
