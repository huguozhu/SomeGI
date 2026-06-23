// ForwardPass RHI — Graphics PSO + IBL set=1。Mesh Shader 保留 Vk。
#include "renderer/core/forward_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_sampler.h"
#include "rhi/vulkan/vk_pso.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/base/command_buffer.h"
#include <array>
#include <cstring>

namespace somegi {
namespace { struct PC { glm::mat4 model; int materialIndex; int p0,p1,p2; };
static_assert(sizeof(PC)==80); }

static VkDescriptorSet VkSet(auto& p) { return (VkDescriptorSet)(uintptr_t)p->nativeHandle(); }
static VkPipelineLayout VkLay(auto& p) { return static_cast<rhi::VkRHIPipelineState&>(*p).layout(); }
static rhi::Format toRF(VkFormat f) {
    switch (f) { case VK_FORMAT_R16G16B16A16_SFLOAT: return rhi::Format::R16G16B16A16_SFLOAT;
                 case VK_FORMAT_D32_SFLOAT: return rhi::Format::D32_SFLOAT; default: return rhi::Format::Unknown; }
}

ForwardPass::~ForwardPass() = default;

void ForwardPass::init(Device& d, rhi::RHIDevice& rhiDevice, VkFormat colorFmt, VkFormat depthFmt, uint32_t maxTextures) {
    m_rhiDevice=&rhiDevice; m_colorFmt=colorFmt; m_depthFmt=depthFmt; m_maxTextures=maxTextures;
    auto& vkD=static_cast<rhi::VkRHIDevice&>(rhiDevice);
    using DS=rhi::DescriptorType; using SS=rhi::ShaderStage;
    auto VSFS=static_cast<SS>(static_cast<uint32_t>(SS::Vertex)|static_cast<uint32_t>(SS::Fragment));

    // set=0 (11 bindings, PARTIALLY_BOUND on 3)
    {
        rhi::DescSetLayoutDesc ld; ld.debugName="Forward";
        ld.bindings={{0,DS::UniformBuffer,1,VSFS},{1,DS::StorageBuffer,1,SS::Fragment},{2,DS::Sampler,1,SS::Fragment}};
        ld.bindings.push_back({3,DS::SampledImage,maxTextures,SS::Fragment,true}); // PARTIALLY_BOUND
        for(uint32_t i=4;i<10;++i) ld.bindings.push_back({i,DS::StorageBuffer,1,SS::Fragment}); // NDGI weights
        ld.bindings.push_back({10,DS::StorageBuffer,1,SS::Vertex}); // DrawData
        m_setLayout=rhiDevice.createDescriptorSetLayout(ld);
        m_set=rhiDevice.createDescriptorSet(*m_setLayout);
    }

    m_frameUbo=Buffer(d,sizeof(FrameUBO),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_rhiFrameUbo=rhi::VkRHIBuffer::createNonOwning(vkD,m_frameUbo.handle(),sizeof(FrameUBO));

    // 占位 STORAGE buffer（NDGI weights 初始值）
    m_dummySBuf=Buffer(d,4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_rhiDummySBuf=rhi::VkRHIBuffer::createNonOwning(vkD,m_dummySBuf.handle(),4);

    // IBL params UBO (set=1 binding 4, gIblParams)
    { struct IblParams { float intensity; float _pad[3]; };
      m_iblParamsUbo=Buffer(d,sizeof(IblParams),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      IblParams p{1.0f}; std::memcpy(m_iblParamsUbo.mapped(),&p,sizeof(p)); }
    m_rhiIblParamsUbo=rhi::VkRHIBuffer::createNonOwning(vkD,m_iblParamsUbo.handle(),sizeof(float)*4);

    // IBL set=1
    {
        rhi::DescSetLayoutDesc ld; ld.debugName="ForwardIBL";
        ld.bindings={{0,DS::SampledImage,1,VSFS},{1,DS::SampledImage,1,VSFS},{2,DS::SampledImage,1,VSFS},{3,DS::Sampler,1,VSFS},{4,DS::UniformBuffer,1,VSFS}};
        m_iblDsl=rhiDevice.createDescriptorSetLayout(ld);
    }

    // Graphics PSO (vertex+fragment from forward_ibl.spv)
    {
        auto spv=shaderDir()/"forward"/"forward_ibl.spv";
        rhi::ShaderDesc vsd,fsd; vsd.stage=SS::Vertex; vsd.entryPoint="vs_main"; fsd.stage=SS::Fragment; fsd.entryPoint="ps_main";
        auto vs=rhi::VkRHIShader::createFromFile(vkD,vsd,spv); auto fs=rhi::VkRHIShader::createFromFile(vkD,fsd,spv);
        rhi::GraphicsPSODesc pd; pd.debugName="Forward"; pd.vertexShader=vs.get(); pd.fragmentShader=fs.get();
        pd.vertexInput.bindings={{0,sizeof(Vertex),false}};
        pd.vertexInput.attributes={{0,rhi::VertexFormat::Float3,offsetof(Vertex,position),0},{1,rhi::VertexFormat::Float3,offsetof(Vertex,normal),0},{2,rhi::VertexFormat::Float2,offsetof(Vertex,uv0),0},{3,rhi::VertexFormat::Float4,offsetof(Vertex,tangent),0}};
        pd.topology=rhi::PrimitiveTopology::TriangleList;
        pd.rasterization.cull=rhi::CullMode::Back; pd.rasterization.frontCCW=true;
        pd.depthStencil.depthTest=true; pd.depthStencil.depthWrite=true; pd.depthStencil.depthCompare=rhi::CompareFunc::LessEqual;
        pd.renderTargets.colorFormats={toRF(colorFmt)}; pd.renderTargets.depthFormat=toRF(depthFmt);
        pd.descriptorSetLayouts={m_setLayout.get(),m_iblDsl.get()};
        pd.pushConstants={{SS::Vertex|SS::Fragment,0,sizeof(PC)}};
        m_pipeline=rhiDevice.createGraphicsPSO(pd);
    }

    // Mesh Shader PSO (RHI)
    if(d.features().meshShader) {
        using DS=rhi::DescriptorType; using SS=rhi::ShaderStage;
        auto kMS = static_cast<SS>(static_cast<uint32_t>(SS::Mesh)|static_cast<uint32_t>(SS::Task));

        // Mesh Shader 描述符集布局
        {
            rhi::DescSetLayoutDesc ld; ld.debugName="ForwardMesh";
            ld.bindings={{0,DS::StorageBuffer,1,kMS},{1,DS::StorageBuffer,1,kMS}};
            for(uint32_t i=0;i<4;++i) ld.bindings.push_back({2u+i,DS::SampledImage,1,kMS});
            ld.bindings.push_back({6,DS::StorageBuffer,1,kMS});
            ld.bindings.push_back({7,DS::StorageBuffer,1,kMS});
            ld.bindings.push_back({8,DS::SampledImage,maxTextures,kMS,true}); // PARTIALLY_BOUND
            for(uint32_t i=9;i<12;++i) ld.bindings.push_back({i,DS::SampledImage,1,kMS});
            m_meshSetLayoutRhi = rhiDevice.createDescriptorSetLayout(ld);
            m_meshSetRhi = rhiDevice.createDescriptorSet(*m_meshSetLayoutRhi);
        }

        m_meshGroupBuf = Buffer(d, m_maxTextures*sizeof(uint32_t)*2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        // Mesh Shader Graphics PSO
        auto sd = shaderDir();
        rhi::ShaderDesc msDesc, fsDesc;
        msDesc.stage = SS::Mesh; msDesc.entryPoint = "main";
        fsDesc.stage = SS::Fragment; fsDesc.entryPoint = "main";
        auto meshShader = rhi::VkRHIShader::createFromFile(vkD, msDesc, sd/"forward"/"forward_mesh_no_task_mesh.spv");
        auto fragShader = rhi::VkRHIShader::createFromFile(vkD, fsDesc, sd/"forward"/"forward_mesh_no_task_frag.spv");

        rhi::GraphicsPSODesc pd;
        pd.debugName = "ForwardMesh";
        pd.meshShader = meshShader.get();
        pd.fragmentShader = fragShader.get();
        pd.topology = rhi::PrimitiveTopology::TriangleList;
        pd.rasterization.cull = rhi::CullMode::Back; pd.rasterization.frontCCW = false;
        pd.depthStencil.depthTest = true; pd.depthStencil.depthWrite = true; pd.depthStencil.depthCompare = rhi::CompareFunc::LessEqual;
        pd.renderTargets.colorFormats = {toRF(colorFmt)};
        pd.renderTargets.depthFormat = toRF(depthFmt);
        pd.descriptorSetLayouts = {m_meshSetLayoutRhi.get()};
        pd.pushConstants = {{static_cast<SS>(static_cast<uint32_t>(SS::Mesh)|static_cast<uint32_t>(SS::Task)), 0, sizeof(PC)}};
        m_meshPipelineRhi = rhiDevice.createGraphicsPSO(pd);
    }
}

void ForwardPass::destroy() {
    m_set.reset(); m_setLayout.reset(); m_iblSet.reset(); m_iblDsl.reset(); m_pipeline.reset();
    m_meshSetRhi.reset(); m_meshSetLayoutRhi.reset(); m_meshPipelineRhi.reset();
    m_frameUbo.reset(); m_iblParamsUbo.reset(); m_dummySBuf.reset();
    m_rhiFrameUbo.reset(); m_rhiIblParamsUbo.reset(); m_rhiDummySBuf.reset();
    m_cullUbo.reset(); m_meshGroupBuf.reset();
    m_rhiDevice=nullptr;
}

void ForwardPass::bindIblResources(Device& d, const IblResources& ibl) {
    auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_iblSet=m_rhiDevice->createDescriptorSet(*m_iblDsl);
    auto diff=rhi::VkRHITextureView::createNonOwning(vkD,ibl.diffuseCube.view());
    auto spec=rhi::VkRHITextureView::createNonOwning(vkD,ibl.specularCube.view());
    auto lut=rhi::VkRHITextureView::createNonOwning(vkD,ibl.brdfLut.view());
    auto ibs=rhi::VkRHISampler::createNonOwning(vkD,ibl.linear);
    auto ibp=m_rhiIblParamsUbo.get();
    m_iblSet->write({{0,rhi::DescriptorType::SampledImage,diff.get()},{1,rhi::DescriptorType::SampledImage,spec.get()},{2,rhi::DescriptorType::SampledImage,lut.get()},{3,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,ibs.get()},{4,rhi::DescriptorType::UniformBuffer,nullptr,ibp}});
}

void ForwardPass::bindScene(Device&, const SceneGpu& gpu, uint32_t tc) {
    auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto mat=gpu.rhiMaterialBuffer();
    auto ibs=rhi::VkRHISampler::createNonOwning(vkD,gpu.linearSampler);
    auto ubo=m_rhiFrameUbo.get();
    m_texViews.clear(); m_texViewPtrs.clear(); m_texViews.reserve(m_maxTextures); m_texViewPtrs.reserve(m_maxTextures);
    for(uint32_t i=0;i<m_maxTextures;++i){ VkImageView v=(i<tc&&i<gpu.images.size())?gpu.images[i].view():gpu.whiteTex.view(); m_texViews.push_back(rhi::VkRHITextureView::createNonOwning(vkD,v)); m_texViewPtrs.push_back(m_texViews.back().get()); }
    auto dumB=m_rhiDummySBuf.get();
    m_set->write({{0,rhi::DescriptorType::UniformBuffer,nullptr,ubo},{1,rhi::DescriptorType::StorageBuffer,nullptr,mat},{2,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,ibs.get()},{3,rhi::DescriptorType::SampledImage,nullptr,nullptr,0,0,nullptr,nullptr,m_maxTextures,m_texViewPtrs.data()},{4,rhi::DescriptorType::StorageBuffer,nullptr,dumB},{5,rhi::DescriptorType::StorageBuffer,nullptr,dumB},{6,rhi::DescriptorType::StorageBuffer,nullptr,dumB},{7,rhi::DescriptorType::StorageBuffer,nullptr,dumB},{8,rhi::DescriptorType::StorageBuffer,nullptr,dumB},{9,rhi::DescriptorType::StorageBuffer,nullptr,dumB}});
}

void ForwardPass::bindDrawData(Device&, VkBuffer db) {
    auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto dd=rhi::VkRHIBuffer::createNonOwning(vkD,db,VK_WHOLE_SIZE);
    m_set->write({{10,rhi::DescriptorType::StorageBuffer,nullptr,dd.get()}});
}

void ForwardPass::updateFrame(const FrameUBO& ubo) { std::memcpy(m_frameUbo.mapped(),&ubo,sizeof(ubo)); }

void ForwardPass::setNdgiWeights(Device&, VkBuffer w1,VkBuffer b1,VkBuffer w2,VkBuffer b2,VkBuffer w3,VkBuffer b3) {
    auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto W1=rhi::VkRHIBuffer::createNonOwning(vkD,w1,VK_WHOLE_SIZE),B1=rhi::VkRHIBuffer::createNonOwning(vkD,b1,VK_WHOLE_SIZE);
    auto W2=rhi::VkRHIBuffer::createNonOwning(vkD,w2,VK_WHOLE_SIZE),B2=rhi::VkRHIBuffer::createNonOwning(vkD,b2,VK_WHOLE_SIZE);
    auto W3=rhi::VkRHIBuffer::createNonOwning(vkD,w3,VK_WHOLE_SIZE),B3=rhi::VkRHIBuffer::createNonOwning(vkD,b3,VK_WHOLE_SIZE);
    m_set->write({{4,rhi::DescriptorType::StorageBuffer,nullptr,W1.get()},{5,rhi::DescriptorType::StorageBuffer,nullptr,B1.get()},{6,rhi::DescriptorType::StorageBuffer,nullptr,W2.get()},{7,rhi::DescriptorType::StorageBuffer,nullptr,B2.get()},{8,rhi::DescriptorType::StorageBuffer,nullptr,W3.get()},{9,rhi::DescriptorType::StorageBuffer,nullptr,B3.get()}});
}

// RHI 路径：前向渲染（1 颜色附件 + 深度）
void ForwardPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt, const rhi::RHIBuffer& ib, uint32_t dc, const SceneGpu& gpu) {
    if(!m_pipeline||!m_set||!m_iblSet) return;

    rhi::RenderingAttachmentInfo cAttach{};
    cAttach.view = rt.rhiHdrColorView();
    cAttach.loadOp = rhi::AttachmentLoadOp::Clear;
    cAttach.storeOp = rhi::AttachmentStoreOp::Store;

    rhi::RenderingAttachmentInfo dAttach{};
    dAttach.view = rt.rhiDepthView();
    dAttach.loadOp = rhi::AttachmentLoadOp::Clear;
    dAttach.storeOp = rhi::AttachmentStoreOp::Store;
    dAttach.clearDepth = 1.0f;

    cmd.beginRendering(&cAttach, 1, &dAttach, rt.extent.width, rt.extent.height);
    cmd.setViewport(0, 0, (float)rt.extent.width, (float)rt.extent.height);
    cmd.setScissor(0, 0, rt.extent.width, rt.extent.height);

    cmd.bindPipelineState(*m_pipeline);
    const rhi::RHIDescriptorSet* sets[2] = {m_set.get(), m_iblSet.get()};
    cmd.bindDescriptorSets(0, 2, sets);

    cmd.bindVertexBuffer(0, *gpu.rhiVertexBuffer());
    cmd.bindIndexBuffer(*gpu.rhiIndexBuffer(), 0, false);
    cmd.drawIndexedIndirectCount(ib, 0, ib, 0, dc, sizeof(VkDrawIndexedIndirectCommand));

    cmd.endRendering();
}


void ForwardPass::bindHiZViews(VkImageView m1,VkImageView m2,VkImageView m3,VkImageView m4) { (void)m1;(void)m2;(void)m3;(void)m4; }
void ForwardPass::buildMeshGroups(const std::vector<DrawEntry>&) {}
void ForwardPass::updateCullUbo(const glm::mat4&,const glm::vec4*,uint32_t,uint32_t,uint32_t,uint32_t) {}

} // namespace somegi
