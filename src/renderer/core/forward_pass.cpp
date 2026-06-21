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
    m_device=&d; m_rhiDevice=&rhiDevice; m_colorFmt=colorFmt; m_depthFmt=depthFmt; m_maxTextures=maxTextures;
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

    // 占位 STORAGE buffer（NDGI weights 初始值）
    m_dummySBuf=Buffer(d,4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // IBL params UBO (set=1 binding 4, gIblParams)
    { struct IblParams { float intensity; float _pad[3]; };
      m_iblParamsUbo=Buffer(d,sizeof(IblParams),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      IblParams p{1.0f}; std::memcpy(m_iblParamsUbo.mapped(),&p,sizeof(p)); }

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

    // Mesh Shader (Vk, 保留原实现)
    if(d.features().meshShader) {
        auto sd=shaderDir();
        const VkShaderStageFlags kTS=VK_SHADER_STAGE_TASK_BIT_EXT|VK_SHADER_STAGE_MESH_BIT_EXT;
        std::array<VkDescriptorSetLayoutBinding,12> mb{};
        mb[0]={0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,kTS,nullptr}; mb[1]={1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,kTS,nullptr};
        for(uint32_t i=0;i<4;++i) mb[2+i]={2u+i,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,kTS,nullptr};
        mb[6]={6,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,kTS,nullptr}; mb[7]={7,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,kTS,nullptr};
        mb[8]={8,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,maxTextures,kTS,nullptr};
        for(uint32_t i=9;i<12;++i) mb[i]={i,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,kTS,nullptr};
        std::array<VkDescriptorBindingFlags,12> mbf{}; mbf[8]=VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo mbfci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        mbfci.bindingCount=(uint32_t)mbf.size(); mbfci.pBindingFlags=mbf.data();
        VkDescriptorSetLayoutCreateInfo mli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        mli.pNext=&mbfci; mli.bindingCount=(uint32_t)mb.size(); mli.pBindings=mb.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(),&mli,nullptr,&m_meshSetLayout));
        std::array<VkDescriptorPoolSize,2> mps{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,10},{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,maxTextures+4}}};
        VkDescriptorPoolCreateInfo mpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        mpci.maxSets=1; mpci.poolSizeCount=(uint32_t)mps.size(); mpci.pPoolSizes=mps.data();
        VK_CHECK(vkCreateDescriptorPool(d.device(),&mpci,nullptr,&m_meshPool));
        VkDescriptorSetAllocateInfo mdai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        mdai.descriptorPool=m_meshPool; mdai.descriptorSetCount=1; mdai.pSetLayouts=&m_meshSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(d.device(),&mdai,&m_meshSet));
        VkPushConstantRange mpc{VK_SHADER_STAGE_TASK_BIT_EXT|VK_SHADER_STAGE_MESH_BIT_EXT,0,sizeof(PC)};
        VkPipelineLayoutCreateInfo mpli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        mpli.setLayoutCount=1; mpli.pSetLayouts=&m_meshSetLayout; mpli.pushConstantRangeCount=1; mpli.pPushConstantRanges=&mpc;
        VK_CHECK(vkCreatePipelineLayout(d.device(),&mpli,nullptr,&m_meshPipelineLayout));
        m_meshGroupBuf=Buffer(d,m_maxTextures*sizeof(uint32_t)*2,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        ShaderModule meshMod(d,sd/"forward"/"forward_mesh_no_task_mesh.spv"),fragMod(d,sd/"forward"/"forward_mesh_no_task_frag.spv");
        std::vector<VkPipelineShaderStageCreateInfo> mstages;
        mstages.push_back({VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_MESH_BIT_EXT,meshMod.handle(),"main",nullptr});
        mstages.push_back({VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,fragMod.handle(),"main",nullptr});
        VkGraphicsPipelineCreateInfo mgci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        mgci.stageCount=(uint32_t)mstages.size(); mgci.pStages=mstages.data();
        VkPipelineVertexInputStateCreateInfo mvi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo mia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        mia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineRasterizationStateCreateInfo mrs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        mrs.cullMode=VK_CULL_MODE_BACK_BIT; mrs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; mrs.lineWidth=1.f;
        VkPipelineMultisampleStateCreateInfo mms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; mms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo mds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        mds.depthTestEnable=VK_TRUE; mds.depthWriteEnable=VK_TRUE; mds.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendAttachmentState mba{}; mba.colorWriteMask=0xF;
        VkPipelineColorBlendStateCreateInfo mcb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; mcb.attachmentCount=1; mcb.pAttachments=&mba;
        VkPipelineViewportStateCreateInfo mvp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; mvp.viewportCount=1; mvp.scissorCount=1;
        VkDynamicState mdyn[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo mdyni{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; mdyni.dynamicStateCount=2; mdyni.pDynamicStates=mdyn;
        VkPipelineRenderingCreateInfo mrci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        mrci.colorAttachmentCount=1; mrci.pColorAttachmentFormats=&m_colorFmt; mrci.depthAttachmentFormat=m_depthFmt;
        mgci.pVertexInputState=&mvi; mgci.pInputAssemblyState=&mia; mgci.pRasterizationState=&mrs;
        mgci.pMultisampleState=&mms; mgci.pDepthStencilState=&mds; mgci.pColorBlendState=&mcb;
        mgci.pViewportState=&mvp; mgci.pDynamicState=&mdyni; mgci.layout=m_meshPipelineLayout; mgci.pNext=&mrci;
        VK_CHECK(vkCreateGraphicsPipelines(d.device(),VK_NULL_HANDLE,1,&mgci,nullptr,&m_meshPipeline));
    }
}

void ForwardPass::destroy() {
    m_set.reset(); m_setLayout.reset(); m_iblSet.reset(); m_iblDsl.reset(); m_pipeline.reset();
    if(m_meshPipeline) vkDestroyPipeline(m_device->device(),m_meshPipeline,nullptr);
    if(m_meshPipelineLayout) vkDestroyPipelineLayout(m_device->device(),m_meshPipelineLayout,nullptr);
    if(m_meshPool) vkDestroyDescriptorPool(m_device->device(),m_meshPool,nullptr);
    if(m_meshSetLayout) vkDestroyDescriptorSetLayout(m_device->device(),m_meshSetLayout,nullptr);
    m_frameUbo.reset(); m_iblParamsUbo.reset(); m_dummySBuf.reset(); m_cullUbo.reset(); m_meshGroupBuf.reset();
    m_device=nullptr; m_rhiDevice=nullptr;
}

void ForwardPass::bindIblResources(Device& d, const IblResources& ibl) {
    auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_iblSet=m_rhiDevice->createDescriptorSet(*m_iblDsl);
    auto diff=rhi::VkRHITextureView::createNonOwning(vkD,ibl.diffuseCube.view());
    auto spec=rhi::VkRHITextureView::createNonOwning(vkD,ibl.specularCube.view());
    auto lut=rhi::VkRHITextureView::createNonOwning(vkD,ibl.brdfLut.view());
    auto ibs=rhi::VkRHISampler::createNonOwning(vkD,ibl.linear);
    auto ibp=rhi::VkRHIBuffer::createNonOwning(vkD,m_iblParamsUbo.handle(),sizeof(float)*4);
    m_iblSet->write({{0,rhi::DescriptorType::SampledImage,diff.get()},{1,rhi::DescriptorType::SampledImage,spec.get()},{2,rhi::DescriptorType::SampledImage,lut.get()},{3,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,ibs.get()},{4,rhi::DescriptorType::UniformBuffer,nullptr,ibp.get()}});
}

void ForwardPass::bindScene(Device&, const SceneGpu& gpu, uint32_t tc) {
    auto& vkD=static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto mat=rhi::VkRHIBuffer::createNonOwning(vkD,gpu.materialBuffer.handle(),VK_WHOLE_SIZE);
    auto ibs=rhi::VkRHISampler::createNonOwning(vkD,gpu.linearSampler);
    auto ubo=rhi::VkRHIBuffer::createNonOwning(vkD,m_frameUbo.handle(),sizeof(FrameUBO));
    m_texViews.clear(); m_texViewPtrs.clear(); m_texViews.reserve(m_maxTextures); m_texViewPtrs.reserve(m_maxTextures);
    for(uint32_t i=0;i<m_maxTextures;++i){ VkImageView v=(i<tc&&i<gpu.images.size())?gpu.images[i].view():gpu.whiteTex.view(); m_texViews.push_back(rhi::VkRHITextureView::createNonOwning(vkD,v)); m_texViewPtrs.push_back(m_texViews.back().get()); }
    auto dumB=rhi::VkRHIBuffer::createNonOwning(vkD,m_dummySBuf.handle(),4);
    m_set->write({{0,rhi::DescriptorType::UniformBuffer,nullptr,ubo.get()},{1,rhi::DescriptorType::StorageBuffer,nullptr,mat.get()},{2,rhi::DescriptorType::Sampler,nullptr,nullptr,0,0,ibs.get()},{3,rhi::DescriptorType::SampledImage,nullptr,nullptr,0,0,nullptr,nullptr,m_maxTextures,m_texViewPtrs.data()},{4,rhi::DescriptorType::StorageBuffer,nullptr,dumB.get()},{5,rhi::DescriptorType::StorageBuffer,nullptr,dumB.get()},{6,rhi::DescriptorType::StorageBuffer,nullptr,dumB.get()},{7,rhi::DescriptorType::StorageBuffer,nullptr,dumB.get()},{8,rhi::DescriptorType::StorageBuffer,nullptr,dumB.get()},{9,rhi::DescriptorType::StorageBuffer,nullptr,dumB.get()}});
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
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

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

    auto vb = rhi::VkRHIBuffer::createNonOwning(vkDev, gpu.vertexBuffer.handle(), VK_WHOLE_SIZE);
    auto ibo = rhi::VkRHIBuffer::createNonOwning(vkDev, gpu.indexBuffer.handle(), VK_WHOLE_SIZE);
    cmd.bindVertexBuffer(0, *vb);
    cmd.bindIndexBuffer(*ibo, 0, false);
    cmd.drawIndexedIndirectCount(ib, 0, ib, 0, dc, sizeof(VkDrawIndexedIndirectCommand));

    cmd.endRendering();
}


void ForwardPass::bindHiZViews(VkImageView m1,VkImageView m2,VkImageView m3,VkImageView m4) { (void)m1;(void)m2;(void)m3;(void)m4; }
void ForwardPass::buildMeshGroups(const std::vector<DrawEntry>&) {}
void ForwardPass::updateCullUbo(const glm::mat4&,const glm::vec4*,uint32_t,uint32_t,uint32_t,uint32_t) {}

} // namespace somegi
