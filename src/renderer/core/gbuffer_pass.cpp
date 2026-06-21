#include "core/path_util.h"
#include "core/shader.h"
// GBufferPass RHI — VS 路径 MRT Graphics PSO。Mesh Shader 保留 Vk。
#include "renderer/core/gbuffer_pass.h"
#include "core/device.h"
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

static rhi::Format toRF(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R16G16B16A16_SFLOAT: return rhi::Format::R16G16B16A16_SFLOAT;
        case VK_FORMAT_R8G8B8A8_UNORM:     return rhi::Format::R8G8B8A8_UNORM;
        case VK_FORMAT_D32_SFLOAT:          return rhi::Format::D32_SFLOAT;
        default: return rhi::Format::Unknown;
    }
}

void GBufferPass::init(Device& d, rhi::RHIDevice& rhiDevice,
                       VkFormat rt0Fmt, VkFormat rt1Fmt, VkFormat rt2Fmt,
                       VkFormat depthFmt, uint32_t maxTextures,
                       VkSampleCountFlagBits msaaSamples) {
    m_device = &d;
    m_rhiDevice = &rhiDevice;
    m_rt0Fmt = rt0Fmt; m_rt1Fmt = rt1Fmt; m_rt2Fmt = rt2Fmt;
    m_depthFmt = depthFmt;
    m_maxTextures = maxTextures;
    m_msaaSamples = msaaSamples;
    using DS = rhi::DescriptorType; using SS = rhi::ShaderStage;
    auto VSFS = static_cast<SS>(static_cast<uint32_t>(SS::Vertex) | static_cast<uint32_t>(SS::Fragment));

    // ── set=0 (RHI): UBO(0) + MaterialSSBO(1) + Sampler(2) + TextureArray(3, PARTIALLY_BOUND) + DrawDataSSBO(10) ──
    {
        rhi::DescSetLayoutDesc ld; ld.debugName = "GBuffer";
        ld.bindings = {
            {0,  DS::UniformBuffer, 1, VSFS},
            {1,  DS::StorageBuffer, 1, SS::Fragment},
            {2,  DS::Sampler,       1, SS::Fragment},
            {3,  DS::SampledImage,  maxTextures, SS::Fragment, true},  // PARTIALLY_BOUND
            {10, DS::StorageBuffer, 1, SS::Vertex},                     // DrawData
        };
        m_setLayout = rhiDevice.createDescriptorSetLayout(ld);
        m_set = rhiDevice.createDescriptorSet(*m_setLayout);
    }

    m_frameUbo = Buffer(d, sizeof(FrameUBO),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    buildPipeline();

    // ── Mesh Shader descriptor set layout（set=0，bindings 0-11）────
    //     保留完整 VK 实现（与 ForwardPass 一致）
    {
        const VkShaderStageFlags kTS = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
        std::array<VkDescriptorSetLayoutBinding, 12> mb{};
        mb[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTS, nullptr};
        mb[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTS, nullptr};  // MeshGroup 映射
        for (uint32_t i = 0; i < 4; ++i)
            mb[2+i] = {2+i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, kTS, nullptr};
        mb[6] = {6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, kTS | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        mb[7] = {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTS, nullptr};
        mb[8] = {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTS, nullptr};
        mb[9] = {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        mb[10]= {10, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        mb[11]= {11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, m_maxTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

        std::array<VkDescriptorBindingFlags, 12> mbf{};
        mbf[11] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo mbfci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        mbfci.bindingCount = (uint32_t)mbf.size(); mbfci.pBindingFlags = mbf.data();

        VkDescriptorSetLayoutCreateInfo mli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        mli.pNext = &mbfci;
        mli.bindingCount = (uint32_t)mb.size(); mli.pBindings = mb.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &mli, nullptr, &m_meshSetLayout));

        uint32_t storageCount = 5u;   // bindings 0,1,7,8,9
        uint32_t uniformCount = 1u;   // binding 6
        std::array<VkDescriptorPoolSize, 4> mps{{
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, storageCount},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformCount},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, m_maxTextures + 4},
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        }};
        VkDescriptorPoolCreateInfo mpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        mpci.maxSets = 1; mpci.poolSizeCount = (uint32_t)mps.size(); mpci.pPoolSizes = mps.data();
        VK_CHECK(vkCreateDescriptorPool(d.device(), &mpci, nullptr, &m_meshPool));

        VkDescriptorSetAllocateInfo mdai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        mdai.descriptorPool = m_meshPool; mdai.descriptorSetCount = 1; mdai.pSetLayouts = &m_meshSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &mdai, &m_meshSet));

        m_cullUbo = Buffer(d, sizeof(glm::vec4)*6 + sizeof(glm::vec2) + sizeof(uint32_t)*2,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    // buildMeshPipeline() 延迟到 setMeshShaderEnabled(true) 调用时，避免 Intel IGC crash
}

// ──────────────────────────────────────────────────────────────────
// RHI VS Pipeline
// ──────────────────────────────────────────────────────────────────
void GBufferPass::buildPipeline() {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    using SS = rhi::ShaderStage;
    using Fmt = rhi::VertexFormat;

    // 着色器
    auto spv = shaderDir() / "gbuffer" / "gbuffer.spv";
    rhi::ShaderDesc vsd, fsd;
    vsd.stage = SS::Vertex; vsd.entryPoint = "vs_main";
    fsd.stage = SS::Fragment; fsd.entryPoint = "ps_main";
    auto vs = rhi::VkRHIShader::createFromFile(vkD, vsd, spv);
    auto fs = rhi::VkRHIShader::createFromFile(vkD, fsd, spv);

    // MRT Graphics PSO
    rhi::GraphicsPSODesc pd; pd.debugName = "GBuffer";
    pd.vertexShader = vs.get(); pd.fragmentShader = fs.get();

    pd.vertexInput.bindings = {{0, sizeof(Vertex), false}};
    pd.vertexInput.attributes = {
        {0, Fmt::Float3, offsetof(Vertex,position), 0},
        {1, Fmt::Float3, offsetof(Vertex,normal), 0},
        {2, Fmt::Float4, offsetof(Vertex,tangent), 0},
        {3, Fmt::Float2, offsetof(Vertex,uv0), 0},
    };

    pd.topology = rhi::PrimitiveTopology::TriangleList;
    pd.rasterization = {rhi::FillMode::Solid, rhi::CullMode::Back, true};
    pd.depthStencil = {true, true, rhi::CompareFunc::LessEqual};
    pd.renderTargets.colorFormats = {toRF(m_rt0Fmt), toRF(m_rt1Fmt), toRF(m_rt2Fmt)};
    pd.renderTargets.depthFormat = toRF(m_depthFmt);
    pd.renderTargets.sampleCount = (uint32_t)m_msaaSamples;
    pd.descriptorSetLayouts = {m_setLayout.get()};

    m_pipeline = m_rhiDevice->createGraphicsPSO(pd);
}

void GBufferPass::setMsaaSamples(VkSampleCountFlagBits samples) {
    if (m_msaaSamples == samples) return;
    m_msaaSamples = samples;
    m_pipeline.reset();
    buildPipeline();
}

void GBufferPass::destroyPipeline() {
    if (!m_device) return;
    m_pipeline.reset();
    if (m_meshPipeline)       vkDestroyPipeline(m_device->device(), m_meshPipeline, nullptr);
    if (m_meshPipelineLayout) vkDestroyPipelineLayout(m_device->device(), m_meshPipelineLayout, nullptr);
    m_meshPipeline = VK_NULL_HANDLE; m_meshPipelineLayout = VK_NULL_HANDLE;
}

// ──────────────────────────────────────────────────────────────────
// Mesh Shader Pipeline（保留完整 VK 实现）
// ──────────────────────────────────────────────────────────────────
void GBufferPass::buildMeshPipeline() {
    auto& d = *m_device;
    m_meshPipelineLayout = VK_NULL_HANDLE;

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_meshSetLayout;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_meshPipelineLayout));

    auto sd = shaderDir();
    ShaderModule meshMod(d, sd / "gbuffer/gbuffer_mesh_no_task_mesh.spv");
    ShaderModule fragMod(d, sd / "gbuffer" / "gbuffer_mesh_no_task_frag.spv");

    std::vector<VkPipelineShaderStageCreateInfo> si;
    {
        VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s.stage = VK_SHADER_STAGE_MESH_BIT_EXT; s.module = meshMod.handle(); s.pName = "main";
        si.push_back(s);
    }
    {
        VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s.stage = VK_SHADER_STAGE_FRAGMENT_BIT; s.module = fragMod.handle(); s.pName = "main";
        si.push_back(s);
    }

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 0; vi.vertexAttributeDescriptionCount = 0;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = m_msaaSamples;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    std::array<VkPipelineColorBlendAttachmentState, 3> ba{};
    for (auto& a : ba) a.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 3; cb.pAttachments = ba.data();

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyni{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyni.dynamicStateCount = 2; dyni.pDynamicStates = dyn;

    std::array<VkFormat, 3> colorFmts{m_rt0Fmt, m_rt1Fmt, m_rt2Fmt};
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 3; rci.pColorAttachmentFormats = colorFmts.data();
    rci.depthAttachmentFormat = m_depthFmt;

    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.pNext = &rci;
    gpci.stageCount = (uint32_t)si.size(); gpci.pStages = si.data();
    gpci.pVertexInputState = &vi; gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp; gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms; gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb; gpci.pDynamicState = &dyni;
    gpci.layout = m_meshPipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(d.device(), VK_NULL_HANDLE, 1, &gpci, nullptr, &m_meshPipeline));
}

void GBufferPass::setMeshShaderEnabled(bool v) {
    if (v && m_meshPipeline == VK_NULL_HANDLE) buildMeshPipeline();
    m_useMeshShader = v;
}

void GBufferPass::bindHiZViews(VkImageView mip1, VkImageView mip2, VkImageView mip3, VkImageView mip4) {
    auto hiZInfo = [](VkImageView v) {
        VkDescriptorImageInfo i{};
        i.imageView = v; i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; return i;
    };
    VkDescriptorImageInfo hz1 = hiZInfo(mip1), hz2 = hiZInfo(mip2), hz3 = hiZInfo(mip3), hz4 = hiZInfo(mip4);
    std::array<VkWriteDescriptorSet, 4> w{};
    for (uint32_t i = 0; i < 4; ++i) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = m_meshSet; w[i].dstBinding = 2 + i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w[i].pImageInfo = (i==0?&hz1:i==1?&hz2:i==2?&hz3:&hz4);
    }
    vkUpdateDescriptorSets(m_device->device(), 4, w.data(), 0, nullptr);
}

// ──────────────────────────────────────────────────────────────────
// Destroy
// ──────────────────────────────────────────────────────────────────
void GBufferPass::destroy() {
    if (!m_device) return;
    destroyPipeline();
    auto dev = m_device->device();
    // RHI 对象自动析构（m_setLayout, m_set, m_pipeline）
    m_setLayout.reset(); m_set.reset(); m_pipeline.reset();
    // Mesh Shader 路径（VK 手动清理）
    if (m_meshPool)      vkDestroyDescriptorPool(dev, m_meshPool, nullptr);
    if (m_meshSetLayout) vkDestroyDescriptorSetLayout(dev, m_meshSetLayout, nullptr);
    m_meshPool = VK_NULL_HANDLE; m_meshSetLayout = VK_NULL_HANDLE;
    m_frameUbo.reset();
    m_cullUbo.reset();
    m_device = nullptr;
}

// ──────────────────────────────────────────────────────────────────
// bindScene（RHI DescriptorSet::write）
// ──────────────────────────────────────────────────────────────────
void GBufferPass::bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    auto uboRHI   = rhi::VkRHIBuffer::createNonOwning(vkD, m_frameUbo.handle(), VK_WHOLE_SIZE);
    auto matRHI   = rhi::VkRHIBuffer::createNonOwning(vkD, gpu.materialBuffer.handle(), VK_WHOLE_SIZE);
    auto sampRHI  = rhi::VkRHISampler::createNonOwning(vkD, gpu.linearSampler);

    // 纹理数组 views
    std::vector<std::unique_ptr<rhi::RHITextureView>> texViews;
    std::vector<const rhi::RHITextureView*> texViewPtrs;
    texViews.reserve(m_maxTextures); texViewPtrs.reserve(m_maxTextures);
    for (uint32_t i = 0; i < m_maxTextures; ++i) {
        VkImageView v = (i < textureCount && i < gpu.images.size())
            ? gpu.images[i].view() : gpu.whiteTex.view();
        texViews.push_back(rhi::VkRHITextureView::createNonOwning(vkD, v));
        texViewPtrs.push_back(texViews.back().get());
    }

    m_set->write({
        {0,  rhi::DescriptorType::UniformBuffer, nullptr, uboRHI.get()},
        {1,  rhi::DescriptorType::StorageBuffer, nullptr, matRHI.get()},
        {2,  rhi::DescriptorType::Sampler,       nullptr, nullptr, 0, 0, sampRHI.get()},
        {3,  rhi::DescriptorType::SampledImage,  nullptr, nullptr, 0, 0, nullptr, nullptr, m_maxTextures, texViewPtrs.data()},
        // binding 10 (DrawData) 在 bindDrawData 中写入
    });

    // ── Mesh Shader 的 set=0 描述符（仅在启用 mesh 时写入，避免 Intel 驱动崩溃）──
    if (m_useMeshShader) {
        VkDescriptorBufferInfo vbInfo{gpu.vertexBuffer.handle(), 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo ibInfo{gpu.indexBuffer.handle(), 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo samplerInfo{};
        samplerInfo.sampler = gpu.linearSampler;

        std::vector<VkDescriptorImageInfo> imgs;
        imgs.reserve(m_maxTextures);
        for (uint32_t i = 0; i < m_maxTextures; ++i) {
            VkDescriptorImageInfo ii{};
            ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ii.imageView = (i < textureCount && i < gpu.images.size())
                ? gpu.images[i].view() : gpu.whiteTex.view();
            imgs.push_back(ii);
        }

        VkDescriptorBufferInfo uboInfo{m_frameUbo.handle(), 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo matInfo{gpu.materialBuffer.handle(), 0, VK_WHOLE_SIZE};

        std::array<VkWriteDescriptorSet, 6> mw{};  // bindings 6-11
        mw[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        mw[0].dstSet=m_meshSet;mw[0].dstBinding=6;mw[0].descriptorCount=1;
        mw[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;mw[0].pBufferInfo=&uboInfo;
        mw[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        mw[1].dstSet=m_meshSet;mw[1].dstBinding=7;mw[1].descriptorCount=1;
        mw[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;mw[1].pBufferInfo=&vbInfo;
        mw[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        mw[2].dstSet=m_meshSet;mw[2].dstBinding=8;mw[2].descriptorCount=1;
        mw[2].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;mw[2].pBufferInfo=&ibInfo;
        mw[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        mw[3].dstSet=m_meshSet;mw[3].dstBinding=9;mw[3].descriptorCount=1;
        mw[3].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;mw[3].pBufferInfo=&matInfo;
        mw[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        mw[4].dstSet=m_meshSet;mw[4].dstBinding=10;mw[4].descriptorCount=1;
        mw[4].descriptorType=VK_DESCRIPTOR_TYPE_SAMPLER;mw[4].pImageInfo=&samplerInfo;
        mw[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        mw[5].dstSet=m_meshSet;mw[5].dstBinding=11;mw[5].descriptorCount=m_maxTextures;
        mw[5].descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;mw[5].pImageInfo=imgs.data();
        // binding 0 (DrawData) 在 bindDrawData 中写入
        vkUpdateDescriptorSets(d.device(), (uint32_t)mw.size(), mw.data(), 0, nullptr);
    }
}

// ──────────────────────────────────────────────────────────────────
// bindDrawData（RHI DescriptorSet::write）
// ──────────────────────────────────────────────────────────────────
void GBufferPass::bindDrawData(Device& d, VkBuffer drawDataBuf) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    // ── VS 路径 binding 10 ──
    auto ddRHI = rhi::VkRHIBuffer::createNonOwning(vkD, drawDataBuf, VK_WHOLE_SIZE);
    m_set->write({{10, rhi::DescriptorType::StorageBuffer, nullptr, ddRHI.get()}});

    // ── Mesh 路径 binding 0 ──（同一 buffer，不同 binding）
    if (m_useMeshShader) {
        VkDescriptorBufferInfo dd{drawDataBuf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet mw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        mw.dstSet=m_meshSet;mw.dstBinding=0;mw.descriptorCount=1;
        mw.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;mw.pBufferInfo=&dd;
        vkUpdateDescriptorSets(d.device(), 1, &mw, 0, nullptr);
    }
}

// ──────────────────────────────────────────────────────────────────
// buildMeshGroups / updateCullUbo / updateFrame（不变）
// ──────────────────────────────────────────────────────────────────
void GBufferPass::buildMeshGroups(const std::vector<DrawEntry>& entries) {
    struct MeshGroup { uint32_t drawIndex; uint32_t triOffset; };
    constexpr uint32_t kShaderMaxTris = 85;
    uint32_t kMaxTris = kShaderMaxTris;
    if (m_device) {
        uint32_t gpuLimit = m_device->features().maxMeshOutputPrimitives;
        if (gpuLimit < kMaxTris) kMaxTris = gpuLimit;
    }
    std::vector<MeshGroup> groups;
    for (uint32_t d = 0; d < (uint32_t)entries.size(); ++d) {
        uint32_t totalTris = entries[d].indexCount / 3;
        for (uint32_t offset = 0; offset < totalTris; offset += kMaxTris) {
            groups.push_back({d, offset});
        }
    }
    m_meshGroupCount = (uint32_t)groups.size();
    if (m_meshGroupCount == 0) return;
    m_meshGroupBuf = Buffer(*m_device, m_meshGroupCount * sizeof(MeshGroup),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(m_meshGroupBuf.mapped(), groups.data(), m_meshGroupCount * sizeof(MeshGroup));
    VkDescriptorBufferInfo gi{m_meshGroupBuf.handle(), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = m_meshSet; w.dstBinding = 1; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &gi;
    vkUpdateDescriptorSets(m_device->device(), 1, &w, 0, nullptr);
}

void GBufferPass::updateCullUbo(const glm::mat4& viewProj, const glm::vec4 frustum[6],
                                 uint32_t drawCount, uint32_t hizMaxMip,
                                 uint32_t screenW, uint32_t screenH) {
    struct {
        glm::mat4 viewProj;
        glm::vec4 frustum[6];
        glm::vec2 screenSize; uint32_t drawCount; uint32_t hizMaxMip;
    } cull{};
    cull.viewProj = viewProj;
    for (int i = 0; i < 6; ++i) cull.frustum[i] = frustum[i];
    cull.screenSize = glm::vec2((float)screenW, (float)screenH);
    cull.drawCount = drawCount;
    cull.hizMaxMip = hizMaxMip;
    std::memcpy(m_cullUbo.mapped(), &cull, sizeof(cull));
}

void GBufferPass::updateFrame(const FrameUBO& ubo) {
    std::memcpy(m_frameUbo.mapped(), &ubo, sizeof(FrameUBO));
}

// ──────────────────────────────────────────────────────────────────
// record（VK 原生 + RHI 重载委托）
// 注：RHI 路径通过 VkRHICommandBuffer::vkCmd() 获取原生句柄后使用 VK API，
// RHI 路径：GBuffer MRT 渲染（3 颜色附件 + 深度，可选 MSAA resolve）
void GBufferPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt,
                         const rhi::RHIBuffer& indirectBuf, uint32_t drawCount, const SceneGpu& gpu) {
    if (drawCount == 0) return;
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    bool useMsaa = m_msaaSamples != VK_SAMPLE_COUNT_1_BIT;

    // 非拥有型 texture view 包装
    auto cv0 = rhi::VkRHITextureView::createNonOwning(vkDev,
        useMsaa ? rt.gAlbedoMetalMs.view() : rt.gAlbedoMetal.view());
    auto cv1 = rhi::VkRHITextureView::createNonOwning(vkDev,
        useMsaa ? rt.gNormalRoughMs.view() : rt.gNormalRough.view());
    auto cv2 = rhi::VkRHITextureView::createNonOwning(vkDev,
        useMsaa ? rt.gEmissiveAOMs.view() : rt.gEmissiveAO.view());
    auto dv = rhi::VkRHITextureView::createNonOwning(vkDev,
        useMsaa ? rt.depthMs.view() : rt.depth.view());

    // MSAA resolve 视图
    auto res0 = useMsaa ? rt.rhiGAlbedoMetalView() : nullptr;
    auto res1 = useMsaa ? rt.rhiGNormalRoughView() : nullptr;
    auto res2 = useMsaa ? rt.rhiGEmissiveAOView() : nullptr;
    auto dres = useMsaa ? rt.rhiDepthView() : nullptr;

    // 颜色附件
    rhi::RenderingAttachmentInfo cAttach[3]{};
    for (int i = 0; i < 3; ++i) {
        cAttach[i].view = (i == 0 ? cv0 : i == 1 ? cv1 : cv2).get();
        cAttach[i].loadOp = rhi::AttachmentLoadOp::Clear;
        cAttach[i].storeOp = rhi::AttachmentStoreOp::Store;
    }
    if (useMsaa) {
        cAttach[0].resolveView = res0; cAttach[0].resolveMode = rhi::ResolveMode::Average;
        cAttach[1].resolveView = res1; cAttach[1].resolveMode = rhi::ResolveMode::Average;
        cAttach[2].resolveView = res2; cAttach[2].resolveMode = rhi::ResolveMode::Average;
    }

    // 深度附件
    rhi::RenderingAttachmentInfo dAttach{};
    dAttach.view = dv.get();
    dAttach.loadOp = rhi::AttachmentLoadOp::Clear;
    dAttach.storeOp = rhi::AttachmentStoreOp::Store;
    dAttach.clearDepth = 1.0f;
    if (useMsaa) {
        dAttach.resolveView = dres;
        dAttach.resolveMode = rhi::ResolveMode::Min;
    }

    cmd.beginRendering(cAttach, 3, &dAttach, rt.extent.width, rt.extent.height);
    cmd.setViewport(0, 0, (float)rt.extent.width, (float)rt.extent.height);
    cmd.setScissor(0, 0, rt.extent.width, rt.extent.height);

    // Mesh shader 路径
    if (m_useMeshShader && m_meshPipeline != VK_NULL_HANDLE) {
        auto vkCmd = static_cast<rhi::VkRHICommandBuffer&>(cmd).vkCmd();
        vkCmdBindPipeline(vkCmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_meshPipeline);
        vkCmdBindDescriptorSets(vkCmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_meshPipelineLayout, 0, 1, &m_meshSet, 0, nullptr);
        m_device->vkCmdDrawMeshTasksEXT(vkCmd, m_meshGroupCount, 1, 1);
    } else {
        // 顶点/索引缓冲路径（RHI）
        cmd.bindPipelineState(*m_pipeline);
        cmd.bindDescriptorSet(0, *m_set);
        auto vb = rhi::VkRHIBuffer::createNonOwning(vkDev, gpu.vertexBuffer.handle(), VK_WHOLE_SIZE);
        auto ib = rhi::VkRHIBuffer::createNonOwning(vkDev, gpu.indexBuffer.handle(), VK_WHOLE_SIZE);
        cmd.bindVertexBuffer(0, *vb);
        cmd.bindIndexBuffer(*ib, 0, false);
        cmd.drawIndexedIndirectCount(indirectBuf, 0, indirectBuf, 0,
                                      drawCount, sizeof(VkDrawIndexedIndirectCommand));
    }
    cmd.endRendering();
}


} // namespace somegi
