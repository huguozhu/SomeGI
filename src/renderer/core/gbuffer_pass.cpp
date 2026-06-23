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
    m_rhiFrameUbo = rhi::VkRHIBuffer::createNonOwning(
        static_cast<rhi::VkRHIDevice&>(rhiDevice), m_frameUbo.handle(), sizeof(FrameUBO));

    buildPipeline();

    // ── Mesh Shader 描述符集布局（set=0，bindings 0-11）────
    {
        using DS = rhi::ShaderStage;
        using DT = rhi::DescriptorType;
        const auto kMS = DS::Mesh | DS::Task;       // Mesh/Task 阶段可见
        const auto kFS = DS::Fragment;               // Fragment 阶段可见

        rhi::DescSetLayoutDesc md; md.debugName = "GBuffer_Mesh";
        md.bindings = {
            {0,  DT::StorageBuffer, 1, kMS},
            {1,  DT::StorageBuffer, 1, kMS},                    // MeshGroup 映射
            {2,  DT::SampledImage,  1, kMS},
            {3,  DT::SampledImage,  1, kMS},
            {4,  DT::SampledImage,  1, kMS},
            {5,  DT::SampledImage,  1, kMS},
            {6,  DT::UniformBuffer, 1, kMS | kFS},
            {7,  DT::StorageBuffer, 1, kMS},
            {8,  DT::StorageBuffer, 1, kMS},
            {9,  DT::StorageBuffer, 1, kFS},
            {10, DT::Sampler,       1, kFS},
            {11, DT::SampledImage,  m_maxTextures, kFS},
        };
        md.bindings[11].partiallyBound = true;  // 纹理数组可能部分绑定
        m_meshSetLayout = rhiDevice.createDescriptorSetLayout(md);
        m_meshSet = rhiDevice.createDescriptorSet(*m_meshSetLayout);

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
    if (!m_rhiDevice) return;
    m_pipeline.reset();
    m_meshPipeline.reset();
}

// ──────────────────────────────────────────────────────────────────
// Mesh Shader Pipeline（RHI 路径）
// ──────────────────────────────────────────────────────────────────
void GBufferPass::buildMeshPipeline() {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto sd = shaderDir();

    // 加载 Mesh + Fragment 着色器
    rhi::ShaderDesc msd, fsd;
    msd.stage = rhi::ShaderStage::Mesh; msd.entryPoint = "main";
    auto ms = rhi::VkRHIShader::createFromFile(vkD, msd,
        sd / "gbuffer" / "gbuffer_mesh_no_task_mesh.spv");
    fsd.stage = rhi::ShaderStage::Fragment; fsd.entryPoint = "main";
    auto fs = rhi::VkRHIShader::createFromFile(vkD, fsd,
        sd / "gbuffer" / "gbuffer_mesh_no_task_frag.spv");

    rhi::GraphicsPSODesc pd; pd.debugName = "GBuffer_Mesh";
    pd.meshShader = ms.get(); pd.fragmentShader = fs.get();
    pd.topology = rhi::PrimitiveTopology::TriangleList;
    pd.rasterization = {rhi::FillMode::Solid, rhi::CullMode::Back, true};
    pd.depthStencil = {true, true, rhi::CompareFunc::LessEqual};
    pd.renderTargets.colorFormats = {toRF(m_rt0Fmt), toRF(m_rt1Fmt), toRF(m_rt2Fmt)};
    pd.renderTargets.depthFormat = toRF(m_depthFmt);
    pd.renderTargets.sampleCount = (uint32_t)m_msaaSamples;
    pd.descriptorSetLayouts = {m_meshSetLayout.get()};

    m_meshPipeline = m_rhiDevice->createGraphicsPSO(pd);
}

void GBufferPass::setMeshShaderEnabled(bool v) {
    if (v && !m_meshPipeline) buildMeshPipeline();
    m_useMeshShader = v;
}

void GBufferPass::bindHiZViews(VkImageView mip1, VkImageView mip2, VkImageView mip3, VkImageView mip4) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto v1 = rhi::VkRHITextureView::createNonOwning(vkD, mip1);
    auto v2 = rhi::VkRHITextureView::createNonOwning(vkD, mip2);
    auto v3 = rhi::VkRHITextureView::createNonOwning(vkD, mip3);
    auto v4 = rhi::VkRHITextureView::createNonOwning(vkD, mip4);
    m_meshSet->write({
        {2, rhi::DescriptorType::SampledImage, v1.get()},
        {3, rhi::DescriptorType::SampledImage, v2.get()},
        {4, rhi::DescriptorType::SampledImage, v3.get()},
        {5, rhi::DescriptorType::SampledImage, v4.get()},
    });
}

// ──────────────────────────────────────────────────────────────────
// Destroy
// ──────────────────────────────────────────────────────────────────
void GBufferPass::destroy() {
    if (!m_rhiDevice) return;
    destroyPipeline();
    // RHI 对象自动析构
    m_setLayout.reset(); m_set.reset(); m_pipeline.reset();
    m_meshSetLayout.reset(); m_meshSet.reset(); m_meshPipeline.reset();
    m_rhiFrameUbo.reset();
    m_frameUbo.reset();
    m_cullUbo.reset();
    m_rhiDevice = nullptr;
}

// ──────────────────────────────────────────────────────────────────
// bindScene（RHI DescriptorSet::write）
// ──────────────────────────────────────────────────────────────────
void GBufferPass::bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    auto uboRHI   = m_rhiFrameUbo.get();
    auto matRHI   = gpu.rhiMaterialBuffer();
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
        {0,  rhi::DescriptorType::UniformBuffer, nullptr, uboRHI},
        {1,  rhi::DescriptorType::StorageBuffer, nullptr, matRHI},
        {2,  rhi::DescriptorType::Sampler,       nullptr, nullptr, 0, 0, sampRHI.get()},
        {3,  rhi::DescriptorType::SampledImage,  nullptr, nullptr, 0, 0, nullptr, nullptr, m_maxTextures, texViewPtrs.data()},
        // binding 10 (DrawData) 在 bindDrawData 中写入
    });

    // ── Mesh Shader 的 set=0 描述符（RHI write）──
    if (m_useMeshShader) {
        auto meshGroupRHI = rhi::VkRHIBuffer::createNonOwning(vkD, m_meshGroupBuf.handle(), m_meshGroupBuf.size());
        auto vertRHI = rhi::VkRHIBuffer::createNonOwning(vkD, gpu.vertexBuffer.handle(), VK_WHOLE_SIZE);
        auto idxRHI  = rhi::VkRHIBuffer::createNonOwning(vkD, gpu.indexBuffer.handle(), VK_WHOLE_SIZE);

        std::vector<rhi::DescriptorWrite> mw;
        mw.push_back({1,  rhi::DescriptorType::StorageBuffer, nullptr, meshGroupRHI.get()});
        mw.push_back({6,  rhi::DescriptorType::UniformBuffer,  nullptr, m_rhiFrameUbo.get()});
        mw.push_back({7,  rhi::DescriptorType::StorageBuffer,  nullptr, vertRHI.get()});
        mw.push_back({8,  rhi::DescriptorType::StorageBuffer,  nullptr, idxRHI.get()});
        mw.push_back({9,  rhi::DescriptorType::StorageBuffer,  nullptr, matRHI});
        // binding 10: Sampler
        { rhi::DescriptorWrite w; w.binding=10; w.type=rhi::DescriptorType::Sampler; w.sampler=sampRHI.get(); mw.push_back(w); }
        // binding 11: Texture array
        { rhi::DescriptorWrite w; w.binding=11; w.type=rhi::DescriptorType::SampledImage;
          w.textureArrayCount=(uint32_t)texViewPtrs.size(); w.textureViewArray=texViewPtrs.data(); mw.push_back(w); }
        m_meshSet->write(mw);
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
        m_meshSet->write({{0, rhi::DescriptorType::StorageBuffer, nullptr, ddRHI.get()}});
    }
}

// ──────────────────────────────────────────────────────────────────
// buildMeshGroups / updateCullUbo / updateFrame（不变）
// ──────────────────────────────────────────────────────────────────
void GBufferPass::buildMeshGroups(const std::vector<DrawEntry>& entries) {
    struct MeshGroup { uint32_t drawIndex; uint32_t triOffset; };
    constexpr uint32_t kShaderMaxTris = 85;
    uint32_t kMaxTris = kShaderMaxTris;
    if (m_rhiDevice) {
        uint32_t gpuLimit = m_rhiDevice->limits().maxMeshOutputPrimitives;
        if (gpuLimit > 0 && gpuLimit < kMaxTris) kMaxTris = gpuLimit;
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
    m_meshGroupBuf = Buffer(static_cast<rhi::VkRHIDevice&>(*m_rhiDevice),
        m_meshGroupCount * sizeof(MeshGroup),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(m_meshGroupBuf.mapped(), groups.data(), m_meshGroupCount * sizeof(MeshGroup));
    // RHI：mesh group buffer 的 binding 1 在 bindScene 中已写入，此处更新 binding 1
    if (m_meshGroupCount > 0) {
        m_rhiMeshGroupBuf = rhi::VkRHIBuffer::createNonOwning(
            static_cast<rhi::VkRHIDevice&>(*m_rhiDevice),
            m_meshGroupBuf.handle(), m_meshGroupBuf.size());
        m_meshSet->write({{1, rhi::DescriptorType::StorageBuffer, nullptr, m_rhiMeshGroupBuf.get()}});
    }
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

    // Mesh shader 路径（RHI）
    if (m_useMeshShader && m_meshPipeline) {
        cmd.bindPipelineState(*m_meshPipeline);
        cmd.bindDescriptorSet(0, *m_meshSet);
        cmd.drawMeshTasks(m_meshGroupCount, 1, 1);
    } else {
        // 顶点/索引缓冲路径（RHI）
        cmd.bindPipelineState(*m_pipeline);
        cmd.bindDescriptorSet(0, *m_set);
        cmd.bindVertexBuffer(0, *gpu.rhiVertexBuffer());
        cmd.bindIndexBuffer(*gpu.rhiIndexBuffer(), 0, false);
        cmd.drawIndexedIndirectCount(indirectBuf, 0, indirectBuf, 0,
                                      drawCount, sizeof(VkDrawIndexedIndirectCommand));
    }
    cmd.endRendering();
}


} // namespace somegi
