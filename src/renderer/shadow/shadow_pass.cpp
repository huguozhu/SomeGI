// ShadowPass RHI — Resources managed by RHI, record VkCompat。
#include "renderer/shadow/shadow_pass.h"
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
#include "rhi/vulkan/vk_acceleration_structure.h"
#include "renderer/core/frame_ubo.h"
#include "scene/scene_gpu.h"
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <cstring>
#include <limits>

namespace somegi {

// Bridge helpers
static VkDescriptorSet VkSet(auto& p) { return (VkDescriptorSet)(uintptr_t)p->nativeHandle(); }
static VkPipelineLayout VkLay(auto& p) { return static_cast<rhi::VkRHIPipelineState&>(*p).layout(); }
static VkPipeline VkPipe(auto& p) { return (VkPipeline)(uintptr_t)p->nativeHandle(); }

namespace {

struct ShadowUbo { glm::mat4 sunViewProj; };

struct ResolvePC {
    uint32_t outSizeX, outSizeY;
    float    invOutSizeX, invOutSizeY;
    float    bias;
    uint32_t _pad;
    float    invShadowMapX, invShadowMapY;
};
static_assert(sizeof(ResolvePC) == 32);

struct PCSS_PC {
    uint32_t outSizeX, outSizeY;
    float    invOutSizeX, invOutSizeY;
    float    bias;
    uint32_t _pad;
    float    invShadowMapX, invShadowMapY;
    float    lightSize;
    float    maxKernelRadius;
};
static_assert(sizeof(PCSS_PC) == 40);

void computeSunViewProj(const glm::vec3& sunDir,
                         const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                         void* mappedUbo) {
    glm::vec3 lightDir = glm::normalize(sunDir);
    glm::vec3 toSun    = -lightDir;
    glm::vec3 sceneCenter = (aabbMin + aabbMax) * 0.5f;
    glm::vec3 sceneSize   = aabbMax - aabbMin;
    float diag = glm::length(sceneSize);
    glm::vec3 sunPos = sceneCenter + toSun * diag;
    glm::vec3 up = (std::abs(toSun.y) < 0.999f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    glm::mat4 view = glm::lookAt(sunPos, sceneCenter, up);
    glm::vec3 mn{ std::numeric_limits<float>::max() };
    glm::vec3 mx{ -std::numeric_limits<float>::max() };
    for (int i = 0; i < 8; ++i) {
        glm::vec3 corner(
            (i & 1) ? aabbMax.x : aabbMin.x,
            (i & 2) ? aabbMax.y : aabbMin.y,
            (i & 4) ? aabbMax.z : aabbMin.z);
        glm::vec3 v = glm::vec3(view * glm::vec4(corner, 1.0f));
        mn = glm::min(mn, v); mx = glm::max(mx, v);
    }
    float padding = diag * 0.05f;
    mn -= glm::vec3(padding); mx += glm::vec3(padding);
    glm::mat4 proj = glm::ortho(mn.x, mx.x, mn.y, mx.y, -mx.z, -mn.z);
    proj[1][1] *= -1.0f;
    ShadowUbo ubo{}; ubo.sunViewProj = proj * view;
    std::memcpy(mappedUbo, &ubo, sizeof(ubo));
}

} // anonymous namespace

// ════════════════════════════════════════════════════════════════
// init
// ════════════════════════════════════════════════════════════════
void ShadowPass::init(Device& d, rhi::RHIDevice& rhiDevice, VkExtent2D shadowMapSize, VkExtent2D outputSize) {
    m_device = &d;
    m_rhiDevice = &rhiDevice;
    m_shadowMapSize = shadowMapSize;
    m_outputSize = outputSize;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(rhiDevice);
    using DS = rhi::DescriptorType; using SS = rhi::ShaderStage;

    // ── 1. Images (保留 VK) ──
    { ImageDesc desc{}; desc.format = VK_FORMAT_D32_SFLOAT; desc.extent = {shadowMapSize.width, shadowMapSize.height, 1}; desc.aspect = VK_IMAGE_ASPECT_DEPTH_BIT; desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; m_shadowMap = Image(d, desc); }
    { ImageDesc desc{}; desc.format = VK_FORMAT_R32G32_SFLOAT; desc.extent = {shadowMapSize.width, shadowMapSize.height, 1}; desc.aspect = VK_IMAGE_ASPECT_COLOR_BIT; desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT; m_vsmMap = Image(d, desc); }
    { ImageDesc desc{}; desc.format = VK_FORMAT_R32G32_SFLOAT; desc.extent = {shadowMapSize.width, shadowMapSize.height, 1}; desc.aspect = VK_IMAGE_ASPECT_COLOR_BIT; desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; m_vsmBlur = Image(d, desc); }
    { ImageDesc desc{}; desc.format = VK_FORMAT_R8_UNORM; desc.extent = {outputSize.width, outputSize.height, 1}; desc.aspect = VK_IMAGE_ASPECT_COLOR_BIT; desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; m_shadowMask = Image(d, desc); }
    std::printf("[shadow] shadowMask view=%p\n", (void*)m_shadowMask.view());

    // ── 2. Samplers (RHI) ──
    m_shadowSampler = rhiDevice.createSampler({rhi::Filter::Linear, rhi::Filter::Linear, rhi::SamplerMipmapMode::Linear, rhi::SamplerAddressMode::ClampToEdge, rhi::SamplerAddressMode::ClampToEdge, rhi::SamplerAddressMode::ClampToEdge, 0.f, true, rhi::CompareFunc::LessEqual});
    m_vsmSampler    = rhiDevice.createSampler({rhi::Filter::Linear, rhi::Filter::Linear, rhi::SamplerMipmapMode::Linear, rhi::SamplerAddressMode::ClampToEdge, rhi::SamplerAddressMode::ClampToEdge, rhi::SamplerAddressMode::ClampToEdge, 0.f});

    // ── 3. Resolve descriptor set (set=0, UPDATE_AFTER_BIND on bindings 1,2) ──
    {
        rhi::DescSetLayoutDesc ld; ld.debugName = "ShadowResolve";
        ld.updateAfterBind = true;
        ld.updateAfterBindBindings = {1, 2};
        ld.bindings = {
            {0, DS::UniformBuffer, 1, SS::Compute},
            {1, DS::SampledImage,  1, SS::Compute},
            {2, DS::Sampler,       1, SS::Compute},
            {3, DS::StorageImage,  1, SS::Compute},
        };
        m_setLayout = rhiDevice.createDescriptorSetLayout(ld);
        m_set = rhiDevice.createDescriptorSet(*m_setLayout);
    }

    // ── 4. SM graphics descriptor set (set=0: UBO + 3 SSBO) ──
    {
        rhi::DescSetLayoutDesc ld; ld.debugName = "ShadowSM";
        ld.bindings = {
            {0, DS::UniformBuffer, 1, SS::Vertex},
            {1, DS::StorageBuffer, 1, SS::Vertex},
            {2, DS::StorageBuffer, 1, SS::Vertex},
            {3, DS::StorageBuffer, 1, SS::Vertex},
        };
        m_smSetLayout = rhiDevice.createDescriptorSetLayout(ld);
        m_smSet = rhiDevice.createDescriptorSet(*m_smSetLayout);
    }

    // ── 5. Frame descriptor set (set=1: UBO + depth, UPDATE_AFTER_BIND) ──
    {
        rhi::DescSetLayoutDesc ld; ld.debugName = "ShadowFrame";
        ld.updateAfterBind = true;
        ld.updateAfterBindBindings = {0, 1};
        ld.bindings = {
            {0, DS::UniformBuffer, 1, SS::Compute},
            {1, DS::SampledImage,  1, SS::Compute},
        };
        m_frameSetLayout = rhiDevice.createDescriptorSetLayout(ld);
        m_frameSet = rhiDevice.createDescriptorSet(*m_frameSetLayout);
    }

    // ── 6. VSM blur descriptor set (UPDATE_AFTER_BIND) ──
    {
        rhi::DescSetLayoutDesc ld; ld.debugName = "ShadowVSMBlur";
        ld.updateAfterBind = true;
        ld.updateAfterBindBindings = {0, 1};
        ld.bindings = {
            {0, DS::SampledImage, 1, SS::Compute},
            {1, DS::StorageImage, 1, SS::Compute},
        };
        m_vsmBlurSetLayout = rhiDevice.createDescriptorSetLayout(ld);
        m_vsmBlurSet = rhiDevice.createDescriptorSet(*m_vsmBlurSetLayout);
    }

    // ── 7. RT descriptor set (仅 HW 支持时创建) ──
    bool rtOk = d.features().accelStruct && d.features().rayQuery;
    if (rtOk) {
        {
            rhi::DescSetLayoutDesc ld; ld.debugName = "ShadowRT";
            ld.updateAfterBind = true;
            ld.updateAfterBindBindings = {2};
            ld.bindings = {
                {0, DS::AccelerationStructure, 1, SS::Compute},
                {1, DS::StorageImage,          1, SS::Compute},
                {2, DS::SampledImage,           1, SS::Compute},
            };
            m_rtSetLayout = rhiDevice.createDescriptorSetLayout(ld);
            m_rtSet = rhiDevice.createDescriptorSet(*m_rtSetLayout);
        }
        // 初始写入 binding 1 = shadowMask
        auto smView = rhi::VkRHITextureView::createNonOwning(vkD, m_shadowMask.view());
        m_rtSet->write({{1, DS::StorageImage, smView.get()}});
        buildPipeline_RTHard();
    }

    // ── 8. UBO ──
    m_shadowUbo = Buffer(d, sizeof(ShadowUbo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // ── 9. 写入初始描述符 ──
    // Resolve set: bindings 0,1,2,3
    {
        auto uboRHI  = rhi::VkRHIBuffer::createNonOwning(vkD, m_shadowUbo.handle(), VK_WHOLE_SIZE);
        auto smView  = rhi::VkRHITextureView::createNonOwning(vkD, m_shadowMap.view());
        auto maskView= rhi::VkRHITextureView::createNonOwning(vkD, m_shadowMask.view());
        m_set->write({
            {0, DS::UniformBuffer, nullptr, uboRHI.get()},
            {1, DS::SampledImage,  smView.get()},
            {2, DS::Sampler,       nullptr, nullptr, 0, 0, m_shadowSampler.get()},
            {3, DS::StorageImage,  maskView.get()},
        });
    }
    // SM set: binding 0 = ShadowUbo
    {
        auto uboRHI = rhi::VkRHIBuffer::createNonOwning(vkD, m_shadowUbo.handle(), VK_WHOLE_SIZE);
        m_smSet->write({{0, DS::UniformBuffer, nullptr, uboRHI.get()}});
    }
    // VSM blur set: 初始值
    {
        auto inView  = rhi::VkRHITextureView::createNonOwning(vkD, m_vsmMap.view());
        auto outView = rhi::VkRHITextureView::createNonOwning(vkD, m_vsmBlur.view());
        m_vsmBlurSet->write({
            {0, DS::SampledImage, inView.get()},
            {1, DS::StorageImage, outView.get()},
        });
    }

    // ── 10. Pipelines ──
    buildPipeline_HardSM();
    buildPipeline_VSMGen();
    buildPipeline_VSMBlur();
    buildResolvePipeline();
}

// ════════════════════════════════════════════════════════════════
// Pipeline builders
// ════════════════════════════════════════════════════════════════

void ShadowPass::buildPipeline_HardSM() {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    using SS = rhi::ShaderStage;
    auto spv = shaderDir() / "shadow" / "shadow_hard.spv";
    rhi::ShaderDesc vsd{SS::Vertex}, fsd{SS::Fragment};
    vsd.entryPoint = "vs_main"; fsd.entryPoint = "ps_main";
    auto vs = rhi::VkRHIShader::createFromFile(vkD, vsd, spv);
    auto fs = rhi::VkRHIShader::createFromFile(vkD, fsd, spv);

    rhi::GraphicsPSODesc pd; pd.debugName = "ShadowHardSM";
    pd.vertexShader = vs.get(); pd.fragmentShader = fs.get();
    pd.topology = rhi::PrimitiveTopology::TriangleList;
    pd.rasterization = {rhi::FillMode::Solid, rhi::CullMode::Back, true, true, 4.0f, 2.0f};
    pd.depthStencil = {true, true, rhi::CompareFunc::LessEqual};
    pd.renderTargets.colorFormats = {};
    pd.renderTargets.depthFormat = rhi::Format::D32_SFLOAT;
    pd.descriptorSetLayouts = {m_smSetLayout.get()};
    m_smPipeline = m_rhiDevice->createGraphicsPSO(pd);
}

void ShadowPass::buildPipeline_VSMGen() {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    using SS = rhi::ShaderStage;
    auto spv = shaderDir() / "shadow" / "shadow_vsm.spv";
    rhi::ShaderDesc vsd2{SS::Vertex}, fsd2{SS::Fragment};
    vsd2.entryPoint = "vs_main"; fsd2.entryPoint = "ps_main";
    auto vs = rhi::VkRHIShader::createFromFile(vkD, vsd2, spv);
    auto fs = rhi::VkRHIShader::createFromFile(vkD, fsd2, spv);

    rhi::GraphicsPSODesc pd; pd.debugName = "ShadowVSMGen";
    pd.vertexShader = vs.get(); pd.fragmentShader = fs.get();
    pd.topology = rhi::PrimitiveTopology::TriangleList;
    pd.rasterization = {rhi::FillMode::Solid, rhi::CullMode::Back, true, true, 4.0f, 2.0f};
    pd.depthStencil = {true, true, rhi::CompareFunc::LessEqual};
    pd.renderTargets.colorFormats = {rhi::Format::R32G32_SFLOAT};
    pd.renderTargets.depthFormat = rhi::Format::D32_SFLOAT;
    pd.descriptorSetLayouts = {m_smSetLayout.get()};
    m_vsmGenPipeline = m_rhiDevice->createGraphicsPSO(pd);
}

void ShadowPass::buildPipeline_VSMBlur() {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto spv = shaderDir() / "shadow" / "shadow_vsm_blur.spv";
    rhi::ShaderDesc csd{rhi::ShaderStage::Compute}; csd.entryPoint = "cs_main";
    auto cs = rhi::VkRHIShader::createFromFile(vkD, csd, spv);
    rhi::ComputePSODesc pd; pd.debugName = "ShadowVSMBlur";
    pd.computeShader = cs.get();
    pd.descriptorSetLayouts = {m_vsmBlurSetLayout.get()};
    pd.pushConstants = {{rhi::ShaderStage::Compute, 0, 24}};
    m_vsmBlurPipeline = m_rhiDevice->createComputePSO(pd);
}

void ShadowPass::buildResolvePipeline() {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    using SS = rhi::ShaderStage;
    auto sd = shaderDir() / "shadow";

    auto mk = [&](const char* spvName, uint32_t pcSize) {
        rhi::ShaderDesc csd{SS::Compute}; csd.entryPoint = "cs_main";
        auto cs = rhi::VkRHIShader::createFromFile(vkD, csd, sd / spvName);
        rhi::ComputePSODesc pd; pd.computeShader = cs.get();
        pd.descriptorSetLayouts = {m_setLayout.get(), m_frameSetLayout.get()};
        if (pcSize) pd.pushConstants = {{SS::Compute, 0, pcSize}};
        return m_rhiDevice->createComputePSO(pd);
    };
    m_resolveHard = mk("shadow_hard_resolve.spv", sizeof(ResolvePC));
    m_resolvePCF  = mk("shadow_pcf_resolve.spv", sizeof(ResolvePC));
    m_resolveVSM  = mk("shadow_vsm_resolve.spv", sizeof(ResolvePC));
    m_resolvePCSS = mk("shadow_pcss_resolve.spv", sizeof(PCSS_PC));
}

void ShadowPass::buildPipeline_RTHard() {
    if (!m_rtSetLayout) return;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    using SS = rhi::ShaderStage;

    auto mkRT = [&](const char* spvName, uint32_t pcSize) {
        rhi::ShaderDesc csd{SS::Compute}; csd.entryPoint = "cs_main";
        auto cs = rhi::VkRHIShader::createFromFile(vkD, csd, shaderDir() / "shadow" / spvName);
        rhi::ComputePSODesc pd; pd.computeShader = cs.get();
        pd.descriptorSetLayouts = {m_rtSetLayout.get(), m_frameSetLayout.get()};
        if (pcSize) pd.pushConstants = {{SS::Compute, 0, pcSize}};
        return m_rhiDevice->createComputePSO(pd);
    };
    m_rtHardPipeline = mkRT("shadow_rt_hard.spv", 16);
    m_rtSoftPipeline = mkRT("shadow_rt_soft.spv", 32);
}

void ShadowPass::destroyPipelines() {
    m_smPipeline.reset(); m_vsmGenPipeline.reset(); m_vsmBlurPipeline.reset();
    m_resolveHard.reset(); m_resolvePCF.reset(); m_resolveVSM.reset(); m_resolvePCSS.reset();
    m_rtHardPipeline.reset(); m_rtSoftPipeline.reset();
}

// ════════════════════════════════════════════════════════════════
// Bind
// ════════════════════════════════════════════════════════════════

void ShadowPass::bindScene(Device&, const SceneGpu& gpu) {
    m_indexBuffer = gpu.indexBuffer.handle();
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto vb = rhi::VkRHIBuffer::createNonOwning(vkD, gpu.vertexBuffer.handle(), VK_WHOLE_SIZE);
    auto ib = rhi::VkRHIBuffer::createNonOwning(vkD, gpu.indexBuffer.handle(), VK_WHOLE_SIZE);
    auto dd = rhi::VkRHIBuffer::createNonOwning(vkD, gpu.drawDataBuffer.handle(), VK_WHOLE_SIZE);
    m_smSet->write({
        {1, rhi::DescriptorType::StorageBuffer, nullptr, vb.get()},
        {2, rhi::DescriptorType::StorageBuffer, nullptr, ib.get()},
        {3, rhi::DescriptorType::StorageBuffer, nullptr, dd.get()},
    });
}

void ShadowPass::bindFrameResources(Device&, VkBuffer frameUbo, VkImageView depthView, VkImageView normalView) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    {
        auto ub = rhi::VkRHIBuffer::createNonOwning(vkD, frameUbo, VK_WHOLE_SIZE);
        auto dv = rhi::VkRHITextureView::createNonOwning(vkD, depthView);
        m_frameSet->write({
            {0, rhi::DescriptorType::UniformBuffer, nullptr, ub.get()},
            {1, rhi::DescriptorType::SampledImage,  dv.get()},
        });
    }
    if (m_rtSet) {
        auto nv = rhi::VkRHITextureView::createNonOwning(vkD, normalView);
        m_rtSet->write({{2, rhi::DescriptorType::SampledImage, nv.get()}});
    }
}

void ShadowPass::bindTLAS(Device&, VkAccelerationStructureKHR tlas) {
    if (!m_rtSet) return;
    m_tlas = tlas;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto tlasRHI = rhi::VkRHIAccelerationStructure::createNonOwning(vkD, tlas);
    m_rtSet->write({{0, rhi::DescriptorType::AccelerationStructure, nullptr, nullptr, 0, 0, nullptr, tlasRHI.get()}});
}

// ════════════════════════════════════════════════════════════════
// Record helpers
// ════════════════════════════════════════════════════════════════

static void restoreResolveDesc(rhi::RHIDescriptorSet& set, rhi::RHIDevice& rhiDevice, VkImageView smView, VkSampler samp) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(rhiDevice);
    auto sv = rhi::VkRHITextureView::createNonOwning(vkD, smView);
    auto ss = rhi::VkRHISampler::createNonOwning(vkD, samp);
    set.write({
        {1, rhi::DescriptorType::SampledImage, sv.get()},
        {2, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, ss.get()},
    });
}

// ════════════════════════════════════════════════════════════════
// Record
// ════════════════════════════════════════════════════════════════

void ShadowPass::record(rhi::RHICommandBuffer& rhiCmd, const RenderTargets& rt,
                         VkBuffer frameUbo, const SceneGpu& sceneGpu,
                         VkBuffer indirectBuf, uint32_t drawCount, uint32_t frameIndex) {
    auto vkCmd = static_cast<rhi::VkRHICommandBuffer&>(rhiCmd).vkCmd();
    record(vkCmd, rt, frameUbo, sceneGpu, indirectBuf, drawCount, frameIndex);}

void ShadowPass::record(VkCommandBuffer cmd, const RenderTargets& rt,
                         VkBuffer frameUbo, const SceneGpu& sceneGpu,
                         VkBuffer indirectBuf, uint32_t drawCount, uint32_t frameIndex) {
    m_currentFrameIndex = frameIndex;
    switch (m_method) {
    case ShadowMethod::None:          recordNone(cmd); break;
    case ShadowMethod::HardShadowMap: recordHardSM(cmd, rt, frameUbo, sceneGpu, indirectBuf, drawCount); break;
    case ShadowMethod::PCF:           recordPCF(cmd, rt, frameUbo, sceneGpu, indirectBuf, drawCount); break;
    case ShadowMethod::PCSS:          recordPCSS(cmd, rt, frameUbo, sceneGpu, indirectBuf, drawCount); break;
    case ShadowMethod::VSM:           recordVSM(cmd, rt, frameUbo, sceneGpu, indirectBuf, drawCount); break;
    case ShadowMethod::RTHard:        recordRTHard(cmd); break;
    case ShadowMethod::RTSoft:        recordRTSoft(cmd); break;
    default:                          recordNone(cmd); break;
    }
}

void ShadowPass::recordNone(VkCommandBuffer cmd) {
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_CLEAR_BIT, 0);
    VkClearColorValue clearVal{};
    clearVal.float32[0] = 1.0f; clearVal.float32[1] = 1.0f; clearVal.float32[2] = 1.0f; clearVal.float32[3] = 1.0f;
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_shadowMask.image(), VK_IMAGE_LAYOUT_GENERAL, &clearVal, 1, &range);
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_CLEAR_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void ShadowPass::renderShadowMap(VkCommandBuffer cmd, VkBuffer, const SceneGpu&, VkBuffer indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) return;
    computeSunViewProj(m_sunDir, m_sceneAabbMin, m_sceneAabbMax, m_shadowUbo.mapped());
    transitionImage(cmd, m_shadowMap.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    {
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = m_shadowMap.view(); depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {1.0f, 0};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, m_shadowMapSize}; ri.layerCount = 1;
        ri.colorAttachmentCount = 0; ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{0, 0, (float)m_shadowMapSize.width, (float)m_shadowMapSize.height, 0, 1};
        VkRect2D sc{{0, 0}, m_shadowMapSize};
        vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, VkPipe(m_smPipeline));
        VkDescriptorSet smDs = VkSet(m_smSet);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, VkLay(m_smPipeline), 0, 1, &smDs, 0, nullptr);
        vkCmdBindIndexBuffer(cmd, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        for (uint32_t d = 0; d < drawCount; ++d) {
            VkDeviceSize offset = d > 0 ? d * sizeof(VkDrawIndexedIndirectCommand) : 0;
            vkCmdDrawIndexedIndirect(cmd, indirectBuf, offset, 1, sizeof(VkDrawIndexedIndirectCommand));
        }
        vkCmdEndRendering(cmd);
    }
    transitionImage(cmd, m_shadowMap.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void ShadowPass::recordHardSM(VkCommandBuffer cmd, const RenderTargets&, VkBuffer frameUbo, const SceneGpu& sceneGpu, VkBuffer indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) { recordNone(cmd); return; }
    renderShadowMap(cmd, frameUbo, sceneGpu, indirectBuf, drawCount);
    restoreResolveDesc(*m_set, *m_rhiDevice, m_shadowMap.view(), (VkSampler)(uintptr_t)m_shadowSampler->nativeHandle());
    if (!m_fgAutoBarrier) {
        transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkPipe(m_resolveHard));
    VkDescriptorSet ds[2] = {VkSet(m_set), VkSet(m_frameSet)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkLay(m_resolveHard), 0, 2, ds, 0, nullptr);
    ResolvePC pc{}; pc.outSizeX = m_outputSize.width; pc.outSizeY = m_outputSize.height;
    pc.invOutSizeX = 1.f / (float)pc.outSizeX; pc.invOutSizeY = 1.f / (float)pc.outSizeY;
    pc.bias = 0.001f; pc.invShadowMapX = 1.f / (float)m_shadowMapSize.width; pc.invShadowMapY = 1.f / (float)m_shadowMapSize.height;
    vkCmdPushConstants(cmd, VkLay(m_resolveHard), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (m_outputSize.width + 7) / 8, (m_outputSize.height + 7) / 8, 1);
    if (!m_fgAutoBarrier) {
        transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
}

void ShadowPass::recordPCF(VkCommandBuffer cmd, const RenderTargets&, VkBuffer frameUbo, const SceneGpu& sceneGpu, VkBuffer indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) { recordNone(cmd); return; }
    renderShadowMap(cmd, frameUbo, sceneGpu, indirectBuf, drawCount);
    restoreResolveDesc(*m_set, *m_rhiDevice, m_shadowMap.view(), (VkSampler)(uintptr_t)m_shadowSampler->nativeHandle());
    if (!m_fgAutoBarrier) {
        transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkPipe(m_resolvePCF));
    VkDescriptorSet ds[2] = {VkSet(m_set), VkSet(m_frameSet)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkLay(m_resolvePCF), 0, 2, ds, 0, nullptr);
    ResolvePC pc{}; pc.outSizeX = m_outputSize.width; pc.outSizeY = m_outputSize.height;
    pc.invOutSizeX = 1.f / (float)pc.outSizeX; pc.invOutSizeY = 1.f / (float)pc.outSizeY;
    pc.bias = 0.001f; pc.invShadowMapX = 1.f / (float)m_shadowMapSize.width; pc.invShadowMapY = 1.f / (float)m_shadowMapSize.height;
    vkCmdPushConstants(cmd, VkLay(m_resolvePCF), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (m_outputSize.width + 7) / 8, (m_outputSize.height + 7) / 8, 1);
    if (!m_fgAutoBarrier) {
        transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
}

void ShadowPass::recordPCSS(VkCommandBuffer cmd, const RenderTargets&, VkBuffer frameUbo, const SceneGpu& sceneGpu, VkBuffer indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) { recordNone(cmd); return; }
    renderShadowMap(cmd, frameUbo, sceneGpu, indirectBuf, drawCount);
    restoreResolveDesc(*m_set, *m_rhiDevice, m_shadowMap.view(), (VkSampler)(uintptr_t)m_shadowSampler->nativeHandle());
    if (!m_fgAutoBarrier) {
        transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkPipe(m_resolvePCSS));
    VkDescriptorSet ds[2] = {VkSet(m_set), VkSet(m_frameSet)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkLay(m_resolvePCSS), 0, 2, ds, 0, nullptr);
    PCSS_PC pc{}; pc.outSizeX = m_outputSize.width; pc.outSizeY = m_outputSize.height;
    pc.invOutSizeX = 1.f / (float)pc.outSizeX; pc.invOutSizeY = 1.f / (float)pc.outSizeY;
    pc.bias = 0.001f; pc.invShadowMapX = 1.f / (float)m_shadowMapSize.width; pc.invShadowMapY = 1.f / (float)m_shadowMapSize.height;
    pc.lightSize = 0.03f; pc.maxKernelRadius = 15.0f;
    vkCmdPushConstants(cmd, VkLay(m_resolvePCSS), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (m_outputSize.width + 7) / 8, (m_outputSize.height + 7) / 8, 1);
    if (!m_fgAutoBarrier) {
        transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
}

void ShadowPass::recordVSM(VkCommandBuffer cmd, const RenderTargets&, VkBuffer, const SceneGpu&, VkBuffer indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) { recordNone(cmd); return; }
    computeSunViewProj(m_sunDir, m_sceneAabbMin, m_sceneAabbMax, m_shadowUbo.mapped());
    transitionImage(cmd, m_vsmMap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    transitionImage(cmd, m_shadowMap.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    {
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = m_vsmMap.view(); color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue.color = {{1.0f, 1.0f, 0.0f, 0.0f}};
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = m_shadowMap.view(); depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {1.0f, 0};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, m_shadowMapSize}; ri.layerCount = 1;
        ri.colorAttachmentCount = 1; ri.pColorAttachments = &color; ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{0, 0, (float)m_shadowMapSize.width, (float)m_shadowMapSize.height, 0, 1};
        VkRect2D sc{{0, 0}, m_shadowMapSize};
        vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, VkPipe(m_vsmGenPipeline));
        VkDescriptorSet smDs = VkSet(m_smSet);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, VkLay(m_vsmGenPipeline), 0, 1, &smDs, 0, nullptr);
        vkCmdBindIndexBuffer(cmd, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        for (uint32_t d = 0; d < drawCount; ++d) {
            VkDeviceSize offset = d > 0 ? d * sizeof(VkDrawIndexedIndirectCommand) : 0;
            vkCmdDrawIndexedIndirect(cmd, indirectBuf, offset, 1, sizeof(VkDrawIndexedIndirectCommand));
        }
        vkCmdEndRendering(cmd);
    }
    auto blurDispatch = [&](VkImageView inView, VkImageLayout, VkImageView outView, VkImageLayout, uint32_t dir) {
        auto& vkD2 = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
        auto iv = rhi::VkRHITextureView::createNonOwning(vkD2, inView);
        auto ov = rhi::VkRHITextureView::createNonOwning(vkD2, outView);
        m_vsmBlurSet->write({
            {0, rhi::DescriptorType::SampledImage, iv.get()},
            {1, rhi::DescriptorType::StorageImage, ov.get()},
        });
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkPipe(m_vsmBlurPipeline));
        VkDescriptorSet bs = VkSet(m_vsmBlurSet);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkLay(m_vsmBlurPipeline), 0, 1, &bs, 0, nullptr);
        struct { uint32_t x, y; float ix, iy; uint32_t dir, pad; } pc;
        pc.x = m_shadowMapSize.width; pc.y = m_shadowMapSize.height;
        pc.ix = 1.f / (float)pc.x; pc.iy = 1.f / (float)pc.y; pc.dir = dir; pc.pad = 0;
        vkCmdPushConstants(cmd, VkLay(m_vsmBlurPipeline), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (m_shadowMapSize.width + 7) / 8, (m_shadowMapSize.height + 7) / 8, 1);
    };
    transitionImage(cmd, m_vsmMap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    transitionImage(cmd, m_vsmBlur.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    blurDispatch(m_vsmMap.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, m_vsmBlur.view(), VK_IMAGE_LAYOUT_GENERAL, 0);
    transitionImage(cmd, m_vsmBlur.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    transitionImage(cmd, m_vsmMap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    blurDispatch(m_vsmBlur.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, m_vsmMap.view(), VK_IMAGE_LAYOUT_GENERAL, 1);
    transitionImage(cmd, m_vsmMap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    transitionImage(cmd, m_shadowMap.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    restoreResolveDesc(*m_set, *m_rhiDevice, m_vsmMap.view(), (VkSampler)(uintptr_t)m_vsmSampler->nativeHandle());
    if (!m_fgAutoBarrier) {
        transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkPipe(m_resolveVSM));
    VkDescriptorSet ds[2] = {VkSet(m_set), VkSet(m_frameSet)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkLay(m_resolveVSM), 0, 2, ds, 0, nullptr);
    {
        ResolvePC pc{}; pc.outSizeX = m_outputSize.width; pc.outSizeY = m_outputSize.height;
        pc.invOutSizeX = 1.f / (float)pc.outSizeX; pc.invOutSizeY = 1.f / (float)pc.outSizeY;
        pc.bias = 0.001f; pc.invShadowMapX = 1.f / (float)m_shadowMapSize.width; pc.invShadowMapY = 1.f / (float)m_shadowMapSize.height;
        vkCmdPushConstants(cmd, VkLay(m_resolveVSM), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    }
    vkCmdDispatch(cmd, (m_outputSize.width + 7) / 8, (m_outputSize.height + 7) / 8, 1);
    if (!m_fgAutoBarrier) {
        transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
}

void ShadowPass::recordRTHard(VkCommandBuffer cmd) {
    if (!m_rtHardPipeline) return;
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkPipe(m_rtHardPipeline));
    VkDescriptorSet ds[2] = {VkSet(m_rtSet), VkSet(m_frameSet)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkLay(m_rtHardPipeline), 0, 2, ds, 0, nullptr);
    struct { uint32_t x, y; float ix, iy; } pc;
    pc.x = m_outputSize.width; pc.y = m_outputSize.height; pc.ix = 1.f / (float)pc.x; pc.iy = 1.f / (float)pc.y;
    vkCmdPushConstants(cmd, VkLay(m_rtHardPipeline), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (m_outputSize.width + 7) / 8, (m_outputSize.height + 7) / 8, 1);
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void ShadowPass::recordRTSoft(VkCommandBuffer cmd) {
    if (!m_rtSoftPipeline) { recordRTHard(cmd); return; }
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkPipe(m_rtSoftPipeline));
    VkDescriptorSet ds[2] = {VkSet(m_rtSet), VkSet(m_frameSet)};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkLay(m_rtSoftPipeline), 0, 2, ds, 0, nullptr);
    struct { uint32_t x, y; float ix, iy; uint32_t fi; float sr; uint32_t rc; uint32_t pad; } pc;
    pc.x = m_outputSize.width; pc.y = m_outputSize.height;
    pc.ix = 1.f / (float)pc.x; pc.iy = 1.f / (float)pc.y;
    pc.fi = m_currentFrameIndex; pc.sr = m_rtSunRadius; pc.rc = (uint32_t)m_rtRayCount; pc.pad = 0;
    vkCmdPushConstants(cmd, VkLay(m_rtSoftPipeline), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (m_outputSize.width + 7) / 8, (m_outputSize.height + 7) / 8, 1);
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

// ════════════════════════════════════════════════════════════════
// Destroy
// ════════════════════════════════════════════════════════════════
void ShadowPass::destroy() {
    if (!m_device) return;
    destroyPipelines();
    m_shadowSampler.reset(); m_vsmSampler.reset();
    m_setLayout.reset(); m_set.reset();
    m_smSetLayout.reset(); m_smSet.reset();
    m_frameSetLayout.reset(); m_frameSet.reset();
    m_vsmBlurSetLayout.reset(); m_vsmBlurSet.reset();
    m_rtSetLayout.reset(); m_rtSet.reset();
    m_shadowMap.reset(); m_vsmMap.reset(); m_shadowMask.reset(); m_vsmBlur.reset();
    m_shadowUbo.reset();
    m_device = nullptr;
}

} // namespace somegi
