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

// RHI 路径：直接分发到 RHI 子方法
void ShadowPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets&,
                         const rhi::RHIBuffer&, const SceneGpu&,
                         const rhi::RHIBuffer& indirectBuf, uint32_t drawCount, uint32_t frameIndex) {
    m_currentFrameIndex = frameIndex;
    switch (m_method) {
    case ShadowMethod::None:          recordNone(cmd); break;
    case ShadowMethod::HardShadowMap: recordHardSM(cmd, indirectBuf, drawCount); break;
    case ShadowMethod::PCF:           recordPCF(cmd, indirectBuf, drawCount); break;
    case ShadowMethod::PCSS:          recordPCSS(cmd, indirectBuf, drawCount); break;
    case ShadowMethod::VSM:           recordVSM(cmd, indirectBuf, drawCount); break;
    case ShadowMethod::RTHard:        recordRTHard(cmd); break;
    case ShadowMethod::RTSoft:        recordRTSoft(cmd); break;
    default:                          recordNone(cmd); break;
    }
}

// Vk 兼容路径：委托到 RHI 记录
void ShadowPass::record(VkCommandBuffer cmd, const RenderTargets& rt,
                         VkBuffer frameUbo, const SceneGpu& sceneGpu,
                         VkBuffer indirectBuf, uint32_t drawCount, uint32_t frameIndex) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
    auto rhiIb = rhi::VkRHIBuffer::createNonOwning(vkDev, indirectBuf, VK_WHOLE_SIZE);
    // frameUbo 目前未被子方法实际使用，传入空包装占位
    auto rhiDummy = rhi::VkRHIBuffer::createNonOwning(vkDev, frameUbo, VK_WHOLE_SIZE);
    record(rhiCmd, rt, *rhiDummy, sceneGpu, *rhiIb, drawCount, frameIndex);
}

// RHI 路径：清除 shadowMask 为白色（1.0 = 无阴影）
void ShadowPass::recordNone(rhi::RHICommandBuffer& cmd) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    // 用非拥有型包装临时包装已有 Vulkan 纹理
    auto rhiTex = rhi::VkRHITexture::createNonOwning(vkDev, m_shadowMask.image(),
        rhi::toRhiFormat(m_shadowMask.format()), m_outputSize.width, m_outputSize.height);

    // UNDEFINED → TransferDst（clearColor 需要此布局）
    cmd.textureBarrier(*rhiTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::TransferDst);
    // 清除为白色（1.0 = 无阴影）
    cmd.clearColor(*rhiTex, 1.0f, 1.0f, 1.0f, 1.0f);
    // TransferDst → ShaderReadOnly（供下游 compute pass 读取）
    cmd.textureBarrier(*rhiTex, rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
}

// Vk 兼容路径：委托到 RHI 版本
void ShadowPass::recordNone(VkCommandBuffer cmd) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
    recordNone(rhiCmd);
}

// RHI 路径：渲染 shadow map 深度图
void ShadowPass::renderShadowMap(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) return;
    computeSunViewProj(m_sunDir, m_sceneAabbMin, m_sceneAabbMax, m_shadowUbo.mapped());

    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    // 非拥有型包装：深度纹理
    auto depthTex = rhi::VkRHITexture::createNonOwning(vkDev, m_shadowMap.image(),
        rhi::toRhiFormat(m_shadowMap.format()), m_shadowMapSize.width, m_shadowMapSize.height);
    auto depthView = rhi::VkRHITextureView::createNonOwning(vkDev, m_shadowMap.view());

    // UNDEFINED → DepthAttachment
    cmd.textureBarrier(*depthTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::DepthAttachment);

    // 深度附件描述
    rhi::RenderingAttachmentInfo depthAttach{};
    depthAttach.view = depthView.get();
    depthAttach.loadOp = rhi::AttachmentLoadOp::Clear;
    depthAttach.storeOp = rhi::AttachmentStoreOp::Store;
    depthAttach.clearDepth = 1.0f;

    cmd.beginRendering(nullptr, 0, &depthAttach, m_shadowMapSize.width, m_shadowMapSize.height);
    cmd.setViewport(0, 0, (float)m_shadowMapSize.width, (float)m_shadowMapSize.height);
    cmd.setScissor(0, 0, m_shadowMapSize.width, m_shadowMapSize.height);
    cmd.bindPipelineState(*m_smPipeline);
    cmd.bindDescriptorSet(0, *m_smSet);

    // 索引缓冲：非拥有型包装已有 VkBuffer
    auto ibo = rhi::VkRHIBuffer::createNonOwning(vkDev, m_indexBuffer, VK_WHOLE_SIZE);
    cmd.bindIndexBuffer(*ibo, 0, false);  // false = UINT32

    cmd.drawIndexedIndirect(indirectBuf, 0, drawCount, sizeof(VkDrawIndexedIndirectCommand));

    cmd.endRendering();

    // DepthAttachment → ShaderReadOnly（供 resolve/downstream 读取）
    cmd.textureBarrier(*depthTex, rhi::TextureLayout::DepthAttachment, rhi::TextureLayout::ShaderReadOnly);
}

// Vk 兼容路径：委托到 RHI 版本
void ShadowPass::renderShadowMap(VkCommandBuffer cmd, VkBuffer, const SceneGpu&, VkBuffer indirectBuf, uint32_t drawCount) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
    auto rhiIb = rhi::VkRHIBuffer::createNonOwning(vkDev, indirectBuf, VK_WHOLE_SIZE);
    renderShadowMap(rhiCmd, *rhiIb, drawCount);
}

// ── HardSM RHI 路径：渲染 shadow map + compute resolve ──
void ShadowPass::recordHardSM(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) { recordNone(cmd); return; }
    renderShadowMap(cmd, indirectBuf, drawCount);

    // 恢复 resolve 描述符（shadow map view + sampler）
    restoreResolveDesc(*m_set, *m_rhiDevice, m_shadowMap.view(),
        (VkSampler)(uintptr_t)m_shadowSampler->nativeHandle());

    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    if (!m_fgAutoBarrier) {
        auto maskTex = rhi::VkRHITexture::createNonOwning(vkDev, m_shadowMask.image(),
            rhi::toRhiFormat(m_shadowMask.format()), m_outputSize.width, m_outputSize.height);
        cmd.textureBarrier(*maskTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
    }

    cmd.bindPipelineState(*m_resolveHard);
    const rhi::RHIDescriptorSet* ds[2] = {m_set.get(), m_frameSet.get()};
    cmd.bindDescriptorSets(0, 2, ds);

    ResolvePC pc{};
    pc.outSizeX = m_outputSize.width; pc.outSizeY = m_outputSize.height;
    pc.invOutSizeX = 1.f / (float)pc.outSizeX; pc.invOutSizeY = 1.f / (float)pc.outSizeY;
    pc.bias = 0.001f;
    pc.invShadowMapX = 1.f / (float)m_shadowMapSize.width;
    pc.invShadowMapY = 1.f / (float)m_shadowMapSize.height;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
    cmd.dispatch((m_outputSize.width + 7) / 8, (m_outputSize.height + 7) / 8);

    if (!m_fgAutoBarrier) {
        auto maskTex = rhi::VkRHITexture::createNonOwning(vkDev, m_shadowMask.image(),
            rhi::toRhiFormat(m_shadowMask.format()), m_outputSize.width, m_outputSize.height);
        cmd.textureBarrier(*maskTex, rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
    }
}
// Vk 兼容路径
void ShadowPass::recordHardSM(VkCommandBuffer cmd, const RenderTargets&, VkBuffer, const SceneGpu&, VkBuffer indirectBuf, uint32_t drawCount) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
    auto rhiIb = rhi::VkRHIBuffer::createNonOwning(vkDev, indirectBuf, VK_WHOLE_SIZE);
    recordHardSM(rhiCmd, *rhiIb, drawCount);
}

// ── PCF RHI 路径 ──
void ShadowPass::recordPCF(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) { recordNone(cmd); return; }
    renderShadowMap(cmd, indirectBuf, drawCount);

    restoreResolveDesc(*m_set, *m_rhiDevice, m_shadowMap.view(),
        (VkSampler)(uintptr_t)m_shadowSampler->nativeHandle());

    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    if (!m_fgAutoBarrier) {
        auto maskTex = rhi::VkRHITexture::createNonOwning(vkDev, m_shadowMask.image(),
            rhi::toRhiFormat(m_shadowMask.format()), m_outputSize.width, m_outputSize.height);
        cmd.textureBarrier(*maskTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
    }

    cmd.bindPipelineState(*m_resolvePCF);
    const rhi::RHIDescriptorSet* ds[2] = {m_set.get(), m_frameSet.get()};
    cmd.bindDescriptorSets(0, 2, ds);

    ResolvePC pc{};
    pc.outSizeX = m_outputSize.width; pc.outSizeY = m_outputSize.height;
    pc.invOutSizeX = 1.f / (float)pc.outSizeX; pc.invOutSizeY = 1.f / (float)pc.outSizeY;
    pc.bias = 0.001f;
    pc.invShadowMapX = 1.f / (float)m_shadowMapSize.width;
    pc.invShadowMapY = 1.f / (float)m_shadowMapSize.height;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
    cmd.dispatch((m_outputSize.width + 7) / 8, (m_outputSize.height + 7) / 8);

    if (!m_fgAutoBarrier) {
        auto maskTex = rhi::VkRHITexture::createNonOwning(vkDev, m_shadowMask.image(),
            rhi::toRhiFormat(m_shadowMask.format()), m_outputSize.width, m_outputSize.height);
        cmd.textureBarrier(*maskTex, rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
    }
}
void ShadowPass::recordPCF(VkCommandBuffer cmd, const RenderTargets&, VkBuffer, const SceneGpu&, VkBuffer indirectBuf, uint32_t drawCount) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
    auto rhiIb = rhi::VkRHIBuffer::createNonOwning(vkDev, indirectBuf, VK_WHOLE_SIZE);
    recordPCF(rhiCmd, *rhiIb, drawCount);
}

// ── PCSS RHI 路径 ──
void ShadowPass::recordPCSS(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) { recordNone(cmd); return; }
    renderShadowMap(cmd, indirectBuf, drawCount);

    restoreResolveDesc(*m_set, *m_rhiDevice, m_shadowMap.view(),
        (VkSampler)(uintptr_t)m_shadowSampler->nativeHandle());

    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    if (!m_fgAutoBarrier) {
        auto maskTex = rhi::VkRHITexture::createNonOwning(vkDev, m_shadowMask.image(),
            rhi::toRhiFormat(m_shadowMask.format()), m_outputSize.width, m_outputSize.height);
        cmd.textureBarrier(*maskTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
    }

    cmd.bindPipelineState(*m_resolvePCSS);
    const rhi::RHIDescriptorSet* ds[2] = {m_set.get(), m_frameSet.get()};
    cmd.bindDescriptorSets(0, 2, ds);

    PCSS_PC pc{};
    pc.outSizeX = m_outputSize.width; pc.outSizeY = m_outputSize.height;
    pc.invOutSizeX = 1.f / (float)pc.outSizeX; pc.invOutSizeY = 1.f / (float)pc.outSizeY;
    pc.bias = 0.001f;
    pc.invShadowMapX = 1.f / (float)m_shadowMapSize.width;
    pc.invShadowMapY = 1.f / (float)m_shadowMapSize.height;
    pc.lightSize = 0.03f;
    pc.maxKernelRadius = 15.0f;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
    cmd.dispatch((m_outputSize.width + 7) / 8, (m_outputSize.height + 7) / 8);

    if (!m_fgAutoBarrier) {
        auto maskTex = rhi::VkRHITexture::createNonOwning(vkDev, m_shadowMask.image(),
            rhi::toRhiFormat(m_shadowMask.format()), m_outputSize.width, m_outputSize.height);
        cmd.textureBarrier(*maskTex, rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
    }
}
void ShadowPass::recordPCSS(VkCommandBuffer cmd, const RenderTargets&, VkBuffer, const SceneGpu&, VkBuffer indirectBuf, uint32_t drawCount) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
    auto rhiIb = rhi::VkRHIBuffer::createNonOwning(vkDev, indirectBuf, VK_WHOLE_SIZE);
    recordPCSS(rhiCmd, *rhiIb, drawCount);
}

// ── VSM RHI 路径：VSM 生成 + 双向模糊 + resolve ──
void ShadowPass::recordVSM(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) { recordNone(cmd); return; }
    computeSunViewProj(m_sunDir, m_sceneAabbMin, m_sceneAabbMax, m_shadowUbo.mapped());

    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    // 非拥有型纹理包装
    auto vsmTex = rhi::VkRHITexture::createNonOwning(vkDev, m_vsmMap.image(),
        rhi::toRhiFormat(m_vsmMap.format()), m_shadowMapSize.width, m_shadowMapSize.height);
    auto depthTex = rhi::VkRHITexture::createNonOwning(vkDev, m_shadowMap.image(),
        rhi::toRhiFormat(m_shadowMap.format()), m_shadowMapSize.width, m_shadowMapSize.height);
    auto vsmView = rhi::VkRHITextureView::createNonOwning(vkDev, m_vsmMap.view());
    auto depthView = rhi::VkRHITextureView::createNonOwning(vkDev, m_shadowMap.view());

    // 过渡到 attachment 布局
    cmd.textureBarrier(*vsmTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::ColorAttachment);
    cmd.textureBarrier(*depthTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::DepthAttachment);

    // VSM 生成 pass（MRT: color + depth）
    {
        rhi::RenderingAttachmentInfo colorAttach{};
        colorAttach.view = vsmView.get();
        colorAttach.loadOp = rhi::AttachmentLoadOp::Clear;
        colorAttach.storeOp = rhi::AttachmentStoreOp::Store;
        colorAttach.clearColor[0] = 1.0f; colorAttach.clearColor[1] = 1.0f;
        colorAttach.clearColor[2] = 0.0f; colorAttach.clearColor[3] = 0.0f;

        rhi::RenderingAttachmentInfo depthAttach{};
        depthAttach.view = depthView.get();
        depthAttach.loadOp = rhi::AttachmentLoadOp::Clear;
        depthAttach.storeOp = rhi::AttachmentStoreOp::Store;
        depthAttach.clearDepth = 1.0f;

        cmd.beginRendering(&colorAttach, 1, &depthAttach, m_shadowMapSize.width, m_shadowMapSize.height);
        cmd.setViewport(0, 0, (float)m_shadowMapSize.width, (float)m_shadowMapSize.height);
        cmd.setScissor(0, 0, m_shadowMapSize.width, m_shadowMapSize.height);
        cmd.bindPipelineState(*m_vsmGenPipeline);
        cmd.bindDescriptorSet(0, *m_smSet);

        auto ibo = rhi::VkRHIBuffer::createNonOwning(vkDev, m_indexBuffer, VK_WHOLE_SIZE);
        cmd.bindIndexBuffer(*ibo, 0, false);
        cmd.drawIndexedIndirect(indirectBuf, 0, drawCount, sizeof(VkDrawIndexedIndirectCommand));
        cmd.endRendering();
    }

    // VSM 双向模糊（RHI 路径）
    auto blurDispatch = [&](rhi::RHITextureView& inView, rhi::TextureLayout inLayout,
                             rhi::RHITextureView& outView, rhi::TextureLayout outLayout, uint32_t dir) {
        // 更新模糊描述符
        m_vsmBlurSet->write({
            {0, rhi::DescriptorType::SampledImage, &inView},
            {1, rhi::DescriptorType::StorageImage, &outView},
        });
        cmd.bindPipelineState(*m_vsmBlurPipeline);
        cmd.bindDescriptorSet(0, *m_vsmBlurSet);
        struct { uint32_t x, y; float ix, iy; uint32_t dir, pad; } pc;
        pc.x = m_shadowMapSize.width; pc.y = m_shadowMapSize.height;
        pc.ix = 1.f / (float)pc.x; pc.iy = 1.f / (float)pc.y;
        pc.dir = dir; pc.pad = 0;
        cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
        cmd.dispatch((m_shadowMapSize.width + 7) / 8, (m_shadowMapSize.height + 7) / 8);
    };

    auto blurTex = rhi::VkRHITexture::createNonOwning(vkDev, m_vsmBlur.image(),
        rhi::toRhiFormat(m_vsmBlur.format()), m_shadowMapSize.width, m_shadowMapSize.height);
    auto blurView = rhi::VkRHITextureView::createNonOwning(vkDev, m_vsmBlur.view());

    // 水平模糊：VSM map → blur scratch
    cmd.textureBarrier(*vsmTex, rhi::TextureLayout::ColorAttachment, rhi::TextureLayout::ShaderReadOnly);
    cmd.textureBarrier(*blurTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
    blurDispatch(*vsmView, rhi::TextureLayout::ShaderReadOnly, *blurView, rhi::TextureLayout::General, 0);

    // 垂直模糊：blur scratch → VSM map
    cmd.textureBarrier(*blurTex, rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
    cmd.textureBarrier(*vsmTex, rhi::TextureLayout::ShaderReadOnly, rhi::TextureLayout::General);
    blurDispatch(*blurView, rhi::TextureLayout::ShaderReadOnly, *vsmView, rhi::TextureLayout::General, 1);

    // VSM map 最终布局
    cmd.textureBarrier(*vsmTex, rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
    // depth map 也过渡到可读布局
    cmd.textureBarrier(*depthTex, rhi::TextureLayout::DepthAttachment, rhi::TextureLayout::ShaderReadOnly);

    // Resolve dispatch
    restoreResolveDesc(*m_set, *m_rhiDevice, m_vsmMap.view(), (VkSampler)(uintptr_t)m_vsmSampler->nativeHandle());
    auto maskTex = rhi::VkRHITexture::createNonOwning(vkDev, m_shadowMask.image(),
        rhi::toRhiFormat(m_shadowMask.format()), m_outputSize.width, m_outputSize.height);

    if (!m_fgAutoBarrier) {
        cmd.textureBarrier(*maskTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
    }

    cmd.bindPipelineState(*m_resolveVSM);
    const rhi::RHIDescriptorSet* ds[2] = {m_set.get(), m_frameSet.get()};
    cmd.bindDescriptorSets(0, 2, ds);

    ResolvePC pc{};
    pc.outSizeX = m_outputSize.width; pc.outSizeY = m_outputSize.height;
    pc.invOutSizeX = 1.f / (float)pc.outSizeX; pc.invOutSizeY = 1.f / (float)pc.outSizeY;
    pc.bias = 0.001f;
    pc.invShadowMapX = 1.f / (float)m_shadowMapSize.width;
    pc.invShadowMapY = 1.f / (float)m_shadowMapSize.height;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
    cmd.dispatch((m_outputSize.width + 7) / 8, (m_outputSize.height + 7) / 8);

    if (!m_fgAutoBarrier) {
        cmd.textureBarrier(*maskTex, rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
    }
}
// Vk 兼容路径
void ShadowPass::recordVSM(VkCommandBuffer cmd, const RenderTargets&, VkBuffer, const SceneGpu&, VkBuffer indirectBuf, uint32_t drawCount) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
    auto rhiIb = rhi::VkRHIBuffer::createNonOwning(vkDev, indirectBuf, VK_WHOLE_SIZE);
    recordVSM(rhiCmd, *rhiIb, drawCount);
}

// ── RT Hard RHI 路径：硬件光追硬阴影 ──
void ShadowPass::recordRTHard(rhi::RHICommandBuffer& cmd) {
    if (!m_rtHardPipeline) return;
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto maskTex = rhi::VkRHITexture::createNonOwning(vkDev, m_shadowMask.image(),
        rhi::toRhiFormat(m_shadowMask.format()), m_outputSize.width, m_outputSize.height);

    cmd.textureBarrier(*maskTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::General);

    cmd.bindPipelineState(*m_rtHardPipeline);
    const rhi::RHIDescriptorSet* ds[2] = {m_rtSet.get(), m_frameSet.get()};
    cmd.bindDescriptorSets(0, 2, ds);

    struct { uint32_t x, y; float ix, iy; } pc;
    pc.x = m_outputSize.width; pc.y = m_outputSize.height;
    pc.ix = 1.f / (float)pc.x; pc.iy = 1.f / (float)pc.y;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
    cmd.dispatch((m_outputSize.width + 7) / 8, (m_outputSize.height + 7) / 8);

    cmd.textureBarrier(*maskTex, rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
}
// Vk 兼容路径
void ShadowPass::recordRTHard(VkCommandBuffer cmd) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
    recordRTHard(rhiCmd);
}

// ── RT Soft RHI 路径：硬件光追软阴影（多采样） ──
void ShadowPass::recordRTSoft(rhi::RHICommandBuffer& cmd) {
    if (!m_rtSoftPipeline) { recordRTHard(cmd); return; }
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto maskTex = rhi::VkRHITexture::createNonOwning(vkDev, m_shadowMask.image(),
        rhi::toRhiFormat(m_shadowMask.format()), m_outputSize.width, m_outputSize.height);

    cmd.textureBarrier(*maskTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::General);

    cmd.bindPipelineState(*m_rtSoftPipeline);
    const rhi::RHIDescriptorSet* ds[2] = {m_rtSet.get(), m_frameSet.get()};
    cmd.bindDescriptorSets(0, 2, ds);

    struct { uint32_t x, y; float ix, iy; uint32_t fi; float sr; uint32_t rc; uint32_t pad; } pc;
    pc.x = m_outputSize.width; pc.y = m_outputSize.height;
    pc.ix = 1.f / (float)pc.x; pc.iy = 1.f / (float)pc.y;
    pc.fi = m_currentFrameIndex; pc.sr = m_rtSunRadius; pc.rc = (uint32_t)m_rtRayCount; pc.pad = 0;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
    cmd.dispatch((m_outputSize.width + 7) / 8, (m_outputSize.height + 7) / 8);

    cmd.textureBarrier(*maskTex, rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
}
void ShadowPass::recordRTSoft(VkCommandBuffer cmd) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
    recordRTSoft(rhiCmd);
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
