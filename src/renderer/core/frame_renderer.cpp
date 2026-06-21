#include "renderer/core/frame_renderer.h"
#include "core/device.h"
#include "rhi/vulkan/vk_device.h"  // VkRHIDevice shared-handle constructor
#include "rhi/vulkan/vk_command.h" // VkRHICommandBuffer for recordRHI callbacks
#include "rhi/vulkan/vk_buffer.h"  // VkRHIBuffer for non-owning buffer wrappers
#include "rhi/vulkan/vk_texture.h" // VkRHITexture for non-owning texture wrappers
#include "core/debug_dump.h"
#include "scene/upload.h"
#include <cstdio>

namespace somegi {

void FrameRenderer::init(Device& d, VkCommandPool pool, VkExtent2D extent,
                          VkSampleCountFlagBits msaaSamples, bool rtSupported,
                          VkFormat /*swapchainFmt*/, GLFWwindow* /*window*/) {
    m_device = &d;
    m_pool   = pool;
    m_rtSupported = rtSupported;
    m_meshShaderSupported = d.features().meshShader;
    m_taskShaderSupported = d.features().taskShader;

    // ── 创建 RHI Device（共享 core::Device 的底层 Vulkan 句柄） ──
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d.physicalDevice(), &props);
        rhi::DeviceLimits rhiLimits;
        rhiLimits.maxTextureSize      = props.limits.maxImageDimension2D;
        rhiLimits.maxSampledTextures  = props.limits.maxPerStageDescriptorSampledImages;
        rhiLimits.maxUniformBufferSize = props.limits.maxUniformBufferRange;
        rhiLimits.maxStorageBufferSize = props.limits.maxStorageBufferRange;
        rhiLimits.maxPushConstantsSize = props.limits.maxPushConstantsSize;
        rhiLimits.timestampPeriod      = props.limits.timestampPeriod;
        rhiLimits.meshShaderSupported  = d.features().meshShader;
        rhiLimits.rayTracingSupported  = d.features().accelStruct && d.features().rayQuery;
        m_rhiDevice = std::make_unique<rhi::VkRHIDevice>(
            d.instance(), d.physicalDevice(), d.device(), d.allocator(),
            d.graphicsQueue(), d.graphicsQueueFamily(), d.dispatch(), rhiLimits);
    }

    // Timestamp query pool
    {
        m_timestampPool = m_rhiDevice->createQueryPool(kFramesInFlight * kTimestampSlots);
    }
    m_passNames[kTsStart]    = "Start";
    m_passNames[kTsGBuffer]  = "GBuffer";
    m_passNames[kTsAO]       = "AO+SS";
    m_passNames[kTsVoxelGI]  = "VoxelGI";
    m_passNames[kTsLighting] = "Lighting";
    m_passNames[kTsSkybox]   = "Skybox";
    m_passNames[kTsTonemap]  = "Tonemap";
    m_passNames[kTsAA]       = "AA";
    m_passNames[kTsEnd]      = "End";

    // Render targets
    std::printf("[init] render targets...\n");
    m_rt.create(d, extent, msaaSamples);

    // Core passes
    std::printf("[init] gbuffer pass...\n");
    m_gbuffer.init(d, *m_rhiDevice, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT,
                   VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT, 128, msaaSamples);
    std::printf("[init] forward pass...\n");
    m_forward.init(d, *m_rhiDevice, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_D32_SFLOAT, 128);
    std::printf("[init] rsm geometry pass...\n");
    m_rsmGeom.init(*m_rhiDevice, 128);
    std::printf("[init] rsm sample pass...\n");
    m_rsmSample.init(*m_rhiDevice);

    // GI resources
    std::printf("[init] lpv resources...\n");
    m_lpv.create(d, kLpvResolution);
    std::printf("[init] vxgi resources...\n");
    m_vxgi.create(d, kVxgiResolution);
    std::printf("[init] prt resources...\n");
    m_prt.create(d, kPrtResolution);
    std::printf("[init] ddgi resources + pass...\n");
    m_ddgi.create(d);
    {
        auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
        oneShotSubmit(*m_rhiDevice, [&](VkCommandBuffer cmd) {
            rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
            auto bufWrap = rhi::VkRHIBuffer::createNonOwning(vkDev, m_ddgi.probeStates().handle(), VK_WHOLE_SIZE);
            rhiCmd.fillBuffer(*bufWrap, 0, VK_WHOLE_SIZE, 1u);
        });
    }
    m_ddgiPass.init(*m_rhiDevice);
    m_ddgiPass.bindResources(m_ddgi, m_vxgi);

    std::printf("[init] ndgi resources + pass...\n");
    m_ndgi.create(d);
    m_ndgiPass.init(*m_rhiDevice, rtSupported);
    m_ndgiInited = false;

    std::printf("[init] shadow pass...\n");
    m_shadow.init(d, *m_rhiDevice, {2048, 2048}, extent);

    std::printf("[init] lighting pass...\n");
    m_lighting.init(*m_rhiDevice);
    m_lighting.bindShadowMask(m_shadow.shadowMask().view());
    m_lighting.bindFrame(m_rt, m_gbuffer.frameUboHandle(),
                         m_lpv.current(), m_vxgi, m_prt, m_ddgi,
                         m_ddgi.probeStates().handle());
    m_lighting.setNdgiWeights(
        m_ndgi.weights1().handle(), m_ndgi.bias1().handle(),
        m_ndgi.weights2().handle(), m_ndgi.bias2().handle(),
        m_ndgi.weights3().handle(), m_ndgi.bias3().handle());
    m_forward.setNdgiWeights(d,
        m_ndgi.weights1().handle(), m_ndgi.bias1().handle(),
        m_ndgi.weights2().handle(), m_ndgi.bias2().handle(),
        m_ndgi.weights3().handle(), m_ndgi.bias3().handle());

    std::printf("[init] ssao pass...\n");
    m_ssao.init(*m_rhiDevice);
    m_ssao.bindFrame(m_rt);
    std::printf("[init] gtao pass...\n");
    m_gtao.init(*m_rhiDevice);
    m_gtao.bindFrame(m_rt);
    std::printf("[init] ssr pass...\n");
    m_ssr.init(*m_rhiDevice);
    m_ssr.bindFrame(m_rt, m_gbuffer.frameUboHandle());
    std::printf("[init] ssgi pass...\n");
    m_ssgi.init(*m_rhiDevice);
    m_ssgi.bindFrame(m_rt, m_gbuffer.frameUboHandle());
    std::printf("[init] gtgi pass...\n");
    m_gtgi.init(*m_rhiDevice);
    m_gtgi.bindFrame(m_rt, m_gbuffer.frameUboHandle());

    std::printf("[init] sdfgi resources/pass...\n");
    m_sdfgi.create(d, kSdfgiResolution);
    m_sdfgiPass.init(*m_rhiDevice);
    m_sdfgiPass.bindResources(m_sdfgi, m_vxgi, m_rt, m_gbuffer.frameUboHandle());

    std::printf("[init] restir resources/pass...\n");
    m_restir.create(d, extent, kRestirMaxLights);
    m_restirPass.init(*m_rhiDevice, rtSupported);

    if (rtSupported) {
        std::printf("[init] rt gi pass...\n");
        m_rtGiPass.init(*m_rhiDevice);
        m_rtGiInited = true;
    }
    if (rtSupported) {
        std::printf("[init] lumen resources...\n");
        m_lumen.create(d, extent);
    }
    m_restirPass.bindResources(m_restir, m_vxgi, m_rt, m_gbuffer.frameUboHandle());

    m_rsmSample.bindFrame(m_rt, m_gbuffer.frameUboHandle(),
        m_rsmGeom.frameUboHandle(),
        m_rsmGeom.position(), m_rsmGeom.normal(), m_rsmGeom.flux());

    std::printf("[init] lpv inject pass...\n");
    m_lpvInject.init(*m_rhiDevice, RsmGeometryPass::kRsmSize);

    std::printf("[init] vxgi voxelize/inject/mipmap pass...\n");
    m_vxgiVoxelize.init(*m_rhiDevice, 128);
    m_vxgiInject.init(*m_rhiDevice, RsmGeometryPass::kRsmSize);
    m_vxgiInject.bindResources(m_rsmGeom.position(), m_rsmGeom.flux(), m_vxgi);
    m_vxgiMipmap.init(*m_rhiDevice, m_vxgi.mipLevels());
    m_vxgiMipmap.bindResources(m_vxgi);
    m_vxgiAniso.init(*m_rhiDevice, m_vxgi.mipLevels());
    m_vxgiAniso.bindResources(m_vxgi);
    m_vxgiRelight.init(*m_rhiDevice);
    m_vxgiRelight.bindResources(m_vxgi, m_vxgi.relightScratch().view());
    m_vxgiRelight.bindResourcesPingPong(m_vxgi, false);
    m_vxgiRelight.bindResourcesPingPong(m_vxgi, true);
    if (m_vxgiSixAxisInited) {
        m_vxgiResolve6Axis.init(*m_rhiDevice);
        m_vxgiResolve6Axis.bindResources(m_vxgi);
    }

    std::printf("[init] prt bake pass...\n");
    m_prtBake.init(*m_rhiDevice);
    m_prtBake.bindResources(m_vxgi, m_prt);
    m_lpvInject.bindResources(m_rsmGeom.position(), m_rsmGeom.normal(),
                              m_rsmGeom.flux(), m_lpv.current(), m_lpv.gv());
    std::printf("[init] lpv propagate pass...\n");
    m_lpvProp.init(*m_rhiDevice);
    m_lpvProp.bindResources(m_lpv.current(), m_lpv.next(), m_lpv.gv());

    bootstrapHdrPrev();
    bootstrapSsgiTemporal();
    bootstrapAllTargets();

    std::printf("[init] skybox pass...\n");
    m_skybox.init(*m_rhiDevice, rhi::Format::R16G16B16A16_SFLOAT, rhi::Format::D32_SFLOAT);

    std::printf("[init] tonemap pass...\n");
    // linearSampler comes from SceneGpu — caller sets after scene load
    std::printf("[init] aa passes...\n");
    m_taa.init(*m_rhiDevice);
    m_smaa.init(*m_rhiDevice, extent);
    std::printf("[init] imgui pass...\n");

    m_cullPass.init(*m_rhiDevice, 4096);
    m_hizPass.init(*m_rhiDevice, extent);
    registerPipelineSteps();

    // Mesh Shader：支持时默认启用
    if (m_meshShaderSupported) {
        m_useMeshShader = true;
        m_gbuffer.setMeshShaderEnabled(true);
        std::printf("[init] Mesh Shader enabled by default\n");
    }

    std::printf("[init] all renderer passes set up.\n");
}

void FrameRenderer::initD3D12(rhi::RHIDevice& rhiDev, VkExtent2D extent) {
    // 创建 D3D12 渲染目标（20 纹理）
    m_rt.createRHI(rhiDev, extent, VK_SAMPLE_COUNT_1_BIT);

    // Phase 5: 逐步接入 Pass（当前需要 core::Device&，D3D12 暂跳过）
    std::printf("[d3d12] FrameRenderer::initD3D12 — render targets created\n");
}

void FrameRenderer::destroy() {
    if (!m_device) return;
    m_imgui.destroy();
    m_tonemap.destroy();
    m_taa.destroy();
    m_smaa.destroy();
    m_rtGiPass.destroy();
    m_rtAS.destroy();
    m_skybox.destroy();
    m_forward.destroy();
    m_ssgi.destroy();
    m_gtgi.destroy();
    m_ssr.destroy();
    m_ssao.destroy();
    m_gtao.destroy();
    m_lighting.destroy();
    m_ndgiPass.destroy();
    m_ndgi.destroy();
    m_ddgiPass.destroy();
    m_ddgi.destroy();
    m_prtBake.destroy();
    m_prt.destroy();
    m_vxgiMipmap.destroy();
    m_vxgiAniso.destroy();
    m_vxgiRelight.destroy();
    m_vxgiResolve6Axis.destroy();
    m_sdfgiPass.destroy();
    m_sdfgi.destroy();
    m_lumenProbePass.destroy();
    m_lumenFilterPass.destroy();
    m_lumenGatherPass.destroy();
    m_lumen.destroy();
    m_restirPass.destroy();
    m_restir.destroy();
    m_vxgiInject.destroy();
    m_vxgiVoxelize.destroy();
    m_vxgi.destroy();
    m_lpvProp.destroy();
    m_lpvInject.destroy();
    m_lpv.destroy();
    m_rsmSample.destroy();
    m_rsmGeom.destroy();
    m_gbuffer.destroy();
    m_shadow.destroy();
    m_cullPass.destroy();
    m_hizPass.destroy();
    m_envIbl.destroy(*m_device);
    m_rt.destroy();
    m_timestampPool.reset();
    m_device = nullptr;
}

void FrameRenderer::onResize(Device& d, VkExtent2D newExtent,
                              VkSampleCountFlagBits msaaSamples,
                              VkFormat /*swapchainFmt*/, GLFWwindow* /*window*/) {
    m_rt.destroy();
    m_rt.create(d, newExtent, msaaSamples);
    m_lighting.bindFrame(m_rt, m_gbuffer.frameUboHandle(),
                         m_lpv.current(), m_vxgi, m_prt, m_ddgi,
                         m_ddgi.probeStates().handle());
    m_ssao.bindFrame(m_rt);
    m_gtao.bindFrame(m_rt);
    m_ssr.bindFrame(m_rt, m_gbuffer.frameUboHandle());
    m_ssgi.bindFrame(m_rt, m_gbuffer.frameUboHandle());
    m_gtgi.bindFrame(m_rt, m_gbuffer.frameUboHandle());
    m_sdfgiPass.bindResources(m_sdfgi, m_vxgi, m_rt, m_gbuffer.frameUboHandle());
    m_restir.resize(d, newExtent);
    m_restirPass.bindResources(m_restir, m_vxgi, m_rt, m_gbuffer.frameUboHandle());
    m_restirOutInited = false;
    m_restirBootstrapped = false;
    m_rsmSample.bindFrame(m_rt, m_gbuffer.frameUboHandle(),
        m_rsmGeom.frameUboHandle(),
        m_rsmGeom.position(), m_rsmGeom.normal(), m_rsmGeom.flux());
    if (m_rtSupported) {
        m_lumen.destroy();
        m_lumen.create(d, newExtent);
        m_lumenAtlasInited = false;
        m_lumenOutInited = false;
    }
    m_tonemap.bindTargets(m_rt);
    bootstrapHdrPrev();
    bootstrapSsgiTemporal();
    bootstrapAllTargets();
}

void FrameRenderer::bindScenePasses(Device& d, const SceneGpu& gpu, uint32_t textureCount) {
    m_gbuffer.bindScene(d, gpu, textureCount);
    m_forward.bindScene(d, gpu, textureCount);
    m_rsmGeom.bindScene(gpu, textureCount);
    m_vxgiVoxelize.bindScene(gpu, textureCount, m_vxgi);
}

void FrameRenderer::applyGiSelection(int giIndex) {
    // SSGI (2) / RSM (3) 屏幕空间技术互斥，其他模式清零
    m_ssgi.enabled       = (giIndex == 2);
    m_rsmSample.enabled  = (giIndex == 3);
    // 体素/网格类 GI
    m_lpvEnabled         = (giIndex == 4);
    m_vxgiEnabled        = (giIndex == 5);
    m_prtEnabled         = (giIndex == 6);
    m_ddgiEnabled        = (giIndex == 7);
    m_gtgi.enabled       = (giIndex == 8);
    m_sdfgiPass.enabled  = (giIndex == 9);
    // RT/index 10 由 rtGiBound() 在 buildPipelineTable 中判断
    m_restirPass.enabled = (giIndex == 11);
    m_lumenEnabled       = (giIndex == 12);
    // Lumen 自动开启 VXGI multi-bounce relight
    m_vxgiRelightEnabled = (giIndex == 12);
}

void FrameRenderer::setupGiGrids(const glm::vec3& aabbMin, const glm::vec3& aabbMax) {
    glm::vec3 c = (aabbMin + aabbMax) * 0.5f;
    glm::vec3 d = aabbMax - aabbMin;

    // LPV 网格：padding 5%，cellSize = maxExtent / resolution
    {
        glm::vec3 padded = d * 1.10f;
        float maxExt = std::max({padded.x, padded.y, padded.z});
        m_lpvCellSize = maxExt / float(kLpvResolution);
        glm::vec3 half = glm::vec3(m_lpvCellSize * float(kLpvResolution) * 0.5f);
        m_lpvGridMin = c - half;
    }
    // VXGI 网格
    {
        glm::vec3 padded = d * 1.10f;
        float maxExt = std::max({padded.x, padded.y, padded.z});
        m_vxgiCellSize = maxExt / float(kVxgiResolution);
        glm::vec3 half = glm::vec3(m_vxgiCellSize * float(kVxgiResolution) * 0.5f);
        m_vxgiGridMin = c - half;
    }
    // PRT 网格（与 LPV 同分辨率，存 visibility transfer SH）
    {
        glm::vec3 padded = d * 1.10f;
        float maxExt = std::max({padded.x, padded.y, padded.z});
        m_prtCellSize = maxExt / float(kPrtResolution);
        glm::vec3 half = glm::vec3(m_prtCellSize * float(kPrtResolution) * 0.5f);
        m_prtGridMin = c - half;
    }
    m_prtBaked = false;  // scene 切换后需重新 bake

    // DDGI probe 网格
    {
        glm::vec3 padded = d * 1.05f;
        m_ddgiSpacing = glm::vec3(
            padded.x / float(DdgiResources::kProbesX - 1),
            padded.y / float(DdgiResources::kProbesY - 1),
            padded.z / float(DdgiResources::kProbesZ - 1));
        glm::vec3 half = padded * 0.5f;
        m_ddgiOrigin = c - half;
    }
    m_ddgiAtlasInited = false;
}

void FrameRenderer::bootstrapHdrPrev() {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    oneShotSubmit(*m_rhiDevice, [&](VkCommandBuffer cmd) {
        rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
        auto& img = m_rt.hdrPrev;
        rhiCmd.textureBarrier(*wrapImage(img.image(), img.format(), img.extent().width, img.extent().height),
                              rhi::TextureLayout::Undefined, rhi::TextureLayout::ShaderReadOnly);
    });
}

void FrameRenderer::bootstrapSsgiTemporal() {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    oneShotSubmit(*m_rhiDevice, [&](VkCommandBuffer cmd) {
        rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
        const Image* imgs[2] = {&m_rt.ssgi, &m_rt.ssgiPrev};
        for (auto* img : imgs) {
            rhiCmd.textureBarrier(*wrapImage(img->image(), img->format(), img->extent().width, img->extent().height),
                                  rhi::TextureLayout::Undefined, rhi::TextureLayout::ShaderReadOnly);
        }
    });
}

// ── 通用引导：将所有未初始化的纹理从 UNDEFINED 转到 SHADER_READ_ONLY ──
// 只做布局转换，不清除内容（避免要求 TRANSFER_DST usage）。
// 各 Pass 首次写入时会自行转换到正确的可写布局。
void FrameRenderer::bootstrapAllTargets() {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    oneShotSubmit(*m_rhiDevice, [&](VkCommandBuffer cmd) {
        rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
        // 辅助：UNDEFINED → SHADER_READ_ONLY 布局转换（支持多 mip）
        auto transitionFromUndefined = [&](VkImage img, VkFormat fmt, uint32_t w, uint32_t h, uint32_t mips=1) {
            if (!img) return;
            rhiCmd.textureBarrier(*wrapImage(img, fmt, w, h, mips),
                                  rhi::TextureLayout::Undefined, rhi::TextureLayout::ShaderReadOnly);
        };
        auto& rt = m_rt;
        auto ti = [&](const Image& img, uint32_t mips=1) {
            transitionFromUndefined(img.image(), img.format(), img.extent().width, img.extent().height, mips);
        };
        ti(rt.ssr); ti(rt.rsmGI); ti(rt.restir); ti(rt.rtGI); ti(rt.lumenGI); ti(rt.ssao);
        // SDFGI
        ti(m_sdfgi.seedA()); ti(m_sdfgi.seedB()); ti(m_sdfgi.udf());
        // LPV 体素网格（2 组 ping-pong × 3 通道 + gv）
        for (int g = 0; g < 2; ++g) {
            auto& gr = g ? m_lpv.next() : m_lpv.current();
            ti(gr.lpvR); ti(gr.lpvG); ti(gr.lpvB);
        }
        ti(m_lpv.gv());
        // VXGI 纹理（voxelGrid 有 mip 链，使用 image 的实际 mip 数）
        uint32_t vxMips = m_vxgi.image().image() ? m_vxgi.image().mipLevels() : 1u;
        transitionFromUndefined(m_vxgi.image().image(), m_vxgi.image().format(), m_vxgi.image().extent().width, m_vxgi.image().extent().height, vxMips);
        ti(m_vxgi.aniso()); ti(m_vxgi.relightScratch()); ti(m_vxgi.relightScratch2());
        ti(m_vxgi.sixAxisX()); ti(m_vxgi.sixAxisY()); ti(m_vxgi.sixAxisZ());
        // PRT（5 张 SH 纹理）+ DDGI（2 张 probe atlas）
        ti(m_prt.image()); ti(m_prt.imageB()); ti(m_prt.imageC());
        ti(m_prt.imageD()); ti(m_prt.imageE());
        ti(m_ddgi.irradiance()); ti(m_ddgi.distance());
    });
}

void FrameRenderer::writeTimestamp(rhi::RHICommandBuffer& cmd, uint32_t slot) {
    uint32_t base = m_currentFrameInFlight * kTimestampSlots;
    cmd.writeTimestamp(*m_timestampPool, base + slot);
}

std::unique_ptr<rhi::RHITexture> FrameRenderer::wrapImage(VkImage img, VkFormat fmt, uint32_t w, uint32_t h, uint32_t mips) const {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    return rhi::VkRHITexture::createNonOwning(vkDev, img, rhi::toRhiFormat(fmt), w, h, mips);
}

void FrameRenderer::rebuildDemoLights(const SceneCpu& cpu) {
    m_demoLights.clear();
    glm::vec3 mn=cpu.aabbMin,mx=cpu.aabbMax;
    if(glm::length(mx-mn)<1e-3f){mn=glm::vec3(-2);mx=glm::vec3(2);}
    glm::vec3 d=mx-mn;
    float fy=mn.y+d.y*0.10f,cy=mn.y+d.y*0.85f;
    glm::vec3 cs[8]={
        {mn.x+d.x*.2f,cy,mn.z+d.z*.2f},{mn.x+d.x*.8f,cy,mn.z+d.z*.2f},
        {mn.x+d.x*.2f,cy,mn.z+d.z*.8f},{mn.x+d.x*.8f,cy,mn.z+d.z*.8f},
        {mn.x+d.x*.3f,fy,mn.z+d.z*.5f},{mn.x+d.x*.7f,fy,mn.z+d.z*.5f},
        {mn.x+d.x*.5f,fy,mn.z+d.z*.3f},{mn.x+d.x*.5f,fy,mn.z+d.z*.7f},
    };
    glm::vec3 cls[8]={
        {1,.4f,.4f},{.4f,1,.4f},{.4f,.4f,1},{1,1,.5f},
        {1,.5f,1},{.5f,1,1},{1,.8f,.4f},{.8f,.4f,1},
    };
    int n=std::min(m_demoLightCount,8);
    for(int i=0;i<n;++i){
        PointLightCpu L;L.pos=cs[i];L.radius=glm::length(d)*.5f;
        L.color=cls[i];L.intensity=m_demoLightIntensity;
        m_demoLights.push_back(L);
    }
}

void FrameRenderer::setUseMeshShader(bool v) {
    m_useMeshShader = v;
    m_gbuffer.setMeshShaderEnabled(v);
    m_forward.setMeshShaderEnabled(v);
}

void FrameRenderer::applyShadowSelection(int idx) {
    if (idx < 0 || idx >= kShadowCount) idx = 1;
    m_shadow.setMethod((ShadowMethod)idx);
    // 更新 lighting 端的 shadow mask image view（shader 同一张 Image）
    if (m_rhiDevice) {
        m_lighting.bindShadowMask(m_shadow.shadowMask().view());
    }
}

void FrameRenderer::registerPipelineSteps() {
    pipeline().clear();

    // ============================
    // Phase 0: Shadow Pass（PrePass 最前面，生成 shadowMask）
    // ============================
    pipeline().addStep({
        .name = "Shadow",
        .phase = "PrePass",
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            // 确保 gDepth 在 SHADER_READ_ONLY_OPTIMAL layout（首帧从 UNDEFINED 过渡）
            cmd.textureBarrier(*wrapImage(rt().depth.image(), rt().depth.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::ShaderReadOnly);
            // 同理 gNormalRough（首帧 layout 不确定）
            cmd.textureBarrier(*wrapImage(rt().gNormalRough.image(), rt().gNormalRough.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::ShaderReadOnly);
            auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
            auto rhiIndirectBuf = rhi::VkRHIBuffer::createNonOwning(vkDev, m_boundScene.indirectBufSun, VK_WHOLE_SIZE);
            auto rhiFrameUbo = rhi::VkRHIBuffer::createNonOwning(vkDev, gbuffer().frameUboHandle(), VK_WHOLE_SIZE);
            shadow().record(cmd, rt(),
                *rhiFrameUbo,
                *m_boundScene.gpu, *rhiIndirectBuf, m_boundScene.drawCount,
                frameIndex());
        }
    });

    // ============================
    // Phase 0: RSM 几何（sun-view MRT）
    // ============================
    pipeline().addStep({
        .name = "RSM-Geometry",
        .phase = "PrePass",
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
            auto rhiIb = rhi::VkRHIBuffer::createNonOwning(vkDev, m_boundScene.indirectBufSun, VK_WHOLE_SIZE);
            rsmGeom().record(cmd, *rhiIb, m_boundScene.drawCount, *m_boundScene.gpu);
        }
    });

    // ============================
    // Step 1 诊断：用 transfer clear 替代 vkCmdBeginRendering 写 hdrColor+depth
    // ============================
    pipeline().addStep({
        .name = "Forward",
        .phase = "PrePass",
        .enabled = false,
        .timestampSlot = kTsGBuffer,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            // hdrColor → COLOR_ATTACHMENT, depth → DEPTH_ATTACHMENT
            cmd.textureBarrier(*wrapImage(rt().hdrColor.image(), rt().hdrColor.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::ColorAttachment);
            cmd.textureBarrier(*wrapImage(rt().depth.image(), rt().depth.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::DepthAttachment);

            auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
            auto rhiIndirectBuf = rhi::VkRHIBuffer::createNonOwning(vkDev, m_boundScene.indirectBuf, VK_WHOLE_SIZE);
            forward().record(cmd, rt(), *rhiIndirectBuf, m_boundScene.drawCount, *m_boundScene.gpu);

            // hdrColor COLOR_ATTACHMENT → GENERAL（匹配延迟 Lighting 输出）
            cmd.textureBarrier(*wrapImage(rt().hdrColor.image(), rt().hdrColor.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::ColorAttachment, rhi::TextureLayout::General);
            // depth DEPTH_ATTACHMENT → SR_O（匹配延迟 GBuffer 输出，供 Skybox 使用）
            cmd.textureBarrier(*wrapImage(rt().depth.image(), rt().depth.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::DepthAttachment, rhi::TextureLayout::ShaderReadOnly);

            writeTimestamp(cmd, kTsGBuffer);
            writeTimestamp(cmd, kTsAO);
            writeTimestamp(cmd, kTsVoxelGI);
            writeTimestamp(cmd, kTsLighting);
            writeTimestamp(cmd, kTsSkybox);
        }
    });

    // ============================
    // Phase 1: GBuffer prepass (graphics MRT with MSAA)
    // ============================
    pipeline().addStep({
        .name = "GBuffer",
        .phase = "PrePass",
        .timestampSlot = kTsGBuffer,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            // MSAA images → attachment layout
            auto toColorAttach = [&](const Image& img) {
                cmd.textureBarrier(*wrapImage(img.image(), img.format(), img.extent().width, img.extent().height),
                                   rhi::TextureLayout::Undefined, rhi::TextureLayout::ColorAttachment);
            };
            toColorAttach(rt().gAlbedoMetalMs); toColorAttach(rt().gNormalRoughMs); toColorAttach(rt().gEmissiveAOMs);
            cmd.textureBarrier(*wrapImage(rt().depthMs.image(), rt().depthMs.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::DepthAttachment);

            // SS resolve targets → attachment layout
            toColorAttach(rt().gAlbedoMetal); toColorAttach(rt().gNormalRough); toColorAttach(rt().gEmissiveAO);
            cmd.textureBarrier(*wrapImage(rt().depth.image(), rt().depth.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::DepthAttachment);

            auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
            auto rhiIndirectBuf = rhi::VkRHIBuffer::createNonOwning(vkDev, m_boundScene.indirectBuf, VK_WHOLE_SIZE);
            gbuffer().record(cmd, rt(), *rhiIndirectBuf, m_boundScene.drawCount, *m_boundScene.gpu);

            // Resolved GBuffer → SHADER_READ_ONLY for downstream compute
            auto toSampled = [&](const Image& img) {
                cmd.textureBarrier(*wrapImage(img.image(), img.format(), img.extent().width, img.extent().height),
                                   rhi::TextureLayout::ColorAttachment, rhi::TextureLayout::ShaderReadOnly);
            };
            toSampled(rt().gAlbedoMetal); toSampled(rt().gNormalRough); toSampled(rt().gEmissiveAO);
            cmd.textureBarrier(*wrapImage(rt().depth.image(), rt().depth.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::DepthAttachment, rhi::TextureLayout::ShaderReadOnly);

            writeTimestamp(cmd, kTsGBuffer);
        }
    });

    // ============================
    // Phase 1.5: AO (SSAO/GTAO/None, 互斥)
    // ============================
    pipeline().addStep({
        .name = "AO-SSAO",
        .phase = "AO",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().ssao.image(), rt().ssao.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
            ssao().record(cmd, rt(),
                m_frameState.proj, glm::inverse(m_frameState.proj), m_frameState.view);
            cmd.textureBarrier(*wrapImage(rt().ssao.image(), rt().ssao.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    pipeline().addStep({
        .name = "AO-GTAO",
        .phase = "AO",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().ssao.image(), rt().ssao.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
            gtao().record(cmd, rt(),
                m_frameState.proj, m_frameState.view);
            cmd.textureBarrier(*wrapImage(rt().ssao.image(), rt().ssao.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    pipeline().addStep({
        .name = "AO-Clear",
        .phase = "AO",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().ssao.image(), rt().ssao.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::TransferDst);
            cmd.clearColor(*wrapImage(rt().ssao.image(), rt().ssao.format(), rt().extent.width, rt().extent.height),
                           1.0f, 1.0f, 1.0f, 1.0f);
            cmd.textureBarrier(*wrapImage(rt().ssao.image(), rt().ssao.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    // ============================
    // Phase 1.6: SSR
    // ============================
    pipeline().addStep({
        .name = "SSR",
        .phase = "AO",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().ssr.image(), rt().ssr.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
            ssr().record(cmd, rt());
            cmd.textureBarrier(*wrapImage(rt().ssr.image(), rt().ssr.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    pipeline().addStep({
        .name = "SSR-Clear",
        .phase = "AO",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().ssr.image(), rt().ssr.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::TransferDst);
            cmd.clearColor(*wrapImage(rt().ssr.image(), rt().ssr.format(), rt().extent.width, rt().extent.height),
                           0.0f, 0.0f, 0.0f, 0.0f);
            cmd.textureBarrier(*wrapImage(rt().ssr.image(), rt().ssr.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    // ============================
    // Phase 1.7: ScreenGI (SSGI/GTGI, 互斥)
    // ============================
    pipeline().addStep({
        .name = "ScreenGI",
        .phase = "AO",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            // Copy ssgi → ssgiPrev for temporal history
            cmd.textureBarrier(*wrapImage(rt().ssgi.image(), rt().ssgi.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::ShaderReadOnly, rhi::TextureLayout::TransferSrc);
            cmd.textureBarrier(*wrapImage(rt().ssgiPrev.image(), rt().ssgiPrev.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::ShaderReadOnly, rhi::TextureLayout::TransferDst);

            cmd.copyTexture(*wrapImage(rt().ssgi.image(), rt().ssgi.format(), rt().extent.width, rt().extent.height),
                           *wrapImage(rt().ssgiPrev.image(), rt().ssgiPrev.format(), rt().extent.width, rt().extent.height));

            // ssgiPrev → SHADER_READ_ONLY for sampling
            cmd.textureBarrier(*wrapImage(rt().ssgiPrev.image(), rt().ssgiPrev.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
            // ssgi → GENERAL for writing new value
            cmd.textureBarrier(*wrapImage(rt().ssgi.image(), rt().ssgi.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::TransferSrc, rhi::TextureLayout::General);

            if (ssgi().enabled) ssgi().record(cmd, rt());
            else                gtgi().record(cmd, rt());

            cmd.textureBarrier(*wrapImage(rt().ssgi.image(), rt().ssgi.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    pipeline().addStep({
        .name = "ScreenGI-Clear",
        .phase = "AO",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().ssgi.image(), rt().ssgi.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::TransferDst);
            cmd.clearColor(*wrapImage(rt().ssgi.image(), rt().ssgi.format(), rt().extent.width, rt().extent.height),
                           0.0f, 0.0f, 0.0f, 0.0f);
            cmd.textureBarrier(*wrapImage(rt().ssgi.image(), rt().ssgi.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    // === AO/SS 结束 timestamp ===
    pipeline().addStep({
        .name = "TS-AO",
        .phase = "AO",
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            writeTimestamp(cmd, kTsAO);
        }
    });

    // ============================
    // Phase 1.83: VXGI voxelize → inject → mipmap → aniso → relight → 6axis
    // ============================
    pipeline().addStep({
        .name = "VXGI-Chain",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            auto& vxgiImg = vxgi().image();
            // 1. Clear entire mip chain to 0
            cmd.textureBarrier(*wrapImage(vxgiImg.image(), vxgiImg.format(), vxgiImg.extent().width, vxgiImg.extent().height, vxgi().mipLevels()),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::TransferDst);
            cmd.clearColor(*wrapImage(vxgiImg.image(), vxgiImg.format(), vxgiImg.extent().width, vxgiImg.extent().height, vxgi().mipLevels()),
                           0.0f, 0.0f, 0.0f, 0.0f);
            cmd.textureBarrier(*wrapImage(vxgiImg.image(), vxgiImg.format(), vxgiImg.extent().width, vxgiImg.extent().height, vxgi().mipLevels()),
                               rhi::TextureLayout::TransferDst, rhi::TextureLayout::General);

            // 2. Voxelize: scatter all primitives to mip 0
            vxgiVoxelize().record(cmd, *m_boundScene.cpu, *m_boundScene.gpu,
                vxgiGridMin(), vxgiCellSize(), kVxgiResolution);

            // 3. Inject: RSM flux → voxel mip 0 RGB
            cmd.textureBarrier(*wrapImage(vxgiImg.image(), vxgiImg.format(), vxgiImg.extent().width, vxgiImg.extent().height, 1),
                               rhi::TextureLayout::General, rhi::TextureLayout::General);
            vxgiInject().record(cmd, kVxgiResolution, vxgiGridMin(), vxgiCellSize());

            // 4. Mipmap: iterate src mip i → dst mip i+1
            vxgiMipmap().record(cmd, vxgi());

            // 5. Final mip → SHADER_READ_ONLY
            cmd.textureBarrier(*wrapImage(vxgiImg.image(), vxgiImg.format(), vxgiImg.extent().width, vxgiImg.extent().height, vxgi().mipLevels()),
                               rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);

            // 6. Aniso alpha mipchain: UNDEFINED → SHADER_READ_ONLY
            auto& anisoImg = vxgi().aniso();
            cmd.textureBarrier(*wrapImage(anisoImg.image(), anisoImg.format(), anisoImg.extent().width, anisoImg.extent().height, vxgi().mipLevels()),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::ShaderReadOnly);
            vxgiAniso().record(cmd, vxgi());
        }
    });

    // VXGI Relight (multi-bounce, within VXGI chain)
    pipeline().addStep({
        .name = "VXGI-Relight",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            int bounces = lumenEnabled() ? 3 : 1;
            uint32_t res = kVxgiResolution;

            auto transImg = [&](const Image& img, rhi::TextureLayout oldL, rhi::TextureLayout newL) {
                cmd.textureBarrier(*wrapImage(img.image(), img.format(), img.extent().width, img.extent().height),
                                   oldL, newL);
            };

            auto blitScratchToVoxel = [&](const Image& srcImg) {
                transImg(vxgi().image(), rhi::TextureLayout::ShaderReadOnly, rhi::TextureLayout::TransferDst);
                transImg(srcImg, rhi::TextureLayout::General, rhi::TextureLayout::TransferSrc);
                cmd.copyTexture(*wrapImage(srcImg.image(), srcImg.format(), res, res),
                               *wrapImage(vxgi().image().image(), vxgi().image().format(), res, res));
                transImg(vxgi().image(), rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
            };

            // Bounce 1: read voxelGrid → write scratch
            transImg(vxgi().relightScratch(), rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
            vxgiRelight().record(cmd, vxgiRelight().voxelSet(), kVxgiResolution,
                vxgi().mipLevels(), vxgiCellSize(), vxgiGridMin(),
                vxgiRelightStrength());

            if (bounces >= 2) {
                transImg(vxgi().relightScratch(), rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
                transImg(vxgi().relightScratch2(), rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
                // Bounce 2: read scratch → write scratch2
                vxgiRelight().record(cmd, vxgiRelight().pingSet0(), kVxgiResolution,
                    vxgi().mipLevels(), vxgiCellSize(), vxgiGridMin(),
                    vxgiRelightStrength());

                if (bounces >= 3) {
                    transImg(vxgi().relightScratch2(), rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
                    transImg(vxgi().relightScratch(), rhi::TextureLayout::ShaderReadOnly, rhi::TextureLayout::General);
                    // Bounce 3: read scratch2 → write scratch
                    vxgiRelight().record(cmd, vxgiRelight().pingSet1(), kVxgiResolution,
                        vxgi().mipLevels(), vxgiCellSize(), vxgiGridMin(),
                        vxgiRelightStrength());
                    blitScratchToVoxel(vxgi().relightScratch());
                } else {
                    blitScratchToVoxel(vxgi().relightScratch2());
                }
            } else {
                blitScratchToVoxel(vxgi().relightScratch());
            }
        }
    });

    // VXGI 6-axis resolve (Lumen mode only)
    pipeline().addStep({
        .name = "VXGI-6Axis",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            rhi::TextureLayout axisOldL = lumenAtlasInited()
                ? rhi::TextureLayout::ShaderReadOnly : rhi::TextureLayout::Undefined;

            auto transAxisToGeneral = [&](const Image& img) {
                cmd.textureBarrier(*wrapImage(img.image(), img.format(), img.extent().width, img.extent().height),
                                   axisOldL, rhi::TextureLayout::General);
            };
            transAxisToGeneral(vxgi().sixAxisX()); transAxisToGeneral(vxgi().sixAxisY()); transAxisToGeneral(vxgi().sixAxisZ());

            vxgi6Axis().record(cmd, kVxgiResolution, vxgi().mipLevels(),
                vxgiCellSize(), vxgiGridMin(), vxgiRelightStrength());

            auto transAxisToSRO = [&](const Image& img) {
                cmd.textureBarrier(*wrapImage(img.image(), img.format(), img.extent().width, img.extent().height),
                                   rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
            };
            transAxisToSRO(vxgi().sixAxisX()); transAxisToSRO(vxgi().sixAxisY()); transAxisToSRO(vxgi().sixAxisZ());
        }
    });

    // VXGI bootstrap: when all consumers are off, transition voxel grid to SR_O
    pipeline().addStep({
        .name = "VXGI-Bootstrap",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            auto& vxgiImg = vxgi().image();
            cmd.textureBarrier(*wrapImage(vxgiImg.image(), vxgiImg.format(), vxgiImg.extent().width, vxgiImg.extent().height, vxgi().mipLevels()),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::ShaderReadOnly);
            // aniso too
            auto& anisoImg = vxgi().aniso();
            cmd.textureBarrier(*wrapImage(anisoImg.image(), anisoImg.format(), anisoImg.extent().width, anisoImg.extent().height, vxgi().mipLevels()),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    // ============================
    // Phase 1.835: SDFGI
    // ============================
    pipeline().addStep({
        .name = "SDFGI",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            if (!sdfgiBootstrapped()) {
                auto bootstrapToGeneral = [&](const Image& img) {
                    cmd.textureBarrier(*wrapImage(img.image(), img.format(), img.extent().width, img.extent().height),
                                       rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
                };
                bootstrapToGeneral(sdfgi().seedA()); bootstrapToGeneral(sdfgi().seedB()); bootstrapToGeneral(sdfgi().udf());
                sdfgiBootstrapped() = true;
            }
            cmd.textureBarrier(*wrapImage(rt().ssgi.image(), rt().ssgi.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::ShaderReadOnly, rhi::TextureLayout::General);

            sdfgiPass().record(cmd, sdfgi(), rt(), frameIndex(),
                sdfgiPass().seedThreshold, sdfgiPass().maxDistCells,
                (uint32_t)sdfgiPass().numRays,
                (uint32_t)sdfgiPass().maxSteps,
                sdfgiPass().rayMaxCells, sdfgiPass().hitEpsCells);

            cmd.textureBarrier(*wrapImage(rt().ssgi.image(), rt().ssgi.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    // ============================
    // Phase 1.836: RT GI
    // ============================
    pipeline().addStep({
        .name = "RTGI",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().rtGI.image(), rt().rtGI.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
            rtGi().record(cmd, rt());
            cmd.textureBarrier(*wrapImage(rt().rtGI.image(), rt().rtGI.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    pipeline().addStep({
        .name = "RTGI-Clear",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().rtGI.image(), rt().rtGI.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::TransferDst);
            cmd.clearColor(*wrapImage(rt().rtGI.image(), rt().rtGI.format(), rt().extent.width, rt().extent.height),
                           0.0f, 0.0f, 0.0f, 0.0f);
            cmd.textureBarrier(*wrapImage(rt().rtGI.image(), rt().rtGI.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    // ============================
    // Phase 1.837: ReSTIR DI
    // ============================
    pipeline().addStep({
        .name = "ReSTIR",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            restir().updateLights(demoLights());
            if (!restirBootstrapped()) {
                auto bootstrapToGeneral = [&](const Image& img) {
                    cmd.textureBarrier(*wrapImage(img.image(), img.format(), img.extent().width, img.extent().height),
                                       rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
                };
                bootstrapToGeneral(restir().reservoirA()); bootstrapToGeneral(restir().reservoirB());
                restirBootstrapped() = true;
            }
            rhi::TextureLayout restirOld = restirOutInited()
                ? rhi::TextureLayout::ShaderReadOnly : rhi::TextureLayout::Undefined;
            cmd.textureBarrier(*wrapImage(rt().restir.image(), rt().restir.format(), rt().extent.width, rt().extent.height),
                               restirOld, rhi::TextureLayout::General);
            restirOutInited() = true;

            uint32_t numLights = (uint32_t)demoLights().size();
            bool useRtVis = rtSupported() && rtGiBound();
            restirPass().record(cmd, restir(), rt(),
                numLights,
                (uint32_t)restirPass().numCandidates,
                (uint32_t)restirPass().numNeighbors,
                restirPass().spatialRadius,
                (uint32_t)restirPass().shadowSteps,
                restirPass().intensityScale,
                frameIndex(),
                useRtVis);

            cmd.textureBarrier(*wrapImage(rt().restir.image(), rt().restir.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    pipeline().addStep({
        .name = "ReSTIR-Clear",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            rhi::TextureLayout restirOld = restirOutInited()
                ? rhi::TextureLayout::ShaderReadOnly : rhi::TextureLayout::Undefined;
            cmd.textureBarrier(*wrapImage(rt().restir.image(), rt().restir.format(), rt().extent.width, rt().extent.height),
                               restirOld, rhi::TextureLayout::TransferDst);
            cmd.clearColor(*wrapImage(rt().restir.image(), rt().restir.format(), rt().extent.width, rt().extent.height),
                           0.0f, 0.0f, 0.0f, 0.0f);
            cmd.textureBarrier(*wrapImage(rt().restir.image(), rt().restir.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
            restirOutInited() = true;
        }
    });

    // ============================
    // Phase 1.84: DDGI
    // ============================
    pipeline().addStep({
        .name = "DDGI",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            auto barrierAtlas = [&](const Image& img, rhi::TextureLayout oldL, rhi::TextureLayout newL) {
                cmd.textureBarrier(*wrapImage(img.image(), img.format(), img.extent().width, img.extent().height),
                                   oldL, newL);
            };

            rhi::TextureLayout oldAtlasL = ddgiAtlasInited()
                ? rhi::TextureLayout::ShaderReadOnly : rhi::TextureLayout::Undefined;

            barrierAtlas(ddgi().irradiance(), oldAtlasL, rhi::TextureLayout::General);
            barrierAtlas(ddgi().distance(), oldAtlasL, rhi::TextureLayout::General);

            float jitterRot = float((frameIndex() % 360) * 0.0174532925);
            ddgiPass().record(cmd, ddgi(), ddgiOrigin(), ddgiSpacing(),
                vxgiGridMin(), vxgiCellSize(), kVxgiResolution,
                jitterRot, frameIndex());

            barrierAtlas(ddgi().irradiance(), rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
            barrierAtlas(ddgi().distance(), rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
            ddgiAtlasInited() = true;
        }
    });

    pipeline().addStep({
        .name = "DDGI-Bootstrap",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            auto barrierAtlas = [&](const Image& img) {
                cmd.textureBarrier(*wrapImage(img.image(), img.format(), img.extent().width, img.extent().height),
                                   rhi::TextureLayout::Undefined, rhi::TextureLayout::ShaderReadOnly);
            };
            barrierAtlas(ddgi().irradiance()); barrierAtlas(ddgi().distance());
            ddgiAtlasInited() = true;
        }
    });

    // ============================
    // Phase 1.845: Lumen-lite
    // ============================
    pipeline().addStep({
        .name = "Lumen-DebugClear",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            rhi::TextureLayout oldL = lumenOutInited()
                ? rhi::TextureLayout::ShaderReadOnly : rhi::TextureLayout::Undefined;
            cmd.textureBarrier(*wrapImage(rt().lumenGI.image(), rt().lumenGI.format(), rt().extent.width, rt().extent.height),
                               oldL, rhi::TextureLayout::TransferDst);
            cmd.clearColor(*wrapImage(rt().lumenGI.image(), rt().lumenGI.format(), rt().extent.width, rt().extent.height),
                           0.3f, 0.3f, 0.3f, 1.0f);
            cmd.textureBarrier(*wrapImage(rt().lumenGI.image(), rt().lumenGI.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
            lumenOutInited() = true;
        }
    });

    pipeline().addStep({
        .name = "Lumen-Probe",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            if (!lumenProbeInited()) {
                lumenProbe().init(*rhiDevice());
                lumenProbe().bindResources(lumen(), rtAS(), *m_boundScene.gpu,
                                                vxgi(), rt(), gbuffer().frameUboHandle(),
                                                vxgiSixAxisInited());
                lumenProbeInited() = true;
            }
            // Transition probe + filtered atlas to GENERAL
            {
                rhi::TextureLayout oldL = lumenAtlasInited()
                    ? rhi::TextureLayout::ShaderReadOnly : rhi::TextureLayout::Undefined;
                auto transToGeneral = [&](const Image& img) {
                    cmd.textureBarrier(*wrapImage(img.image(), img.format(), img.extent().width, img.extent().height),
                                       oldL, rhi::TextureLayout::General);
                };
                transToGeneral(lumen().probeAtlas()); transToGeneral(lumen().filteredAtlas());
                lumenAtlasInited() = true;
            }
            lumenProbe().record(cmd, lumen(), frameIndex(),
                                 lumenDebugMode() >= 3 ? (uint32_t)lumenDebugMode() - 1u
                                                       : (vxgiSixAxisInited() ? 1u : 0u));

            // ProbeAtlas GENERAL → SR_O for filter
            cmd.textureBarrier(*wrapImage(lumen().probeAtlas().image(), lumen().probeAtlas().format(), lumen().atlasWidth(), lumen().atlasHeight()),
                               rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    pipeline().addStep({
        .name = "Lumen-Filter",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            if (!lumenFilterInited()) {
                cmd.textureBarrier(*wrapImage(lumen().prevAtlas().image(), lumen().prevAtlas().format(), lumen().atlasWidth(), lumen().atlasHeight()),
                                   rhi::TextureLayout::Undefined, rhi::TextureLayout::ShaderReadOnly);

                lumenFilter().init(*rhiDevice());
                lumenFilter().bindResources(lumen(), rt(), gbuffer().frameUboHandle());
                lumenFilterInited() = true;
            }
            lumenFilter().record(cmd, lumen(), rt());

            // Copy filteredAtlas → prevAtlas for next frame
            auto imgBarrier = [&](const Image& img, rhi::TextureLayout oldL, rhi::TextureLayout newL) {
                cmd.textureBarrier(*wrapImage(img.image(), img.format(), img.extent().width, img.extent().height),
                                   oldL, newL);
            };

            imgBarrier(lumen().filteredAtlas(), rhi::TextureLayout::General, rhi::TextureLayout::TransferSrc);
            imgBarrier(lumen().prevAtlas(), rhi::TextureLayout::ShaderReadOnly, rhi::TextureLayout::TransferDst);

            cmd.copyTexture(*wrapImage(lumen().filteredAtlas().image(), lumen().filteredAtlas().format(), lumen().atlasWidth(), lumen().atlasHeight()),
                           *wrapImage(lumen().prevAtlas().image(), lumen().prevAtlas().format(), lumen().atlasWidth(), lumen().atlasHeight()));

            imgBarrier(lumen().filteredAtlas(), rhi::TextureLayout::TransferSrc, rhi::TextureLayout::ShaderReadOnly);
            imgBarrier(lumen().prevAtlas(), rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    pipeline().addStep({
        .name = "Lumen-Gather",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            if (!lumenGatherInited()) {
                lumenGather().init(*rhiDevice());
                lumenGather().bindResources(lumen(), rt(),
                                                 gbuffer().frameUboHandle(), true);
                lumenGatherInited() = true;
            }
            {
                rhi::TextureLayout oldL = lumenOutInited()
                    ? rhi::TextureLayout::ShaderReadOnly : rhi::TextureLayout::Undefined;
                cmd.textureBarrier(*wrapImage(rt().lumenGI.image(), rt().lumenGI.format(), rt().extent.width, rt().extent.height),
                                   oldL, rhi::TextureLayout::General);
            }
            lumenGather().record(cmd, lumen(), rt(), (uint32_t)lumenDebugMode());

            cmd.textureBarrier(*wrapImage(rt().lumenGI.image(), rt().lumenGI.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
            lumenOutInited() = true;
        }
    });

    pipeline().addStep({
        .name = "Lumen-Clear",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            rhi::TextureLayout oldL = lumenOutInited()
                ? rhi::TextureLayout::ShaderReadOnly : rhi::TextureLayout::Undefined;
            cmd.textureBarrier(*wrapImage(rt().lumenGI.image(), rt().lumenGI.format(), rt().extent.width, rt().extent.height),
                               oldL, rhi::TextureLayout::TransferDst);
            cmd.clearColor(*wrapImage(rt().lumenGI.image(), rt().lumenGI.format(), rt().extent.width, rt().extent.height),
                           0.0f, 0.0f, 0.0f, 0.0f);
            cmd.textureBarrier(*wrapImage(rt().lumenGI.image(), rt().lumenGI.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
            lumenOutInited() = true;
        }
    });

    // ============================
    // Phase 1.85: LPV inject + propagate
    // ============================
    pipeline().addStep({
        .name = "LPV",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            lpvInject().record(cmd, kLpvResolution, lpvGridMin(), lpvCellSize());

            cmd.textureBarrier(*wrapImage(lpv().gv().image(), lpv().gv().format(), kLpvResolution, kLpvResolution),
                               rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);

            auto barrierLpv = [&](const LpvGrid& g, rhi::TextureLayout oldL, rhi::TextureLayout newL) {
                const Image* imgs[3] = {&g.lpvR, &g.lpvG, &g.lpvB};
                for (auto* img : imgs) {
                    cmd.textureBarrier(*wrapImage(img->image(), img->format(), img->extent().width, img->extent().height),
                                       oldL, newL);
                }
            };

            int propIter = lpvProp().iterations & ~1;
            for (int it = 0; it < propIter; ++it) {
                LpvGrid& src = lpv().current();
                LpvGrid& dst = lpv().next();

                barrierLpv(src, rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
                barrierLpv(dst, rhi::TextureLayout::Undefined, rhi::TextureLayout::General);

                lpvProp().record(cmd, lpv().curIdx(),
                                 kLpvResolution, lpvProp().occlusionAmplifier,
                                 lpvProp().gvOcclusionStrength);
                lpv().swap();
            }

            barrierLpv(lpv().current(), rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    pipeline().addStep({
        .name = "LPV-Bootstrap",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            const Image* imgs[3] = {&lpv().current().lpvR, &lpv().current().lpvG, &lpv().current().lpvB};
            for (auto* img : imgs) {
                cmd.textureBarrier(*wrapImage(img->image(), img->format(), img->extent().width, img->extent().height),
                                   rhi::TextureLayout::Undefined, rhi::TextureLayout::ShaderReadOnly);
            }
        }
    });

    // ============================
    // Phase 1.8: RSM sample
    // ============================
    pipeline().addStep({
        .name = "RSM-Sample",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().rsmGI.image(), rt().rsmGI.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
            rsmSample().record(cmd, rt());
            cmd.textureBarrier(*wrapImage(rt().rsmGI.image(), rt().rsmGI.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    pipeline().addStep({
        .name = "RSM-Clear",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().rsmGI.image(), rt().rsmGI.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::TransferDst);
            cmd.clearColor(*wrapImage(rt().rsmGI.image(), rt().rsmGI.format(), rt().extent.width, rt().extent.height),
                           0.0f, 0.0f, 0.0f, 0.0f);
            cmd.textureBarrier(*wrapImage(rt().rsmGI.image(), rt().rsmGI.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
        }
    });

    // ============================
    // NDGI (Neural Dynamic GI) —— 探针光线追踪收集训练样本
    // ============================
    pipeline().addStep({
        .name = "NDGI",
        .phase = "GI",
        .enabled = false,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            // 延迟到第一帧：此时 bindResources 已执行，descriptor 已就绪
            if (!ndgiInited()) {
                ndgiPass().initWeights(cmd);
                ndgiInited() = true;
            }
            ndgiPass().record(cmd, ndgi(), frameIndex(),
                ddgiOrigin(), ddgiSpacing());
            ndgiPass().recordTraining(cmd, ndgi(), frameIndex());
        }
    });

    // === GI 结束 timestamp ===
    pipeline().addStep({
        .name = "TS-GI",
        .phase = "GI",
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            writeTimestamp(cmd, kTsVoxelGI);
        }
    });

    // ============================
    // Phase 2: Lighting (compute)
    // ============================
    pipeline().addStep({
        .name = "Lighting",
        .phase = "Shading",
        .timestampSlot = kTsLighting,
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().hdrColor.image(), rt().hdrColor.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::Undefined, rhi::TextureLayout::General);
            lighting().record(cmd, rt());
            writeTimestamp(cmd, kTsLighting);
        }
    });

    // ============================
    // Phase 3: Skybox (graphics)
    // ============================
    pipeline().addStep({
        .name = "Skybox",
        .phase = "Shading",
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().hdrColor.image(), rt().hdrColor.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::General, rhi::TextureLayout::ColorAttachment);
            cmd.textureBarrier(*wrapImage(rt().depth.image(), rt().depth.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::ShaderReadOnly, rhi::TextureLayout::DepthAttachment);
            skybox().record(cmd, rt());
        }
    });

    // ============================
    // Phase 3.5: Copy hdrColor → hdrPrev
    // ============================
    pipeline().addStep({
        .name = "Copy-hdrPrev",
        .phase = "Shading",
        .recordRHI = [this](rhi::RHICommandBuffer& cmd) {
            cmd.textureBarrier(*wrapImage(rt().hdrColor.image(), rt().hdrColor.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::ColorAttachment, rhi::TextureLayout::TransferSrc);
            cmd.textureBarrier(*wrapImage(rt().hdrPrev.image(), rt().hdrPrev.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::ShaderReadOnly, rhi::TextureLayout::TransferDst);
            cmd.copyTexture(*wrapImage(rt().hdrColor.image(), rt().hdrColor.format(), rt().extent.width, rt().extent.height),
                           *wrapImage(rt().hdrPrev.image(), rt().hdrPrev.format(), rt().extent.width, rt().extent.height));
            cmd.textureBarrier(*wrapImage(rt().hdrPrev.image(), rt().hdrPrev.format(), rt().extent.width, rt().extent.height),
                               rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);

            writeTimestamp(cmd, kTsSkybox);
        }
    });
}


void FrameRenderer::buildPipelineTable(const PipelineConfig& cfg) {
    bool fwd = cfg.forwardMode;

    // 前向/延迟核心路径切换
    m_pipeline.setEnabled("RSM-Geometry", true);
    m_pipeline.setEnabled("GBuffer",  !fwd);
    m_pipeline.setEnabled("Lighting", !fwd);
    m_pipeline.setEnabled("Forward",  fwd);
    m_pipeline.setEnabled("Skybox",   true);
    m_pipeline.setEnabled("Copy-hdrPrev", true);

    // AO 互斥：SSAO / GTAO / Clear
    m_pipeline.setEnabled("AO-SSAO", !fwd && cfg.aoMethod == 1);
    m_pipeline.setEnabled("AO-GTAO", !fwd && cfg.aoMethod == 2);
    m_pipeline.setEnabled("AO-Clear", !fwd && cfg.aoMethod == 0);

    // SSR
    m_pipeline.setEnabled("SSR", !fwd && m_ssr.enabled);
    m_pipeline.setEnabled("SSR-Clear", !fwd && !m_ssr.enabled);

    // ScreenGI (SSGI/GTGI)
    bool screenGiOn = !fwd && (m_ssgi.enabled || m_gtgi.enabled);
    m_pipeline.setEnabled("ScreenGI", screenGiOn);
    m_pipeline.setEnabled("ScreenGI-Clear", !screenGiOn);

    // VXGI chain
    bool needVoxelGrid = !fwd && (m_vxgiEnabled || m_ddgiEnabled || m_sdfgiPass.enabled
                       || m_lumenEnabled || m_restirPass.enabled);
    m_pipeline.setEnabled("VXGI-Chain", needVoxelGrid);
    m_pipeline.setEnabled("VXGI-Bootstrap", !needVoxelGrid);
    m_pipeline.setEnabled("VXGI-Relight", m_vxgiRelightEnabled && needVoxelGrid);
    m_pipeline.setEnabled("VXGI-6Axis", m_lumenEnabled && m_vxgiSixAxisInited && needVoxelGrid);

    // SDFGI
    m_pipeline.setEnabled("SDFGI", !fwd && m_sdfgiPass.enabled);

    // RT GI
    bool rtGiActive = m_rtGiBound && cfg.giIndex == 10;
    m_pipeline.setEnabled("RTGI", !fwd && rtGiActive);
    m_pipeline.setEnabled("RTGI-Clear", !fwd && m_rtGiInited && !rtGiActive);

    // ReSTIR DI
    m_pipeline.setEnabled("ReSTIR", !fwd && m_restirPass.enabled);
    m_pipeline.setEnabled("ReSTIR-Clear", !fwd && !m_restirPass.enabled);

    // DDGI
    m_pipeline.setEnabled("DDGI", !fwd && m_ddgiEnabled);
    m_pipeline.setEnabled("DDGI-Bootstrap", !fwd && !m_ddgiEnabled);
    m_pipeline.setEnabled("NDGI", m_ndgiEnabled && m_rtSupported);

    // Lumen-lite
    bool lumenActive = !fwd && m_lumenEnabled;
    m_pipeline.setEnabled("Lumen-Probe", lumenActive && m_lumenDebugMode != 5);
    m_pipeline.setEnabled("Lumen-Filter", lumenActive && m_lumenDebugMode != 5);
    m_pipeline.setEnabled("Lumen-Gather", lumenActive && m_lumenDebugMode != 5);
    m_pipeline.setEnabled("Lumen-DebugClear", lumenActive && m_lumenDebugMode == 5);
    m_pipeline.setEnabled("Lumen-Clear", !fwd && !m_lumenEnabled);

    // LPV
    m_pipeline.setEnabled("LPV", !fwd && m_lpvEnabled);
    m_pipeline.setEnabled("LPV-Bootstrap", !fwd && !m_lpvEnabled);

    // RSM Sample
    m_pipeline.setEnabled("RSM-Sample", !fwd && m_rsmSample.enabled);
    m_pipeline.setEnabled("RSM-Clear", !fwd && !m_rsmSample.enabled);

    m_pipeline.build();
}

// ──────────────────────────────────────────────────────────────────
// Debug: 保存 GBuffer 渲染目标到 PNG 文件
// ──────────────────────────────────────────────────────────────────
void FrameRenderer::debugDumpGBuffer(Device& d, VkCommandPool pool) {
    auto& rt = m_rt;
    uint32_t w = rt.extent.width, h = rt.extent.height;

    DebugDump dump;
    dump.init(d, w, h);

    // 辅助：dump 单个附件（transition→copy→transition back）
    // 注：使用 GENERAL layout 避免要求 TRANSFER_SRC usage
    auto capture = [&](VkImage img, VkFormat fmt, VkImageAspectFlags aspect, const char* name) {
        oneShotSubmit(*m_rhiDevice, [&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            b.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            b.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.oldLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            b.image         = img;
            b.subresourceRange = {aspect, 0, 1, 0, 1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &di);

            dump.recordCopy(cmd, img, fmt, w, h);

            // Transition back
            b.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
            b.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            b.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier2(cmd, &di);
        });
        dump.savePng(std::string("debug_dump/") + name, fmt, w, h);
    };

    capture(rt.gAlbedoMetal.image(), VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_ASPECT_COLOR_BIT, "gbuffer_albedo.png");
    capture(rt.gNormalRough.image(), VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_ASPECT_COLOR_BIT, "gbuffer_normal.png");
    capture(rt.gEmissiveAO.image(), VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_ASPECT_COLOR_BIT, "gbuffer_emissive.png");
    capture(rt.depth.image(), VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_ASPECT_DEPTH_BIT, "gbuffer_depth.png");

    dump.destroy();
    std::printf("[debug_dump] GBuffer targets saved to debug_dump/\n");
}

} // namespace somegi
