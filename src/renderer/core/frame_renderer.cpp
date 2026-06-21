#include "renderer/core/frame_renderer.h"
#include "core/device.h"
#include "rhi/vulkan/vk_device.h"  // VkRHIDevice shared-handle constructor
#include "rhi/vulkan/vk_command.h" // VkRHICommandBuffer for recordRHI callbacks
#include "rhi/vulkan/vk_buffer.h"  // VkRHIBuffer for non-owning buffer wrappers
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
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = kFramesInFlight * kTimestampSlots;
        VK_CHECK(vkCreateQueryPool(d.device(), &qpci, nullptr, &m_timestampPool));
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
    m_rsmGeom.init(d, *m_rhiDevice, 128);
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
    oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, m_ddgi.probeStates().handle(), 0, VK_WHOLE_SIZE, 1u);
    });
    m_ddgiPass.init(*m_rhiDevice);
    m_ddgiPass.bindResources(m_ddgi, m_vxgi);

    std::printf("[init] ndgi resources + pass...\n");
    m_ndgi.create(d);
    m_ndgiPass.init(*m_rhiDevice, rtSupported);
    m_ndgiInited = false;

    std::printf("[init] shadow pass...\n");
    m_shadow.init(d, *m_rhiDevice, {2048, 2048}, extent);

    std::printf("[init] lighting pass...\n");
    m_lighting.init(d, *m_rhiDevice);
    m_lighting.bindShadowMask(d, m_shadow.shadowMask().view());
    m_lighting.bindFrame(d, m_rt, m_gbuffer.frameUboHandle(),
                         m_lpv.current(), m_vxgi, m_prt, m_ddgi,
                         m_ddgi.probeStates().handle());
    m_lighting.setNdgiWeights(d,
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
    m_skybox.init(d, *m_rhiDevice, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_D32_SFLOAT);

    std::printf("[init] tonemap pass...\n");
    // linearSampler comes from SceneGpu — caller sets after scene load
    std::printf("[init] aa passes...\n");
    m_taa.init(*m_rhiDevice);
    m_smaa.init(d, *m_rhiDevice, extent);
    std::printf("[init] imgui pass...\n");

    m_cullPass.init(*m_rhiDevice, 4096);
    m_hizPass.init(d, *m_rhiDevice, extent);
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
    if (m_timestampPool) vkDestroyQueryPool(m_device->device(), m_timestampPool, nullptr);
    m_device = nullptr;
}

void FrameRenderer::onResize(Device& d, VkExtent2D newExtent,
                              VkSampleCountFlagBits msaaSamples,
                              VkFormat /*swapchainFmt*/, GLFWwindow* /*window*/) {
    m_rt.destroy();
    m_rt.create(d, newExtent, msaaSamples);
    m_lighting.bindFrame(d, m_rt, m_gbuffer.frameUboHandle(),
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
    oneShotSubmit(*m_device, m_pool, [&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask=VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT; b.srcAccessMask=0;
        b.dstStageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT; b.dstAccessMask=VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.image=m_rt.hdrPrev.image(); b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO}; di.imageMemoryBarrierCount=1; di.pImageMemoryBarriers=&b;
        vkCmdPipelineBarrier2(cmd,&di);
    });
}

void FrameRenderer::bootstrapSsgiTemporal() {
    oneShotSubmit(*m_device, m_pool, [&](VkCommandBuffer cmd) {
        VkImage imgs[2]={m_rt.ssgi.image(),m_rt.ssgiPrev.image()};
        for(auto img:imgs){
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask=VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT; b.srcAccessMask=0;
            b.dstStageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT; b.dstAccessMask=VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.image=img; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO}; di.imageMemoryBarrierCount=1; di.pImageMemoryBarriers=&b;
            vkCmdPipelineBarrier2(cmd,&di);
        }
    });
}

// ── 通用引导：将所有未初始化的纹理从 UNDEFINED 转到 SHADER_READ_ONLY ──
// 只做布局转换，不清除内容（避免要求 TRANSFER_DST usage）。
// 各 Pass 首次写入时会自行转换到正确的可写布局。
void FrameRenderer::bootstrapAllTargets() {
    oneShotSubmit(*m_device, m_pool, [&](VkCommandBuffer cmd) {
        // 辅助：UNDEFINED → SHADER_READ_ONLY 布局转换（支持多 mip）
        auto transitionFromUndefined = [&](VkImage img, VkImageAspectFlags aspect, uint32_t mips=1) {
            if (!img) return;
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b.srcAccessMask = 0;
            b.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            b.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.image         = img;
            b.subresourceRange = {aspect, 0, mips, 0, 1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &di);
        };
        auto& rt = m_rt;
        transitionFromUndefined(rt.ssr.image(),    VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(rt.rsmGI.image(),  VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(rt.restir.image(), VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(rt.rtGI.image(),   VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(rt.lumenGI.image(),VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(rt.ssao.image(),   VK_IMAGE_ASPECT_COLOR_BIT);
        // SDFGI
        transitionFromUndefined(m_sdfgi.seedA().image(), VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_sdfgi.seedB().image(), VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_sdfgi.udf().image(),   VK_IMAGE_ASPECT_COLOR_BIT);
        // LPV 体素网格（2 组 ping-pong × 3 通道 + gv）
        for (int g = 0; g < 2; ++g) {
            auto& gr = g ? m_lpv.next() : m_lpv.current();
            transitionFromUndefined(gr.lpvR.image(), VK_IMAGE_ASPECT_COLOR_BIT);
            transitionFromUndefined(gr.lpvG.image(), VK_IMAGE_ASPECT_COLOR_BIT);
            transitionFromUndefined(gr.lpvB.image(), VK_IMAGE_ASPECT_COLOR_BIT);
        }
        transitionFromUndefined(m_lpv.gv().image(), VK_IMAGE_ASPECT_COLOR_BIT);
        // VXGI 纹理（voxelGrid 有 mip 链，使用 image 的实际 mip 数）
        uint32_t vxMips = m_vxgi.image().image() ? m_vxgi.image().mipLevels() : 1u;
        transitionFromUndefined(m_vxgi.image().image(),  VK_IMAGE_ASPECT_COLOR_BIT, vxMips);
        transitionFromUndefined(m_vxgi.aniso().image(),  VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_vxgi.relightScratch().image(),  VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_vxgi.relightScratch2().image(), VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_vxgi.sixAxisX().image(), VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_vxgi.sixAxisY().image(), VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_vxgi.sixAxisZ().image(), VK_IMAGE_ASPECT_COLOR_BIT);
        // PRT（5 张 SH 纹理）+ DDGI（2 张 probe atlas）
        transitionFromUndefined(m_prt.image().image(),  VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_prt.imageB().image(), VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_prt.imageC().image(), VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_prt.imageD().image(), VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_prt.imageE().image(), VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_ddgi.irradiance().image(), VK_IMAGE_ASPECT_COLOR_BIT);
        transitionFromUndefined(m_ddgi.distance().image(),   VK_IMAGE_ASPECT_COLOR_BIT);
    });
}

void FrameRenderer::writeTimestamp(VkCommandBuffer cmd, uint32_t slot) {
    uint32_t base = m_currentFrameInFlight * kTimestampSlots;
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                         m_timestampPool, base + slot);
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
    if (m_device) {
        m_lighting.bindShadowMask(*m_device, m_shadow.shadowMask().view());
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
        .record = [this](VkCommandBuffer cmd) {
            // 确保 gDepth 在 SHADER_READ_ONLY_OPTIMAL layout（首帧从 UNDEFINED 过渡）
            transitionImage(cmd, rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            // 同理 gNormalRough（首帧 layout 不确定）
            transitionImage(cmd, rt().gNormalRough.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            shadow().record(cmd, rt(),
                gbuffer().frameUboHandle(),
                *m_boundScene.gpu, m_boundScene.indirectBufSun, m_boundScene.drawCount,
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
        .record = [this](VkCommandBuffer cmd) {
            // hdrColor → COLOR_ATTACHMENT, depth → DEPTH_ATTACHMENT
            transitionImage(cmd, rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            transitionImage(cmd, rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            forward().record(cmd, rt(), m_boundScene.indirectBuf, m_boundScene.drawCount, *m_boundScene.gpu);

            // hdrColor COLOR_ATTACHMENT → GENERAL（匹配延迟 Lighting 输出）
            transitionImage(cmd, rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            // depth DEPTH_ATTACHMENT → SR_O（匹配延迟 GBuffer 输出，供 Skybox 使用）
            transitionImage(cmd, rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

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
        .record = [this](VkCommandBuffer cmd) {
            // MSAA images → attachment layout
            auto toColorAttach = [&](VkImage img) {
                transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            };
            toColorAttach(rt().gAlbedoMetalMs.image());
            toColorAttach(rt().gNormalRoughMs.image());
            toColorAttach(rt().gEmissiveAOMs.image());
            transitionImage(cmd, rt().depthMs.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            // SS resolve targets → attachment layout
            toColorAttach(rt().gAlbedoMetal.image());
            toColorAttach(rt().gNormalRough.image());
            toColorAttach(rt().gEmissiveAO.image());
            transitionImage(cmd, rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            gbuffer().record(cmd, rt(), m_boundScene.indirectBuf, m_boundScene.drawCount, *m_boundScene.gpu);

            // Resolved GBuffer → SHADER_READ_ONLY for downstream compute
            auto toSampled = [&](VkImage img) {
                transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            };
            toSampled(rt().gAlbedoMetal.image());
            toSampled(rt().gNormalRough.image());
            toSampled(rt().gEmissiveAO.image());
            transitionImage(cmd, rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

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
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            ssao().record(cmd, rt(),
                m_frameState.proj, glm::inverse(m_frameState.proj), m_frameState.view);
            transitionImage(cmd, rt().ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    pipeline().addStep({
        .name = "AO-GTAO",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            gtao().record(cmd, rt(),
                m_frameState.proj, m_frameState.view);
            transitionImage(cmd, rt().ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    pipeline().addStep({
        .name = "AO-Clear",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue white{};
            white.float32[0] = 1.0f; white.float32[1] = 1.0f;
            white.float32[2] = 1.0f; white.float32[3] = 1.0f;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, rt().ssao.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &range);
            transitionImage(cmd, rt().ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    // ============================
    // Phase 1.6: SSR
    // ============================
    pipeline().addStep({
        .name = "SSR",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().ssr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            ssr().record(cmd, rt());
            transitionImage(cmd, rt().ssr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    pipeline().addStep({
        .name = "SSR-Clear",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().ssr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, rt().ssr.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, rt().ssr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    // ============================
    // Phase 1.7: ScreenGI (SSGI/GTGI, 互斥)
    // ============================
    pipeline().addStep({
        .name = "ScreenGI",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            // Copy ssgi → ssgiPrev for temporal history
            transitionImage(cmd, rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            transitionImage(cmd, rt().ssgiPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.extent = {rt().extent.width, rt().extent.height, 1};
            vkCmdCopyImage(cmd,
                rt().ssgi.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                rt().ssgiPrev.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &region);

            // ssgiPrev → SHADER_READ_ONLY for sampling
            transitionImage(cmd, rt().ssgiPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            // ssgi → GENERAL for writing new value
            transitionImage(cmd, rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            if (ssgi().enabled) ssgi().record(cmd, rt());
            else                gtgi().record(cmd, rt());

            transitionImage(cmd, rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    pipeline().addStep({
        .name = "ScreenGI-Clear",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, rt().ssgi.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    // === AO/SS 结束 timestamp ===
    pipeline().addStep({
        .name = "TS-AO",
        .phase = "AO",
        .record = [this](VkCommandBuffer cmd) {
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
        .record = [this](VkCommandBuffer cmd) {
            // 1. Clear entire mip chain to 0
            auto barrierAllMips = [&](VkImageLayout oldL, VkImageLayout newL,
                                       VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                       VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
                b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
                b.oldLayout = oldL; b.newLayout = newL;
                b.image = vxgi().image().image();
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                      0, vxgi().mipLevels(), 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };
            barrierAllMips(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT,
                                       0, vxgi().mipLevels(), 0, 1};
            vkCmdClearColorImage(cmd, vxgi().image().image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &rg);
            barrierAllMips(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            // 2. Voxelize: scatter all primitives to mip 0
            vxgiVoxelize().record(cmd, *m_boundScene.cpu, *m_boundScene.gpu,
                vxgiGridMin(), vxgiCellSize(), kVxgiResolution);

            // 3. Inject: RSM flux → voxel mip 0 RGB
            {
                VkImageMemoryBarrier2 vbar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                vbar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                vbar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                vbar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                vbar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                vbar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                vbar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                vbar.image = vxgi().image().image();
                vbar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo vdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                vdi.imageMemoryBarrierCount = 1; vdi.pImageMemoryBarriers = &vbar;
                vkCmdPipelineBarrier2(cmd, &vdi);
            }
            vxgiInject().record(cmd, kVxgiResolution, vxgiGridMin(), vxgiCellSize());

            // 4. Mipmap: iterate src mip i → dst mip i+1
            vxgiMipmap().record(cmd, vxgi());

            // 5. Final mip → SHADER_READ_ONLY
            {
                VkImageMemoryBarrier2 fb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                fb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                fb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                fb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                fb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                fb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                fb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                fb.image = vxgi().image().image();
                fb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                       vxgi().mipLevels() - 1, 1, 0, 1};
                VkDependencyInfo fdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                fdi.imageMemoryBarrierCount = 1; fdi.pImageMemoryBarriers = &fb;
                vkCmdPipelineBarrier2(cmd, &fdi);
            }

            // 6. Aniso alpha mipchain: UNDEFINED → SHADER_READ_ONLY
            {
                VkImageMemoryBarrier2 ab{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                ab.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                ab.srcAccessMask = 0;
                ab.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                ab.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                ab.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                ab.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                ab.image = vxgi().aniso().image();
                ab.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                       0, vxgi().mipLevels(), 0, 1};
                VkDependencyInfo adi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                adi.imageMemoryBarrierCount = 1; adi.pImageMemoryBarriers = &ab;
                vkCmdPipelineBarrier2(cmd, &adi);
            }
            vxgiAniso().record(cmd, vxgi());
        }
    });

    // VXGI Relight (multi-bounce, within VXGI chain)
    pipeline().addStep({
        .name = "VXGI-Relight",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            int bounces = lumenEnabled() ? 3 : 1;

            auto transImg = [&](VkImage img, VkImageLayout oldL, VkImageLayout newL,
                                VkPipelineStageFlags2 srcS, VkAccessFlags2 srcA,
                                VkPipelineStageFlags2 dstS, VkAccessFlags2 dstA) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcS; b.srcAccessMask = srcA;
                b.dstStageMask = dstS; b.dstAccessMask = dstA;
                b.oldLayout = oldL; b.newLayout = newL;
                b.image = img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };

            auto blitScratchToVoxel = [&](VkImage srcImg) {
                transImg(vxgi().image().image(),
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT);
                transImg(srcImg,
                    VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_READ_BIT);
                VkImageCopy region{};
                region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.extent = {kVxgiResolution, kVxgiResolution, kVxgiResolution};
                vkCmdCopyImage(cmd,
                    srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    vxgi().image().image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &region);
                transImg(vxgi().image().image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            };

            // Bounce 1: read voxelGrid → write scratch
            transImg(vxgi().relightScratch().image(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            {
                rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*rhiDevice()), cmd);
                vxgiRelight().record(rhiCmd, vxgiRelight().voxelSet(), kVxgiResolution,
                    vxgi().mipLevels(), vxgiCellSize(), vxgiGridMin(),
                    vxgiRelightStrength());
            }

            if (bounces >= 2) {
                transImg(vxgi().relightScratch().image(),
                    VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                transImg(vxgi().relightScratch2().image(),
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                // Bounce 2: read scratch → write scratch2
                {
                    rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*rhiDevice()), cmd);
                    vxgiRelight().record(rhiCmd, vxgiRelight().pingSet0(), kVxgiResolution,
                        vxgi().mipLevels(), vxgiCellSize(), vxgiGridMin(),
                        vxgiRelightStrength());
                }

                if (bounces >= 3) {
                    transImg(vxgi().relightScratch2().image(),
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                    transImg(vxgi().relightScratch().image(),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                    // Bounce 3: read scratch2 → write scratch
                    {
                        rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*rhiDevice()), cmd);
                        vxgiRelight().record(rhiCmd, vxgiRelight().pingSet1(), kVxgiResolution,
                            vxgi().mipLevels(), vxgiCellSize(), vxgiGridMin(),
                            vxgiRelightStrength());
                    }
                    blitScratchToVoxel(vxgi().relightScratch().image());
                } else {
                    blitScratchToVoxel(vxgi().relightScratch2().image());
                }
            } else {
                blitScratchToVoxel(vxgi().relightScratch().image());
            }
        }
    });

    // VXGI 6-axis resolve (Lumen mode only)
    pipeline().addStep({
        .name = "VXGI-6Axis",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            VkImageLayout axisOldL = lumenAtlasInited()
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            VkPipelineStageFlags2 axisSrcS = lumenAtlasInited()
                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            VkAccessFlags2 axisSrcA = lumenAtlasInited()
                ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0u;

            auto transAxisToGeneral = [&](VkImage img) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = axisSrcS; b.srcAccessMask = axisSrcA;
                b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                b.oldLayout = axisOldL;
                b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.image = img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };
            transAxisToGeneral(vxgi().sixAxisX().image());
            transAxisToGeneral(vxgi().sixAxisY().image());
            transAxisToGeneral(vxgi().sixAxisZ().image());

            vxgi6Axis().record(cmd, kVxgiResolution, vxgi().mipLevels(),
                vxgiCellSize(), vxgiGridMin(), vxgiRelightStrength());

            auto transAxisToSRO = [&](VkImage img) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b.image = img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };
            transAxisToSRO(vxgi().sixAxisX().image());
            transAxisToSRO(vxgi().sixAxisY().image());
            transAxisToSRO(vxgi().sixAxisZ().image());
        }
    });

    // VXGI bootstrap: when all consumers are off, transition voxel grid to SR_O
    pipeline().addStep({
        .name = "VXGI-Bootstrap",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b.srcAccessMask = 0;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.image = vxgi().image().image();
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                  0, vxgi().mipLevels(), 0, 1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &di);
            // aniso too
            b.image = vxgi().aniso().image();
            vkCmdPipelineBarrier2(cmd, &di);
        }
    });

    // ============================
    // Phase 1.835: SDFGI
    // ============================
    pipeline().addStep({
        .name = "SDFGI",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            if (!sdfgiBootstrapped()) {
                auto bootstrapToGeneral = [&](VkImage img) {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    b.srcAccessMask = 0;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.image = img;
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                };
                bootstrapToGeneral(sdfgi().seedA().image());
                bootstrapToGeneral(sdfgi().seedB().image());
                bootstrapToGeneral(sdfgi().udf().image());
                sdfgiBootstrapped() = true;
            }
            transitionImage(cmd, rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            sdfgiPass().record(cmd, sdfgi(), rt(), frameIndex(),
                sdfgiPass().seedThreshold, sdfgiPass().maxDistCells,
                (uint32_t)sdfgiPass().numRays,
                (uint32_t)sdfgiPass().maxSteps,
                sdfgiPass().rayMaxCells, sdfgiPass().hitEpsCells);

            transitionImage(cmd, rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    // ============================
    // Phase 1.836: RT GI
    // ============================
    pipeline().addStep({
        .name = "RTGI",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().rtGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            rtGi().record(cmd, rt());
            transitionImage(cmd, rt().rtGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    pipeline().addStep({
        .name = "RTGI-Clear",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().rtGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, rt().rtGI.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, rt().rtGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    // ============================
    // Phase 1.837: ReSTIR DI
    // ============================
    pipeline().addStep({
        .name = "ReSTIR",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            restir().updateLights(demoLights());
            if (!restirBootstrapped()) {
                auto bootstrapToGeneral = [&](VkImage img) {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    b.srcAccessMask = 0;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.image = img;
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                };
                bootstrapToGeneral(restir().reservoirA().image());
                bootstrapToGeneral(restir().reservoirB().image());
                restirBootstrapped() = true;
            }
            VkImageLayout restirOld = restirOutInited()
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            transitionImage(cmd, rt().restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                restirOld, VK_IMAGE_LAYOUT_GENERAL,
                restirOutInited() ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                   : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                restirOutInited() ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
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

            transitionImage(cmd, rt().restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    pipeline().addStep({
        .name = "ReSTIR-Clear",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            VkImageLayout restirOld = restirOutInited()
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            transitionImage(cmd, rt().restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                restirOld, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                restirOutInited() ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                   : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                restirOutInited() ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, rt().restir.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, rt().restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
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
        .record = [this](VkCommandBuffer cmd) {
            auto barrierAtlas = [&](VkImage img,
                                    VkImageLayout oldL, VkImageLayout newL,
                                    VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                    VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
                b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
                b.oldLayout = oldL; b.newLayout = newL;
                b.image = img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };

            VkImageLayout oldAtlasL = ddgiAtlasInited()
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            VkAccessFlags2 srcAcc = ddgiAtlasInited()
                ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;
            VkPipelineStageFlags2 srcStg = ddgiAtlasInited()
                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;

            barrierAtlas(ddgi().irradiance().image(),
                oldAtlasL, VK_IMAGE_LAYOUT_GENERAL,
                srcStg, srcAcc,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            barrierAtlas(ddgi().distance().image(),
                oldAtlasL, VK_IMAGE_LAYOUT_GENERAL,
                srcStg, srcAcc,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            float jitterRot = float((frameIndex() % 360) * 0.0174532925);
            ddgiPass().record(cmd, ddgi(), ddgiOrigin(), ddgiSpacing(),
                vxgiGridMin(), vxgiCellSize(), kVxgiResolution,
                jitterRot, frameIndex());

            barrierAtlas(ddgi().irradiance().image(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            barrierAtlas(ddgi().distance().image(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            ddgiAtlasInited() = true;
        }
    });

    pipeline().addStep({
        .name = "DDGI-Bootstrap",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            auto barrierAtlas = [&](VkImage img,
                                    VkImageLayout oldL, VkImageLayout newL,
                                    VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                    VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
                b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
                b.oldLayout = oldL; b.newLayout = newL;
                b.image = img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };
            barrierAtlas(ddgi().irradiance().image(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            barrierAtlas(ddgi().distance().image(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
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
        .record = [this](VkCommandBuffer cmd) {
            VkImageLayout oldL = lumenOutInited()
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            VkPipelineStageFlags2 srcS = lumenOutInited()
                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            VkAccessFlags2 srcA = lumenOutInited()
                ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0u;
            transitionImage(cmd, rt().lumenGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                oldL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                srcS, srcA,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue grey{};
            grey.float32[0] = 0.3f; grey.float32[1] = 0.3f;
            grey.float32[2] = 0.3f; grey.float32[3] = 1.0f;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, rt().lumenGI.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &grey, 1, &range);
            transitionImage(cmd, rt().lumenGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            lumenOutInited() = true;
        }
    });

    pipeline().addStep({
        .name = "Lumen-Probe",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            if (!lumenProbeInited()) {
                lumenProbe().init(*rhiDevice());
                lumenProbe().bindResources(lumen(), rtAS(), *m_boundScene.gpu,
                                                vxgi(), rt(), gbuffer().frameUboHandle(),
                                                vxgiSixAxisInited());
                lumenProbeInited() = true;
            }
            // Transition probe + filtered atlas to GENERAL
            {
                VkImageLayout oldL = lumenAtlasInited()
                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    : VK_IMAGE_LAYOUT_UNDEFINED;
                VkPipelineStageFlags2 srcS = lumenAtlasInited()
                    ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                    : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                VkAccessFlags2 srcA = lumenAtlasInited()
                    ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0u;
                auto transToGeneral = [&](VkImage img) {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = srcS; b.srcAccessMask = srcA;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.oldLayout = oldL;
                    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.image = img;
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                };
                transToGeneral(lumen().probeAtlas().image());
                transToGeneral(lumen().filteredAtlas().image());
                lumenAtlasInited() = true;
            }
            lumenProbe().record(cmd, lumen(), frameIndex(),
                                     lumenDebugMode() >= 3 ? (uint32_t)lumenDebugMode() - 1u
                                                           : (vxgiSixAxisInited() ? 1u : 0u));

            // ProbeAtlas GENERAL → SR_O for filter
            {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b.image = lumen().probeAtlas().image();
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            }
        }
    });

    pipeline().addStep({
        .name = "Lumen-Filter",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            if (!lumenFilterInited()) {
                VkImageMemoryBarrier2 pb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                pb.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                pb.srcAccessMask = 0;
                pb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                pb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                pb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                pb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                pb.image = lumen().prevAtlas().image();
                pb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo pdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                pdi.imageMemoryBarrierCount = 1; pdi.pImageMemoryBarriers = &pb;
                vkCmdPipelineBarrier2(cmd, &pdi);

                lumenFilter().init(*rhiDevice());
                lumenFilter().bindResources( lumen(), rt(),
                                                 gbuffer().frameUboHandle());
                lumenFilterInited() = true;
            }
            lumenFilter().record(cmd, lumen(), rt());

            // Copy filteredAtlas → prevAtlas for next frame
            auto imgBarrier = [&](VkImage img, VkImageLayout oldL, VkImageLayout newL,
                                  VkPipelineStageFlags2 srcS, VkAccessFlags2 srcA,
                                  VkPipelineStageFlags2 dstS, VkAccessFlags2 dstA) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcS; b.srcAccessMask = srcA;
                b.dstStageMask = dstS; b.dstAccessMask = dstA;
                b.oldLayout = oldL; b.newLayout = newL;
                b.image = img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };

            imgBarrier(lumen().filteredAtlas().image(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            imgBarrier(lumen().prevAtlas().image(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.extent = {lumen().atlasWidth(), lumen().atlasHeight(), 1};
            vkCmdCopyImage(cmd,
                lumen().filteredAtlas().image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                lumen().prevAtlas().image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &region);

            imgBarrier(lumen().filteredAtlas().image(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            imgBarrier(lumen().prevAtlas().image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    pipeline().addStep({
        .name = "Lumen-Gather",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            if (!lumenGatherInited()) {
                lumenGather().init(*rhiDevice());
                lumenGather().bindResources(lumen(), rt(),
                                                 gbuffer().frameUboHandle(), true);
                lumenGatherInited() = true;
            }
            {
                VkImageLayout oldL = lumenOutInited()
                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    : VK_IMAGE_LAYOUT_UNDEFINED;
                VkAccessFlags2 srcA = lumenOutInited()
                    ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;
                VkPipelineStageFlags2 srcS = lumenOutInited()
                    ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                    : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcS; b.srcAccessMask = srcA;
                b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                b.oldLayout = oldL;
                b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.image = rt().lumenGI.image();
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            }
            lumenGather().record(cmd, lumen(), rt(),
                                     (uint32_t)lumenDebugMode());

            {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b.image = rt().lumenGI.image();
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            }
            lumenOutInited() = true;
        }
    });

    pipeline().addStep({
        .name = "Lumen-Clear",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            VkImageLayout oldL = lumenOutInited()
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            VkPipelineStageFlags2 srcS = lumenOutInited()
                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            VkAccessFlags2 srcA = lumenOutInited()
                ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;
            transitionImage(cmd, rt().lumenGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                oldL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                srcS, srcA,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, rt().lumenGI.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, rt().lumenGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
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
        .record = [this](VkCommandBuffer cmd) {
            lpvInject().record(cmd, kLpvResolution, lpvGridMin(), lpvCellSize());

            transitionImage(cmd, lpv().gv().image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

            auto barrierLpv = [&](const LpvGrid& g,
                                  VkImageLayout oldL, VkImageLayout newL,
                                  VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                  VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                VkImage imgs[3] = {g.lpvR.image(), g.lpvG.image(), g.lpvB.image()};
                for (auto img : imgs) {
                    transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                        oldL, newL, srcStg, srcAcc, dstStg, dstAcc);
                }
            };

            int propIter = lpvProp().iterations & ~1;
            for (int it = 0; it < propIter; ++it) {
                LpvGrid& src = lpv().current();
                LpvGrid& dst = lpv().next();

                barrierLpv(src,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                barrierLpv(dst,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                lpvProp().record(cmd, lpv().curIdx(),
                                 kLpvResolution, lpvProp().occlusionAmplifier,
                                 lpvProp().gvOcclusionStrength);
                lpv().swap();
            }

            barrierLpv(lpv().current(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    pipeline().addStep({
        .name = "LPV-Bootstrap",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            VkImage imgs[3] = {lpv().current().lpvR.image(),
                               lpv().current().lpvG.image(),
                               lpv().current().lpvB.image()};
            for (auto img : imgs) {
                transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
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
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().rsmGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            rsmSample().record(cmd, rt());
            transitionImage(cmd, rt().rsmGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    pipeline().addStep({
        .name = "RSM-Clear",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().rsmGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, rt().rsmGI.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, rt().rsmGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
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
        .record = [this](VkCommandBuffer cmd) {
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
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
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
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                    VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
            transitionImage(cmd, rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
            skybox().record(cmd, rt());
        }
    });

    // ============================
    // Phase 3.5: Copy hdrColor → hdrPrev
    // ============================
    pipeline().addStep({
        .name = "Copy-hdrPrev",
        .phase = "Shading",
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            transitionImage(cmd, rt().hdrPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkImageCopy hdrCopy{};
            hdrCopy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            hdrCopy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            hdrCopy.extent = {rt().extent.width, rt().extent.height, 1};
            vkCmdCopyImage(cmd,
                rt().hdrColor.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                rt().hdrPrev.image(),  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &hdrCopy);
            transitionImage(cmd, rt().hdrPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

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
        oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
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
