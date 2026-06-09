#include "renderer/core/frame_renderer.h"
#include "core/device.h"
#include "scene/upload.h"
#include <cstdio>

namespace somegi {

void FrameRenderer::init(Device& d, VkCommandPool pool, VkExtent2D extent,
                          VkSampleCountFlagBits msaaSamples, bool rtSupported,
                          VkFormat swapchainFmt, GLFWwindow* window) {
    m_device = &d;
    m_pool   = pool;
    m_rtSupported = rtSupported;

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

    // Register pipeline images with BarrierManager
    m_barriers.registerImage(m_rt.gAlbedoMetal.image(), VK_IMAGE_ASPECT_COLOR_BIT);
    m_barriers.registerImage(m_rt.gNormalRough.image(),  VK_IMAGE_ASPECT_COLOR_BIT);
    m_barriers.registerImage(m_rt.gEmissiveAO.image(),   VK_IMAGE_ASPECT_COLOR_BIT);
    m_barriers.registerImage(m_rt.depth.image(),          VK_IMAGE_ASPECT_DEPTH_BIT);
    m_barriers.registerImage(m_rt.hdrColor.image(),       VK_IMAGE_ASPECT_COLOR_BIT);
    m_barriers.registerImage(m_rt.ssao.image(),           VK_IMAGE_ASPECT_COLOR_BIT);
    m_barriers.registerImage(m_rt.ssr.image(),            VK_IMAGE_ASPECT_COLOR_BIT);
    m_barriers.registerImage(m_rt.ssgi.image(),           VK_IMAGE_ASPECT_COLOR_BIT);
    m_barriers.registerImage(m_rt.ldrTonemap.image(),     VK_IMAGE_ASPECT_COLOR_BIT);

    // Core passes
    std::printf("[init] gbuffer pass...\n");
    m_gbuffer.init(d, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT,
                   VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT, 128, msaaSamples);
    std::printf("[init] forward pass...\n");
    m_forward.init(d, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_D32_SFLOAT, 128);
    std::printf("[init] rsm geometry pass...\n");
    m_rsmGeom.init(d, 128);
    std::printf("[init] rsm sample pass...\n");
    m_rsmSample.init(d);

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
    m_ddgiPass.init(d);
    m_ddgiPass.bindResources(d, m_ddgi, m_vxgi);

    std::printf("[init] ndgi resources + pass...\n");
    m_ndgi.create(d);
    m_ndgiPass.init(d, rtSupported);
    m_ndgiInited = false;

    std::printf("[init] lighting pass...\n");
    m_lighting.init(d);
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
    m_ssao.init(d);
    m_ssao.bindFrame(d, m_rt);
    std::printf("[init] gtao pass...\n");
    m_gtao.init(d);
    m_gtao.bindFrame(d, m_rt, m_gbuffer.frameUboHandle());
    std::printf("[init] ssr pass...\n");
    m_ssr.init(d);
    m_ssr.bindFrame(d, m_rt, m_gbuffer.frameUboHandle());
    std::printf("[init] ssgi pass...\n");
    m_ssgi.init(d);
    m_ssgi.bindFrame(d, m_rt, m_gbuffer.frameUboHandle());
    std::printf("[init] gtgi pass...\n");
    m_gtgi.init(d);
    m_gtgi.bindFrame(d, m_rt, m_gbuffer.frameUboHandle());

    std::printf("[init] sdfgi resources/pass...\n");
    m_sdfgi.create(d, kSdfgiResolution);
    m_sdfgiPass.init(d);
    m_sdfgiPass.bindResources(d, m_sdfgi, m_vxgi, m_rt, m_gbuffer.frameUboHandle());

    std::printf("[init] restir resources/pass...\n");
    m_restir.create(d, extent, kRestirMaxLights);
    m_restirPass.init(d, rtSupported);

    if (rtSupported) {
        std::printf("[init] rt gi pass...\n");
        m_rtGiPass.init(d);
        m_rtGiInited = true;
    }
    if (rtSupported) {
        std::printf("[init] lumen resources...\n");
        m_lumen.create(d, extent);
    }
    m_restirPass.bindResources(d, m_restir, m_vxgi, m_rt, m_gbuffer.frameUboHandle());

    m_rsmSample.bindFrame(d, m_rt, m_gbuffer.frameUboHandle(),
        m_rsmGeom.frameUboHandle(),
        m_rsmGeom.position(), m_rsmGeom.normal(), m_rsmGeom.flux());

    std::printf("[init] lpv inject pass...\n");
    m_lpvInject.init(d, RsmGeometryPass::kRsmSize);

    std::printf("[init] vxgi voxelize/inject/mipmap pass...\n");
    m_vxgiVoxelize.init(d, 128);
    m_vxgiInject.init(d, RsmGeometryPass::kRsmSize);
    m_vxgiInject.bindResources(d, m_rsmGeom.position(), m_rsmGeom.flux(), m_vxgi);
    m_vxgiMipmap.init(d, m_vxgi.mipLevels());
    m_vxgiMipmap.bindResources(d, m_vxgi);
    m_vxgiAniso.init(d, m_vxgi.mipLevels());
    m_vxgiAniso.bindResources(d, m_vxgi);
    m_vxgiRelight.init(d);
    m_vxgiRelight.bindResources(d, m_vxgi, m_vxgi.relightScratch().view());
    m_vxgiRelight.bindResourcesPingPong(d, m_vxgi, false);
    m_vxgiRelight.bindResourcesPingPong(d, m_vxgi, true);
    if (m_vxgiSixAxisInited) {
        m_vxgiResolve6Axis.init(d);
        m_vxgiResolve6Axis.bindResources(d, m_vxgi);
    }

    std::printf("[init] prt bake pass...\n");
    m_prtBake.init(d);
    m_prtBake.bindResources(d, m_vxgi, m_prt);
    m_lpvInject.bindResources(d, m_rsmGeom.position(), m_rsmGeom.normal(),
                              m_rsmGeom.flux(), m_lpv.current(), m_lpv.gv());
    std::printf("[init] lpv propagate pass...\n");
    m_lpvProp.init(d);
    m_lpvProp.bindResources(d, m_lpv.current(), m_lpv.next(), m_lpv.gv());

    bootstrapHdrPrev();
    bootstrapSsgiTemporal();

    std::printf("[init] skybox pass...\n");
    m_skybox.init(d, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_D32_SFLOAT);

    std::printf("[init] tonemap pass...\n");
    // linearSampler comes from SceneGpu — caller sets after scene load
    std::printf("[init] aa passes...\n");
    m_taa.init(d);
    m_smaa.init(d, extent);
    std::printf("[init] imgui pass...\n");

    registerPipelineSteps();
    std::printf("[init] all renderer passes set up.\n");
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
    m_giTech.reset();
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
    m_ssao.bindFrame(d, m_rt);
    m_gtao.bindFrame(d, m_rt, m_gbuffer.frameUboHandle());
    m_ssr.bindFrame(d, m_rt, m_gbuffer.frameUboHandle());
    m_ssgi.bindFrame(d, m_rt, m_gbuffer.frameUboHandle());
    m_gtgi.bindFrame(d, m_rt, m_gbuffer.frameUboHandle());
    m_sdfgiPass.bindResources(d, m_sdfgi, m_vxgi, m_rt, m_gbuffer.frameUboHandle());
    m_restir.resize(d, newExtent);
    m_restirPass.bindResources(d, m_restir, m_vxgi, m_rt, m_gbuffer.frameUboHandle());
    m_restirOutInited = false;
    m_restirBootstrapped = false;
    m_rsmSample.bindFrame(d, m_rt, m_gbuffer.frameUboHandle(),
        m_rsmGeom.frameUboHandle(),
        m_rsmGeom.position(), m_rsmGeom.normal(), m_rsmGeom.flux());
    if (m_rtSupported) {
        m_lumen.destroy();
        m_lumen.create(d, newExtent);
        m_lumenAtlasInited = false;
        m_lumenOutInited = false;
    }
    m_tonemap.bindTargets(d, m_rt);
    bootstrapHdrPrev();
    bootstrapSsgiTemporal();
}

void FrameRenderer::bindScenePasses(Device& d, const SceneGpu& gpu, uint32_t textureCount) {
    m_gbuffer.bindScene(d, gpu, textureCount);
    m_forward.bindScene(d, gpu, textureCount);
    if (m_giTech) m_forward.setTechnique(m_giTech.get());
    m_rsmGeom.bindScene(d, gpu, textureCount);
    m_vxgiVoxelize.bindScene(d, gpu, textureCount, m_vxgi);
}

void FrameRenderer::setGiTechnique(IGITechnique* tech) {
    m_lighting.setTechnique(tech);
    if (m_giTech) m_forward.setTechnique(m_giTech.get());
}

void FrameRenderer::applyGiFlags(int effectiveGiIndex) {
    m_lpvEnabled         = (effectiveGiIndex == 4);
    m_vxgiEnabled        = (effectiveGiIndex == 5);
    m_prtEnabled         = (effectiveGiIndex == 6);
    m_ddgiEnabled        = (effectiveGiIndex == 7);
    m_lumenEnabled       = (effectiveGiIndex == 12);
    m_vxgiRelightEnabled = m_lumenEnabled;
    m_ssgi.enabled       = (effectiveGiIndex == 2);
    m_rsmSample.enabled  = (effectiveGiIndex == 3);
    m_gtgi.enabled       = (effectiveGiIndex == 8);
    m_sdfgiPass.enabled  = (effectiveGiIndex == 9);
    m_restirPass.enabled = (effectiveGiIndex == 11);
}

void FrameRenderer::bootstrapHdrPrev() {
    oneShotSubmit(*m_device, m_pool, [&](VkCommandBuffer cmd) {
        auto ti = [](VkCommandBuffer c, VkImage img, VkImageAspectFlags a,
                     VkImageLayout oldL, VkImageLayout newL,
                     VkPipelineStageFlags2 srcS, VkAccessFlags2 srcA,
                     VkPipelineStageFlags2 dstS, VkAccessFlags2 dstA) {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask=srcS;b.srcAccessMask=srcA;b.dstStageMask=dstS;b.dstAccessMask=dstA;
            b.oldLayout=oldL;b.newLayout=newL;b.image=img;
            b.subresourceRange={a,0,1,0,1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount=1;di.pImageMemoryBarriers=&b;
            vkCmdPipelineBarrier2(c,&di);
        };
        VkImage img=m_rt.hdrPrev.image();
        ti(cmd,img,VK_IMAGE_ASPECT_COLOR_BIT,
           VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,0,
           VK_PIPELINE_STAGE_2_CLEAR_BIT,VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkClearColorValue z{};
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        vkCmdClearColorImage(cmd,img,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&z,1,&r);
        ti(cmd,img,VK_IMAGE_ASPECT_COLOR_BIT,
           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
           VK_PIPELINE_STAGE_2_CLEAR_BIT,VK_ACCESS_2_TRANSFER_WRITE_BIT,
           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    });
}

void FrameRenderer::bootstrapSsgiTemporal() {
    oneShotSubmit(*m_device, m_pool, [&](VkCommandBuffer cmd) {
        auto ti = [&](VkImage img,VkImageLayout oldL,VkImageLayout newL,
                      VkPipelineStageFlags2 srcS,VkAccessFlags2 srcA,
                      VkPipelineStageFlags2 dstS,VkAccessFlags2 dstA) {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask=srcS;b.srcAccessMask=srcA;b.dstStageMask=dstS;b.dstAccessMask=dstA;
            b.oldLayout=oldL;b.newLayout=newL;b.image=img;
            b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount=1;di.pImageMemoryBarriers=&b;
            vkCmdPipelineBarrier2(cmd,&di);
        };
        VkImage imgs[2]={m_rt.ssgi.image(),m_rt.ssgiPrev.image()};
        for(auto img:imgs){
            ti(img,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,0,
               VK_PIPELINE_STAGE_2_CLEAR_BIT,VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue z{};
            VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            vkCmdClearColorImage(cmd,img,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&z,1,&r);
            ti(img,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_2_CLEAR_BIT,VK_ACCESS_2_TRANSFER_WRITE_BIT,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });
}

void FrameRenderer::writeTimestamp(VkCommandBuffer cmd, uint32_t slot) {
    uint32_t base = (m_frameIndex % kFramesInFlight) * kTimestampSlots;
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

// registerPipelineSteps and buildPipelineTable are stubs.
// Their implementation mirrors the inline logic currently in App::run().
// Migration: move the body of App::run()'s pipeline step registration here.
void FrameRenderer::registerPipelineSteps() { /* TODO: migrate from App */ }
void FrameRenderer::buildPipelineTable()    { /* TODO: migrate from App */ }

} // namespace somegi
