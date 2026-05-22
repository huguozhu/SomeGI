#include "render_targets.h"
#include "core/device.h"

namespace somegi {

void RenderTargets::create(Device& d, VkExtent2D ext, VkSampleCountFlagBits msaaSamples) {
    extent = ext;

    ImageDesc hdr{};
    hdr.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    hdr.extent = {ext.width, ext.height, 1};
    hdr.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
              | VK_IMAGE_USAGE_SAMPLED_BIT
              | VK_IMAGE_USAGE_STORAGE_BIT
              | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    hdrColor = Image(d, hdr);

    ImageDesc dep{};
    dep.format = VK_FORMAT_D32_SFLOAT;
    dep.extent = {ext.width, ext.height, 1};
    dep.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    dep.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
              | VK_IMAGE_USAGE_SAMPLED_BIT;
    depth = Image(d, dep);

    ImageDesc ldr{};
    ldr.format = VK_FORMAT_B8G8R8A8_UNORM;
    ldr.extent = {ext.width, ext.height, 1};
    ldr.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ldrTonemap = Image(d, ldr);

    // M4: GBuffer color attachments. All sampled by LightingPass; SS* passes
    // sample a subset.
    auto mkGBufferRT = [&](VkFormat fmt, Image& img) {
        ImageDesc id{};
        id.format = fmt;
        id.extent = {ext.width, ext.height, 1};
        id.usage  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                  | VK_IMAGE_USAGE_STORAGE_BIT;   // needed by vis-buffer resolve pass
        img = Image(d, id);
    };
    mkGBufferRT(VK_FORMAT_R8G8B8A8_UNORM,      gAlbedoMetal);
    mkGBufferRT(VK_FORMAT_R16G16B16A16_SFLOAT, gNormalRough);
    mkGBufferRT(VK_FORMAT_R8G8B8A8_UNORM,      gEmissiveAO);

    // Nanite Phase 1: Visibility buffer (R32G32_UINT)
    {
        ImageDesc vb{};
        vb.format = VK_FORMAT_R32G32_UINT;
        vb.extent = {ext.width, ext.height, 1};
        vb.usage  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        visBuffer = Image(d, vb);
    }

    auto mkMSAA = [&](VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect, Image& img) {
        ImageDesc id{};
        id.format = fmt;
        id.extent = {ext.width, ext.height, 1};
        id.samples = msaaSamples;
        id.usage = usage;
        id.aspect = aspect;
        img = Image(d, id);
    };
    mkMSAA(VK_FORMAT_R8G8B8A8_UNORM,
           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
           VK_IMAGE_ASPECT_COLOR_BIT, gAlbedoMetalMs);
    mkMSAA(VK_FORMAT_R16G16B16A16_SFLOAT,
           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
           VK_IMAGE_ASPECT_COLOR_BIT, gNormalRoughMs);
    mkMSAA(VK_FORMAT_R8G8B8A8_UNORM,
           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
           VK_IMAGE_ASPECT_COLOR_BIT, gEmissiveAOMs);
    mkMSAA(VK_FORMAT_D32_SFLOAT,
           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
           VK_IMAGE_ASPECT_DEPTH_BIT, depthMs);

    // SSAO: STORAGE (compute write) + SAMPLED (lighting reads).
    ImageDesc s{};
    s.format = VK_FORMAT_R8_UNORM;
    s.extent = {ext.width, ext.height, 1};
    s.usage  = VK_IMAGE_USAGE_STORAGE_BIT
             | VK_IMAGE_USAGE_SAMPLED_BIT
             | VK_IMAGE_USAGE_TRANSFER_DST_BIT;   // for vkCmdClearColorImage when SSAO disabled
    ssao = Image(d, s);

    // SSR output (storage write by SsrPass + sampled by Lighting +
    // transfer_dst for clear-when-disabled).
    ImageDesc rs{};
    rs.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    rs.extent = {ext.width, ext.height, 1};
    rs.usage  = VK_IMAGE_USAGE_STORAGE_BIT
              | VK_IMAGE_USAGE_SAMPLED_BIT
              | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ssr = Image(d, rs);

    // Previous-frame HDR copy: TRANSFER_DST (we copy hdrColor into it) +
    // SAMPLED (SsrPass / SsgiPass read it).
    ImageDesc hp{};
    hp.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    hp.extent = {ext.width, ext.height, 1};
    hp.usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT
              | VK_IMAGE_USAGE_SAMPLED_BIT;
    hdrPrev = Image(d, hp);

    // SSGI output: storage write + sampled by Lighting + clear-when-disabled.
    // 加 TRANSFER_SRC：B.4 时序累积每帧 copy ssgi → ssgiPrev。
    ImageDesc gi{};
    gi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    gi.extent = {ext.width, ext.height, 1};
    gi.usage  = VK_IMAGE_USAGE_STORAGE_BIT
              | VK_IMAGE_USAGE_SAMPLED_BIT
              | VK_IMAGE_USAGE_TRANSFER_DST_BIT
              | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ssgi = Image(d, gi);

    // SSGI 上一帧副本：TRANSFER_DST（每帧 copy 进来）+ SAMPLED（SSGI shader
    // reproject 后 sample）。layout 链：每帧前 copy 时 TRANSFER_DST，给
    // SSGI shader 用时 SHADER_READ_ONLY。
    ImageDesc gp{};
    gp.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    gp.extent = {ext.width, ext.height, 1};
    gp.usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT
              | VK_IMAGE_USAGE_SAMPLED_BIT;
    ssgiPrev = Image(d, gp);

    // M5 RSM gather 输出 —— 与 SSGI 同形 (RGBA16F, swapchain 大小, storage
    // 写 + sampled 读 + 关闭时 vkCmdClearColorImage 清成 0)。
    ImageDesc rg{};
    rg.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    rg.extent = {ext.width, ext.height, 1};
    rg.usage  = VK_IMAGE_USAGE_STORAGE_BIT
              | VK_IMAGE_USAGE_SAMPLED_BIT
              | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    rsmGI = Image(d, rg);

    // C.4 ReSTIR DI：与 ssgi 同形（RGBA16F screen-res，storage 写 + sampled
    // 读 + 关闭时 clear-to-0）。
    ImageDesc rs2{};
    rs2.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    rs2.extent = {ext.width, ext.height, 1};
    rs2.usage  = VK_IMAGE_USAGE_STORAGE_BIT
               | VK_IMAGE_USAGE_SAMPLED_BIT
               | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    restir = Image(d, rs2);

    // M9 RT GI：STORAGE（RT 写）+ TRANSFER_DST（关闭时 clear）+ SAMPLED
    // （lighting 读）。
    ImageDesc rt{};
    rt.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    rt.extent = {ext.width, ext.height, 1};
    rt.usage  = VK_IMAGE_USAGE_STORAGE_BIT
              | VK_IMAGE_USAGE_SAMPLED_BIT
              | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    rtGI = Image(d, rt);

    // L.5 Lumen-lite gather 输出：STORAGE + SAMPLED + TRANSFER_DST。
    ImageDesc lg{};
    lg.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    lg.extent = {ext.width, ext.height, 1};
    lg.usage  = VK_IMAGE_USAGE_STORAGE_BIT
              | VK_IMAGE_USAGE_SAMPLED_BIT
              | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    lumenGI = Image(d, lg);
}

void RenderTargets::recreateMsaa(Device& d, VkSampleCountFlagBits samples) {
    gAlbedoMetalMs.reset();
    gNormalRoughMs.reset();
    gEmissiveAOMs.reset();
    depthMs.reset();

    auto mkMSAA = [&](VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect, Image& img) {
        ImageDesc id{};
        id.format = fmt;
        id.extent = {extent.width, extent.height, 1};
        id.samples = samples;
        id.usage = usage;
        id.aspect = aspect;
        img = Image(d, id);
    };
    mkMSAA(VK_FORMAT_R8G8B8A8_UNORM,
           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
           VK_IMAGE_ASPECT_COLOR_BIT, gAlbedoMetalMs);
    mkMSAA(VK_FORMAT_R16G16B16A16_SFLOAT,
           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
           VK_IMAGE_ASPECT_COLOR_BIT, gNormalRoughMs);
    mkMSAA(VK_FORMAT_R8G8B8A8_UNORM,
           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
           VK_IMAGE_ASPECT_COLOR_BIT, gEmissiveAOMs);
    mkMSAA(VK_FORMAT_D32_SFLOAT,
           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
           VK_IMAGE_ASPECT_DEPTH_BIT, depthMs);
}

void RenderTargets::ensureAaResources(Device& d) {
    if (aaHdr.image() != VK_NULL_HANDLE) return;
    ImageDesc hdr{};
    hdr.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    hdr.extent = {extent.width, extent.height, 1};
    hdr.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    aaHdr = Image(d, hdr);

    ImageDesc hist{};
    hist.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    hist.extent = {extent.width, extent.height, 1};
    hist.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    aaHistory = Image(d, hist);
}

void RenderTargets::destroyAaResources() {
    aaHdr.reset();
    aaHistory.reset();
}

void RenderTargets::destroy() {
    hdrColor.reset();
    depth.reset();
    ldrTonemap.reset();
    gAlbedoMetal.reset();
    gNormalRough.reset();
    gEmissiveAO.reset();
    visBuffer.reset();
    gAlbedoMetalMs.reset();
    gNormalRoughMs.reset();
    gEmissiveAOMs.reset();
    depthMs.reset();
    ssao.reset();
    ssr.reset();
    hdrPrev.reset();
    ssgi.reset();
    ssgiPrev.reset();
    rsmGI.reset();
    restir.reset();
    rtGI.reset();
    lumenGI.reset();
    aaHdr.reset();
    aaHistory.reset();
}

}
