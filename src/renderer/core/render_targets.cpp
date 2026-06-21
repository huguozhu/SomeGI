#include "renderer/core/render_targets.h"
#include "core/device.h"
#include "rhi/base/device.h"
#include "rhi/base/texture.h"

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
        id.usage  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        img = Image(d, id);
    };
    mkGBufferRT(VK_FORMAT_R8G8B8A8_UNORM,      gAlbedoMetal);
    mkGBufferRT(VK_FORMAT_R16G16B16A16_SFLOAT, gNormalRough);
    mkGBufferRT(VK_FORMAT_R8G8B8A8_UNORM,      gEmissiveAO);

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

void RenderTargets::createRHI(rhi::RHIDevice& rhiDev, VkExtent2D ext, VkSampleCountFlagBits msaaSamples) {
    extent = ext;
    auto makeTex = [&](rhi::Format fmt, VkImageUsageFlags usage,
                       std::unique_ptr<rhi::RHITexture>& tex,
                       std::unique_ptr<rhi::RHITextureView>& view) {
        rhi::TextureDesc td;
        td.format = fmt;
        td.width = ext.width; td.height = ext.height;
        uint32_t u = (uint32_t)rhi::TextureUsage::Sampled | (uint32_t)rhi::TextureUsage::Storage;
        if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
            u |= (uint32_t)rhi::TextureUsage::ColorAttachment;
        if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
            u = (uint32_t)rhi::TextureUsage::DepthStencil;
        if (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
            u |= (uint32_t)rhi::TextureUsage::TransferSrc;
        if (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            u |= (uint32_t)rhi::TextureUsage::TransferDst;
        td.usage = (rhi::TextureUsage)u;
        tex.reset(static_cast<rhi::RHITexture*>(
            rhiDev.createTexture(td).release()));
        view = tex->createView({});
    };

    makeTex(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        rhi.hdrColor, rhi.hdrColorView);

    makeTex(rhi::Format::D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        rhi.depth, rhi.depthView);

    makeTex(rhi::Format::B8G8R8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        rhi.ldrTonemap, rhi.ldrTonemapView);

    makeTex(rhi::Format::R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        rhi.gAlbedoMetal, rhi.gAlbedoMetalView);

    makeTex(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        rhi.gNormalRough, rhi.gNormalRoughView);

    makeTex(rhi::Format::R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        rhi.gEmissiveAO, rhi.gEmissiveAOView);

    // MSAA targets (skip if samples == 1)
    auto makeMsaa = [&](rhi::Format fmt, VkImageUsageFlags usage,
                        std::unique_ptr<rhi::RHITexture>& tex) {
        if (msaaSamples == VK_SAMPLE_COUNT_1_BIT) return;
        rhi::TextureDesc td;
        td.format = fmt; td.width = ext.width; td.height = ext.height;
        td.samples = (uint32_t)msaaSamples;
        uint32_t u = 0;
        if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) u |= (uint32_t)rhi::TextureUsage::ColorAttachment;
        if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) u = (uint32_t)rhi::TextureUsage::DepthStencil;
        td.usage = (rhi::TextureUsage)u;
        tex.reset(static_cast<rhi::RHITexture*>(rhiDev.createTexture(td).release()));
    };
    makeMsaa(rhi::Format::R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, rhi.gAlbedoMetalMs);
    makeMsaa(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, rhi.gNormalRoughMs);
    makeMsaa(rhi::Format::R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, rhi.gEmissiveAOMs);
    makeMsaa(rhi::Format::D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, rhi.depthMs);

    // AO + screen-space
    auto makeAux = [&](rhi::Format fmt, VkImageUsageFlags usage,
                       std::unique_ptr<rhi::RHITexture>& tex,
                       std::unique_ptr<rhi::RHITextureView>& view) {
        rhi::TextureDesc td;
        td.format = fmt; td.width = ext.width; td.height = ext.height;
        uint32_t u = (uint32_t)rhi::TextureUsage::Sampled | (uint32_t)rhi::TextureUsage::Storage;
        if (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) u |= (uint32_t)rhi::TextureUsage::TransferSrc;
        if (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) u |= (uint32_t)rhi::TextureUsage::TransferDst;
        td.usage = (rhi::TextureUsage)u;
        tex.reset(static_cast<rhi::RHITexture*>(rhiDev.createTexture(td).release()));
        view = tex->createView({});
    };
    makeAux(rhi::Format::R8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, rhi.ssao, rhi.ssaoView);
    makeAux(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, rhi.ssr, rhi.ssrView);
    makeAux(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        rhi.hdrPrev, rhi.hdrPrevView);
    makeAux(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, rhi.ssgi, rhi.ssgiView);
    makeAux(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        rhi.ssgiPrev, rhi.ssgiPrevView);

    // GI outputs
    makeAux(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, rhi.rsmGI, rhi.rsmGIView);
    makeAux(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, rhi.restir, rhi.restirView);
    makeAux(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        rhi.rtGI, rhi.rtGIView);
    makeAux(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        rhi.lumenGI, rhi.lumenGIView);

    // AA textures
    makeAux(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        rhi.aaHdr, rhi.aaHdrView);
    makeAux(rhi::Format::R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        rhi.aaHistory, rhi.aaHistoryView);
}

} // namespace somegi
