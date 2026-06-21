// app_fg.cpp — FrameGraph 管线编译
#include "app.h"
#include "core/swapchain.h"
#include "renderer/fg/fg_graph.h"
#include "renderer/fg/fg_executor.h"
#include "renderer/fg/fg_pass_node.h"
#include "renderer/fg/fg_builder.h"
#include "renderer/fg/fg_resources.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_texture.h"

namespace somegi {

void App::setupFrameGraph() {
    using namespace somegi::fg;
    auto ext2d = m_swap->extent();
    VkExtent3D ext{ext2d.width, ext2d.height, 1};

    bool aaEnabled = (m_aaMethod == AAMethod::TAA || m_aaMethod == AAMethod::SMAA);
    setupFgImports(ext, aaEnabled);

    // ================================================================
    // 注册所有 Pass（按执行顺序）
    // 注意：FrameGraph 会自动插入 barrier，不需要手动 transitionImage
    // ================================================================

    bool fwd = (m_renderingMode == RenderingMode::Forward);
    bool aoEnabled = (!fwd && m_aoMethod != AOMethod::None);
    bool needVoxelGrid = !fwd && (m_renderer.vxgiEnabled() || m_renderer.ddgiEnabled()
        || m_renderer.sdfgiPass().enabled || m_renderer.lumenEnabled()
        || m_renderer.restirPass().enabled);

    // ════════════════════════════════════════════════════════════════
    // 注意：必须先声明 GBuffer/Forward（写 depth），再声明 Shadow（读 depth），
    // 否则 FrameGraph 的 buildEdges 无法建立 RAW 依赖边。
    // ════════════════════════════════════════════════════════════════

    // --- Forward (替代 GBuffer，前向渲染模式) ---
    if (fwd) {
        // FrameGraph auto-barrier: hdrColor → COLOR_ATTACHMENT（entry），depth → DEPTH_ATTACHMENT（entry）
        //                           hdrColor → COPY_SRC（exit，由 Copy-hdrPrev 触发），depth → SR_O（exit，由 Shadow 触发）
        m_fg.addPass("Forward", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Graphics);
            b.write(m_fgh.hdrColor);
            b.write(m_fgh.depth);
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice());
                rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
                auto rhiIb = rhi::VkRHIBuffer::createNonOwning(vkDev, m_indirectBuf.handle(), VK_WHOLE_SIZE);
                m_renderer.forward().record(rhiCmd, m_renderer.rt(), *rhiIb, m_drawCount, m_sceneGpu);
            });
        });
    }

    // --- GBuffer ---
    // FrameGraph auto-barrier:
    //   MSAA targets → COLOR_ATTACHMENT / DEPTH_ATTACHMENT（entry）
    //   Resolve targets → COLOR_ATTACHMENT / DEPTH_ATTACHMENT（entry）
    //   Resolved GBuffer → SR_O（exit，由 SSAO/SSR/Lighting 的 read 触发）
    if (!fwd) {
        m_fg.addPass("GBuffer", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Graphics);
            b.write(m_fgh.gAlbedoMetalMs);
            b.write(m_fgh.gNormalRoughMs);
            b.write(m_fgh.gEmissiveAOMs);
            b.write(m_fgh.depthMs);
            b.write(m_fgh.gAlbedoMetal);   // resolve targets
            b.write(m_fgh.gNormalRough);
            b.write(m_fgh.gEmissiveAO);
            b.write(m_fgh.depth);
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice());
                rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
                auto rhiIb = rhi::VkRHIBuffer::createNonOwning(vkDev, m_indirectBuf.handle(), VK_WHOLE_SIZE);
                m_renderer.gbuffer().record(rhiCmd, m_renderer.rt(), *rhiIb, m_drawCount, m_sceneGpu);
            });
        });
    }

    // --- RSM Geometry ---
    m_fg.addPass("RSM-Geometry", [&](FGBuilder& b) {
        b.setPassType(FGPassType::Graphics);
        b.setManualBarriers();  // 复杂 Pass：内部自行管理 layout
        b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
            auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice());
            rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
            auto rhiIb = rhi::VkRHIBuffer::createNonOwning(vkDev, m_indirectBufSun.handle(), VK_WHOLE_SIZE);
            m_renderer.rsmGeom().record(rhiCmd, *rhiIb, m_drawCount, m_sceneGpu);
        });
    });

    // --- Shadow Pass ---
    // Auto-barrier：depth read → SHADER_READ_ONLY（compute shader 采样）
    //               shadowMask write → GENERAL（storage image write）
    // ShadowPass 内部跳过 UNDEFINED→GENERAL 和 GENERAL→SR_O 过渡
    m_renderer.shadow().setFgAutoBarrier(true);
    m_fg.addPass("Shadow", [&](FGBuilder& b) {
        b.setPassType(FGPassType::Compute);
        b.read(m_fgh.depth);
        b.write(m_fgh.shadowMask);
        b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
            auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice());
            rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
            auto rhiIb = rhi::VkRHIBuffer::createNonOwning(vkDev, m_indirectBufSun.handle(), VK_WHOLE_SIZE);
            auto rhiFrameUbo = rhi::VkRHIBuffer::createNonOwning(vkDev, m_renderer.gbuffer().frameUboHandle(), VK_WHOLE_SIZE);
            m_renderer.shadow().record(rhiCmd, m_renderer.rt(),
                *rhiFrameUbo,
                m_sceneGpu, *rhiIb, m_drawCount,
                m_renderer.frameIndex());
        });
    });

    // --- AO ---
    if (aoEnabled) {
        const char* aoName = (m_aoMethod == AOMethod::SSAO) ? "SSAO" : "GTAO";
        // FrameGraph auto-barrier: depth/normal → SR_O（entry），ssao → GENERAL（entry）
        //                           ssao → SR_O（exit，由 Lighting 的 read 触发）
        m_fg.addPass(aoName, [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.read(m_fgh.depth);
            b.read(m_fgh.gNormalRough);
            b.write(m_fgh.ssao);
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), vkCmd);
                if (m_aoMethod == AOMethod::SSAO) {
                    m_renderer.ssao().record(rhiCmd, m_renderer.rt(),
                        m_frameCtx.proj, glm::inverse(m_frameCtx.proj), m_frameCtx.view);
                } else {
                    m_renderer.gtao().record(rhiCmd, m_renderer.rt(),
                        m_frameCtx.proj, m_frameCtx.view);
                }
            });
        });
    } else if (!fwd) {
        // AO-Clear: 清除 ssao 为 1.0（无遮挡）
        // FrameGraph auto-barrier 负责 UNDEFINED→TRANSFER_DST（entry）和 TRANSFER_DST→SR_O（exit）
        m_fg.addPass("AO-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.ssao, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue white{};
                white.float32[0] = 1.0f; white.float32[1] = 1.0f;
                white.float32[2] = 1.0f; white.float32[3] = 1.0f;
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().ssao.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &range);
            });
        });
    } else {
        // Forward mode: AO not applicable
    }

    // --- SSR ---
    if (!fwd && m_renderer.ssr().enabled) {
        // FrameGraph auto-barrier: GBuffer → SR_O（entry），ssr → GENERAL（entry）
        //                           ssr → SR_O（exit，由 Lighting 的 read 触发）
        m_fg.addPass("SSR", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.read(m_fgh.gAlbedoMetal);
            b.read(m_fgh.gNormalRough);
            b.read(m_fgh.gEmissiveAO);
            b.read(m_fgh.depth);
            b.read(m_fgh.hdrPrev);
            b.write(m_fgh.ssr);
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), vkCmd);
                m_renderer.ssr().record(rhiCmd, m_renderer.rt());
            });
        });
    } else if (!fwd && !m_renderer.ssr().enabled) {
        m_fg.addPass("SSR-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.ssr, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().ssr.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            });
        });
    }

    // --- ScreenGI (SSGI/GTGI) ---
    bool screenGiOn = !fwd && (m_renderer.ssgi().enabled || m_renderer.gtgi().enabled);
    if (screenGiOn) {
        m_fg.addPass("ScreenGI", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：内部 copy+transition+dispatch
            b.read(m_fgh.gAlbedoMetal);
            b.read(m_fgh.gNormalRough);
            b.read(m_fgh.gEmissiveAO);
            b.read(m_fgh.depth);
            b.read(m_fgh.ssgiPrev);
            b.write(m_fgh.ssgi);
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice());
                    rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
                    auto& rt = m_renderer.rt();
                // Copy ssgi → ssgiPrev for temporal history
                auto transImg = [&](VkImage img, VkFormat fmt,
                                    rhi::TextureLayout oldL, rhi::TextureLayout newL) {
                    auto tex = rhi::VkRHITexture::createNonOwning(vkDev, img,
                        rhi::toRhiFormat(fmt), rt.extent.width, rt.extent.height, 1);
                    rhiCmd.textureBarrier(*tex, oldL, newL);
                };
                transImg(rt.ssgi.image(), rt.ssgi.format(),
                    rhi::TextureLayout::Undefined, rhi::TextureLayout::TransferSrc);
                transImg(rt.ssgiPrev.image(), rt.ssgiPrev.format(),
                    rhi::TextureLayout::Undefined, rhi::TextureLayout::TransferDst);

                VkImageCopy region{};
                region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.extent = {rt.extent.width, rt.extent.height, 1};
                vkCmdCopyImage(vkCmd,
                    rt.ssgi.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    rt.ssgiPrev.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &region);

                // ssgiPrev → SHADER_READ_ONLY for sampling
                transImg(rt.ssgiPrev.image(), rt.ssgiPrev.format(),
                    rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
                // ssgi → GENERAL for writing new value
                transImg(rt.ssgi.image(), rt.ssgi.format(),
                    rhi::TextureLayout::TransferSrc, rhi::TextureLayout::General);

                if (m_renderer.ssgi().enabled) m_renderer.ssgi().record(rhiCmd, rt);
                else m_renderer.gtgi().record(rhiCmd, rt);

                transImg(rt.ssgi.image(), rt.ssgi.format(),
                    rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
            });
        });
    } else if (!fwd) {
        m_fg.addPass("ScreenGI-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.ssgi, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().ssgi.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            });
        });
    }

    // --- VXGI Chain ---
    if (needVoxelGrid) {
        m_fg.addPass("VXGI-Chain", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：clear+voxelize+inject+mipmap+aniso 多步操作
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), vkCmd);
                // 1. Clear entire mip chain to 0
                auto barrierAllMips = [&](VkImageLayout oldL, VkImageLayout newL,
                                           VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                           VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
                    b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
                    b.oldLayout = oldL; b.newLayout = newL;
                    b.image = m_renderer.vxgi().image().image();
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                          0, m_renderer.vxgi().mipLevels(), 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(vkCmd, &di);
                };
                barrierAllMips(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                VkClearColorValue zeroV{};
                VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT,
                                           0, m_renderer.vxgi().mipLevels(), 0, 1};
                vkCmdClearColorImage(vkCmd, m_renderer.vxgi().image().image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zeroV, 1, &rg);
                barrierAllMips(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                // 2. Voxelize: scatter all primitives to mip 0
                m_renderer.vxgiVoxelize().record(rhiCmd, m_scene, m_sceneGpu,
                    m_renderer.vxgiGridMin(), m_renderer.vxgiCellSize(), m_renderer.kVxgiResolution);

                // 3. Inject: RSM flux → voxel mip 0 RGB
                {
                    VkImageMemoryBarrier2 vbar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    vbar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    vbar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    vbar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    vbar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    vbar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    vbar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    vbar.image = m_renderer.vxgi().image().image();
                    vbar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo vdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    vdi.imageMemoryBarrierCount = 1; vdi.pImageMemoryBarriers = &vbar;
                    vkCmdPipelineBarrier2(vkCmd, &vdi);
                }
                m_renderer.vxgiInject().record(rhiCmd, m_renderer.kVxgiResolution,
                    m_renderer.vxgiGridMin(), m_renderer.vxgiCellSize());

                // 4. Mipmap: iterate src mip i → dst mip i+1
                {
                    m_renderer.vxgiMipmap().record(rhiCmd, m_renderer.vxgi());
                }

                // 5. Final mip → SHADER_READ_ONLY
                {
                    VkImageMemoryBarrier2 fb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    fb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    fb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    fb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    fb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    fb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    fb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    fb.image = m_renderer.vxgi().image().image();
                    fb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                           m_renderer.vxgi().mipLevels() - 1, 1, 0, 1};
                    VkDependencyInfo fdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    fdi.imageMemoryBarrierCount = 1; fdi.pImageMemoryBarriers = &fb;
                    vkCmdPipelineBarrier2(vkCmd, &fdi);
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
                    ab.image = m_renderer.vxgi().aniso().image();
                    ab.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                           0, m_renderer.vxgi().mipLevels(), 0, 1};
                    VkDependencyInfo adi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    adi.imageMemoryBarrierCount = 1; adi.pImageMemoryBarriers = &ab;
                    vkCmdPipelineBarrier2(vkCmd, &adi);
                }
                {
                    m_renderer.vxgiAniso().record(rhiCmd, m_renderer.vxgi());
                }
            });
        });

        if (m_renderer.vxgiRelightEnabled()) {
            m_fg.addPass("VXGI-Relight", [&](FGBuilder& b) {
                b.setPassType(FGPassType::Compute);
                b.setManualBarriers();  // 复杂 Pass：multi-bounce ping-pong
                b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                    int bounces = m_renderer.lumenEnabled() ? 3 : 1;

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
                        transImg(m_renderer.vxgi().image().image(),
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
                        region.extent = {m_renderer.kVxgiResolution, m_renderer.kVxgiResolution, m_renderer.kVxgiResolution};
                        vkCmdCopyImage(cmd,
                            srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            m_renderer.vxgi().image().image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            1, &region);
                        transImg(m_renderer.vxgi().image().image(),
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COPY_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                    };

                    // Bounce 1: read voxelGrid → write scratch
                    transImg(m_renderer.vxgi().relightScratch().image(),
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                    {
                        rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), cmd);
                        m_renderer.vxgiRelight().record(rhiCmd, m_renderer.vxgiRelight().voxelSet(),
                            m_renderer.kVxgiResolution, m_renderer.vxgi().mipLevels(),
                            m_renderer.vxgiCellSize(), m_renderer.vxgiGridMin(),
                            m_renderer.vxgiRelightStrength());
                    }

                    if (bounces >= 2) {
                        transImg(m_renderer.vxgi().relightScratch().image(),
                            VK_IMAGE_LAYOUT_GENERAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                        transImg(m_renderer.vxgi().relightScratch2().image(),
                            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                        // Bounce 2: read scratch → write scratch2
                        {
                            rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), cmd);
                            m_renderer.vxgiRelight().record(rhiCmd, m_renderer.vxgiRelight().pingSet0(),
                                m_renderer.kVxgiResolution, m_renderer.vxgi().mipLevels(),
                                m_renderer.vxgiCellSize(), m_renderer.vxgiGridMin(),
                                m_renderer.vxgiRelightStrength());
                        }

                        if (bounces >= 3) {
                            transImg(m_renderer.vxgi().relightScratch2().image(),
                                VK_IMAGE_LAYOUT_GENERAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                            transImg(m_renderer.vxgi().relightScratch().image(),
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_LAYOUT_GENERAL,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                            // Bounce 3: read scratch2 → write scratch
                            {
                                rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), cmd);
                                m_renderer.vxgiRelight().record(rhiCmd, m_renderer.vxgiRelight().pingSet1(),
                                    m_renderer.kVxgiResolution, m_renderer.vxgi().mipLevels(),
                                    m_renderer.vxgiCellSize(), m_renderer.vxgiGridMin(),
                                    m_renderer.vxgiRelightStrength());
                            }
                            blitScratchToVoxel(m_renderer.vxgi().relightScratch().image());
                        } else {
                            blitScratchToVoxel(m_renderer.vxgi().relightScratch2().image());
                        }
                    } else {
                        blitScratchToVoxel(m_renderer.vxgi().relightScratch().image());
                    }
                });
            });
        }

        if (m_renderer.lumenEnabled() && m_renderer.vxgiSixAxisInited()) {
            m_fg.addPass("VXGI-6Axis", [&](FGBuilder& b) {
                b.setPassType(FGPassType::Compute);
                b.setManualBarriers();  // 复杂 Pass：多 image layout 过渡
                b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                    rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), vkCmd);
                    VkImageLayout axisOldL = m_renderer.lumenAtlasInited()
                        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                        : VK_IMAGE_LAYOUT_UNDEFINED;
                    VkPipelineStageFlags2 axisSrcS = m_renderer.lumenAtlasInited()
                        ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                        : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    VkAccessFlags2 axisSrcA = m_renderer.lumenAtlasInited()
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
                        vkCmdPipelineBarrier2(vkCmd, &di);
                    };
                    transAxisToGeneral(m_renderer.vxgi().sixAxisX().image());
                    transAxisToGeneral(m_renderer.vxgi().sixAxisY().image());
                    transAxisToGeneral(m_renderer.vxgi().sixAxisZ().image());

                    m_renderer.vxgi6Axis().record(rhiCmd, m_renderer.kVxgiResolution,
                        m_renderer.vxgi().mipLevels(), m_renderer.vxgiCellSize(),
                        m_renderer.vxgiGridMin(), m_renderer.vxgiRelightStrength());

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
                        vkCmdPipelineBarrier2(vkCmd, &di);
                    };
                    transAxisToSRO(m_renderer.vxgi().sixAxisX().image());
                    transAxisToSRO(m_renderer.vxgi().sixAxisY().image());
                    transAxisToSRO(m_renderer.vxgi().sixAxisZ().image());
                });
            });
        }
    }

    // --- SDFGI ---
    if (!fwd && m_renderer.sdfgiPass().enabled) {
        m_fg.addPass("SDFGI", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：bootstrap + compute
            b.read(m_fgh.gNormalRough);
            b.read(m_fgh.depth);
            b.read(m_fgh.ssgi);
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice());
                    rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
                auto& rt = m_renderer.rt();
                auto transImg = [&](VkImage img, VkFormat fmt,
                                    rhi::TextureLayout oldL, rhi::TextureLayout newL) {
                    auto tex = rhi::VkRHITexture::createNonOwning(vkDev, img,
                        rhi::toRhiFormat(fmt), rt.extent.width, rt.extent.height, 1);
                    rhiCmd.textureBarrier(*tex, oldL, newL);
                };
                if (!m_renderer.sdfgiBootstrapped()) {
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
                        vkCmdPipelineBarrier2(vkCmd, &di);
                    };
                    bootstrapToGeneral(m_renderer.sdfgi().seedA().image());
                    bootstrapToGeneral(m_renderer.sdfgi().seedB().image());
                    bootstrapToGeneral(m_renderer.sdfgi().udf().image());
                    m_renderer.sdfgiBootstrapped() = true;
                }
                transImg(m_renderer.rt().ssgi.image(), m_renderer.rt().ssgi.format(),
                    rhi::TextureLayout::Undefined, rhi::TextureLayout::General);

                m_renderer.sdfgiPass().record(rhiCmd, m_renderer.sdfgi(), m_renderer.rt(), m_renderer.frameIndex(),
                    m_renderer.sdfgiPass().seedThreshold, m_renderer.sdfgiPass().maxDistCells,
                    (uint32_t)m_renderer.sdfgiPass().numRays,
                    (uint32_t)m_renderer.sdfgiPass().maxSteps,
                    m_renderer.sdfgiPass().rayMaxCells, m_renderer.sdfgiPass().hitEpsCells);

                transImg(m_renderer.rt().ssgi.image(), m_renderer.rt().ssgi.format(),
                    rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
            });
        });
    }

    // --- RT GI ---
    bool rtGiActive = m_renderer.rtGiBound() && m_giIndexApplied == 10;
    if (!fwd && rtGiActive) {
        // FrameGraph auto-barrier: rtGI → GENERAL（entry），rtGI → SR_O（exit 由 Lighting read 触发）
        m_fg.addPass("RTGI", [&](FGBuilder& b) {
            b.setPassType(FGPassType::RayTracing);
            b.write(m_fgh.rtGI);
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), vkCmd);
                m_renderer.rtGi().record(rhiCmd, m_renderer.rt());
            });
        });
    } else if (!fwd && m_renderer.rtGiInited() && !rtGiActive) {
        m_fg.addPass("RTGI-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.rtGI, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().rtGI.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            });
        });
    }

    // --- ReSTIR ---
    if (!fwd && m_renderer.restirPass().enabled) {
        m_fg.addPass("ReSTIR", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：bootstrap + 条件 oldLayout 过渡
            b.write(m_fgh.restir);
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice());
                    rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
                auto& rt = m_renderer.rt();
                auto transImg = [&](VkImage img, VkFormat fmt,
                                    rhi::TextureLayout oldL, rhi::TextureLayout newL) {
                    auto tex = rhi::VkRHITexture::createNonOwning(vkDev, img,
                        rhi::toRhiFormat(fmt), rt.extent.width, rt.extent.height, 1);
                    rhiCmd.textureBarrier(*tex, oldL, newL);
                };
                m_renderer.restir().updateLights(m_renderer.demoLights());
                if (!m_renderer.restirBootstrapped()) {
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
                        vkCmdPipelineBarrier2(vkCmd, &di);
                    };
                    bootstrapToGeneral(m_renderer.restir().reservoirA().image());
                    bootstrapToGeneral(m_renderer.restir().reservoirB().image());
                    m_renderer.restirBootstrapped() = true;
                }
                rhi::TextureLayout restirOld = m_renderer.restirOutInited()
                    ? rhi::TextureLayout::ShaderReadOnly
                    : rhi::TextureLayout::Undefined;
                transImg(m_renderer.rt().restir.image(), m_renderer.rt().restir.format(),
                    restirOld, rhi::TextureLayout::General);
                m_renderer.restirOutInited() = true;

                uint32_t numLights = (uint32_t)m_renderer.demoLights().size();
                bool useRtVis = m_renderer.rtSupported() && m_renderer.rtGiBound();
                m_renderer.restirPass().record(rhiCmd, m_renderer.restir(), m_renderer.rt(),
                    numLights,
                    (uint32_t)m_renderer.restirPass().numCandidates,
                    (uint32_t)m_renderer.restirPass().numNeighbors,
                    m_renderer.restirPass().spatialRadius,
                    (uint32_t)m_renderer.restirPass().shadowSteps,
                    m_renderer.restirPass().intensityScale,
                    m_renderer.frameIndex(),
                    useRtVis);

                transImg(m_renderer.rt().restir.image(), m_renderer.rt().restir.format(),
                    rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
            });
        });
    } else if (!fwd && !m_renderer.restirPass().enabled) {
        m_fg.addPass("ReSTIR-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.restir, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().restir.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            });
        });
    }

    // --- DDGI ---
    if (!fwd && m_renderer.ddgiEnabled()) {
        m_fg.addPass("DDGI", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：多 atlas barrier 管理
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
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

                VkImageLayout oldAtlasL = m_renderer.ddgiAtlasInited()
                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    : VK_IMAGE_LAYOUT_UNDEFINED;
                VkAccessFlags2 srcAcc = m_renderer.ddgiAtlasInited()
                    ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;
                VkPipelineStageFlags2 srcStg = m_renderer.ddgiAtlasInited()
                    ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                    : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;

                barrierAtlas(m_renderer.ddgi().irradiance().image(),
                    oldAtlasL, VK_IMAGE_LAYOUT_GENERAL,
                    srcStg, srcAcc,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                barrierAtlas(m_renderer.ddgi().distance().image(),
                    oldAtlasL, VK_IMAGE_LAYOUT_GENERAL,
                    srcStg, srcAcc,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                float jitterRot = float((m_renderer.frameIndex() % 360) * 0.0174532925);
                rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), cmd);
                m_renderer.ddgiPass().record(rhiCmd, m_renderer.ddgi(), m_renderer.ddgiOrigin(), m_renderer.ddgiSpacing(),
                    m_renderer.vxgiGridMin(), m_renderer.vxgiCellSize(), m_renderer.kVxgiResolution,
                    jitterRot, m_renderer.frameIndex());

                barrierAtlas(m_renderer.ddgi().irradiance().image(),
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                barrierAtlas(m_renderer.ddgi().distance().image(),
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                m_renderer.ddgiAtlasInited() = true;
            });
        });
    }

    // --- Lumen ---
    bool lumenActive = !fwd && m_renderer.lumenEnabled();
    if (lumenActive) {
        m_fg.addPass("Lumen-Probe", [&](FGBuilder& b) {
            b.setPassType(FGPassType::RayTracing);
            b.setManualBarriers();  // 复杂 Pass：bootstrap + 多 image 过渡 + dispatch
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                if (!m_renderer.lumenProbeInited()) {
                    m_renderer.lumenProbe().init(*m_renderer.rhiDevice());
                    m_renderer.lumenProbe().bindResources(m_renderer.lumen(), m_renderer.rtAS(), m_sceneGpu,
                                                    m_renderer.vxgi(), m_renderer.rt(), m_renderer.gbuffer().frameUboHandle(),
                                                    m_renderer.vxgiSixAxisInited());
                    m_renderer.lumenProbeInited() = true;
                }
                // Transition probe + filtered atlas to GENERAL
                {
                    VkImageLayout oldL = m_renderer.lumenAtlasInited()
                        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                        : VK_IMAGE_LAYOUT_UNDEFINED;
                    VkPipelineStageFlags2 srcS = m_renderer.lumenAtlasInited()
                        ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                        : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    VkAccessFlags2 srcA = m_renderer.lumenAtlasInited()
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
                    transToGeneral(m_renderer.lumen().probeAtlas().image());
                    transToGeneral(m_renderer.lumen().filteredAtlas().image());
                    m_renderer.lumenAtlasInited() = true;
                }
                rhi::VkRHICommandBuffer rhiCmd2(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), cmd);
                m_renderer.lumenProbe().record(rhiCmd2, m_renderer.lumen(), m_renderer.frameIndex(),
                    m_renderer.vxgiSixAxisInited() ? 1u : 0u);

                // ProbeAtlas GENERAL → SR_O for filter
                {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    b.image = m_renderer.lumen().probeAtlas().image();
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                }
            });
        });
        m_fg.addPass("Lumen-Filter", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：init + dispatch + copy + 多 barrier
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                    rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), vkCmd);
                if (!m_renderer.lumenFilterInited()) {
                    VkImageMemoryBarrier2 pb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    pb.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    pb.srcAccessMask = 0;
                    pb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    pb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    pb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    pb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    pb.image = m_renderer.lumen().prevAtlas().image();
                    pb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo pdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    pdi.imageMemoryBarrierCount = 1; pdi.pImageMemoryBarriers = &pb;
                    vkCmdPipelineBarrier2(vkCmd, &pdi);

                    m_renderer.lumenFilter().init(*m_renderer.rhiDevice());
                    m_renderer.lumenFilter().bindResources( m_renderer.lumen(), m_renderer.rt(),
                                                     m_renderer.gbuffer().frameUboHandle());
                    m_renderer.lumenFilterInited() = true;
                }
                m_renderer.lumenFilter().record(rhiCmd, m_renderer.lumen(), m_renderer.rt());

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
                    vkCmdPipelineBarrier2(vkCmd, &di);
                };

                imgBarrier(m_renderer.lumen().filteredAtlas().image(),
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                imgBarrier(m_renderer.lumen().prevAtlas().image(),
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

                VkImageCopy region{};
                region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.extent = {m_renderer.lumen().atlasWidth(), m_renderer.lumen().atlasHeight(), 1};
                vkCmdCopyImage(vkCmd,
                    m_renderer.lumen().filteredAtlas().image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    m_renderer.lumen().prevAtlas().image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &region);

                imgBarrier(m_renderer.lumen().filteredAtlas().image(),
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                imgBarrier(m_renderer.lumen().prevAtlas().image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            });
        });
        m_fg.addPass("Lumen-Gather", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：bootstrap + 多 layout 过渡 + dispatch
            b.write(m_fgh.lumenGI);
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                    rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), vkCmd);
                if (!m_renderer.lumenGatherInited()) {
                    m_renderer.lumenGather().init(*m_renderer.rhiDevice());
                    m_renderer.lumenGather().bindResources(m_renderer.lumen(), m_renderer.rt(),
                                                     m_renderer.gbuffer().frameUboHandle(), true);
                    m_renderer.lumenGatherInited() = true;
                }
                {
                    VkImageLayout oldL = m_renderer.lumenOutInited()
                        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                        : VK_IMAGE_LAYOUT_UNDEFINED;
                    VkAccessFlags2 srcA = m_renderer.lumenOutInited()
                        ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;
                    VkPipelineStageFlags2 srcS = m_renderer.lumenOutInited()
                        ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                        : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = srcS; b.srcAccessMask = srcA;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.oldLayout = oldL;
                    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.image = m_renderer.rt().lumenGI.image();
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(vkCmd, &di);
                }
                m_renderer.lumenGather().record(rhiCmd, m_renderer.lumen(), m_renderer.rt(),
                    (uint32_t)m_renderer.lumenDebugMode());

                {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    b.image = m_renderer.rt().lumenGI.image();
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(vkCmd, &di);
                }
                m_renderer.lumenOutInited() = true;
            });
        });
    } else if (!fwd) {
        m_fg.addPass("Lumen-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.lumenGI, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().lumenGI.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            });
        });
    }

    // --- LPV ---
    if (!fwd && m_renderer.lpvEnabled()) {
        m_fg.addPass("LPV-Inject", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：compute + exit transition
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice());
                    rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
                m_renderer.lpvInject().record(rhiCmd, m_renderer.kLpvResolution,
                    m_renderer.lpvGridMin(), m_renderer.lpvCellSize());

                {
                    auto gvTex = rhi::VkRHITexture::createNonOwning(vkDev,
                        m_renderer.lpv().gv().image(),
                        rhi::toRhiFormat(m_renderer.lpv().gv().format()),
                        m_renderer.kLpvResolution, m_renderer.kLpvResolution, 1);
                    rhiCmd.textureBarrier(*gvTex,
                        rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
                }
            });
        });
        m_fg.addPass("LPV-Propagate", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：multi-iteration ping-pong barrier
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice());
                    rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
                auto lpvRes = m_renderer.kLpvResolution;
                auto barrierLpv = [&](const LpvGrid& g,
                                      rhi::TextureLayout oldL, rhi::TextureLayout newL) {
                    VkImage imgs[3] = {g.lpvR.image(), g.lpvG.image(), g.lpvB.image()};
                    for (auto img : imgs) {
                        auto tex = rhi::VkRHITexture::createNonOwning(vkDev, img,
                            rhi::toRhiFormat(VK_FORMAT_R16G16B16A16_SFLOAT),
                            lpvRes, lpvRes, 1);
                        rhiCmd.textureBarrier(*tex, oldL, newL);
                    }
                };

                int propIter = m_renderer.lpvProp().iterations & ~1;
                for (int it = 0; it < propIter; ++it) {
                    LpvGrid& src = m_renderer.lpv().current();
                    LpvGrid& dst = m_renderer.lpv().next();

                    // 首次迭代 src 布局未知，使用 Undefined；后续迭代 src 来自上轮 dst 写入的 General
                    rhi::TextureLayout srcOldL = (it == 0) ? rhi::TextureLayout::Undefined
                                                           : rhi::TextureLayout::General;
                    barrierLpv(src, srcOldL, rhi::TextureLayout::ShaderReadOnly);
                    barrierLpv(dst, rhi::TextureLayout::Undefined, rhi::TextureLayout::General);

                    m_renderer.lpvProp().record(rhiCmd, m_renderer.lpv().curIdx(),
                        m_renderer.kLpvResolution, m_renderer.lpvProp().occlusionAmplifier,
                        m_renderer.lpvProp().gvOcclusionStrength);
                    m_renderer.lpv().swap();
                }

                barrierLpv(m_renderer.lpv().current(),
                    rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly);
            });
        });
    }

    // --- RSM Sample ---
    if (!fwd && m_renderer.rsmSample().enabled) {
        // FrameGraph auto-barrier: rsmGI → GENERAL（entry），rsmGI → SR_O（exit 由 Lighting read 触发）
        m_fg.addPass("RSM-Sample", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.rsmGI);
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), vkCmd);
                m_renderer.rsmSample().record(rhiCmd, m_renderer.rt());
            });
        });
    } else if (!fwd && !m_renderer.rsmSample().enabled) {
        m_fg.addPass("RSM-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.rsmGI, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().rsmGI.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            });
        });
    }

    // --- NDGI ---
    if (m_renderer.ndgiEnabled() && m_renderer.rtSupported()) {
        m_fg.addPass("NDGI", [&](FGBuilder& b) {
            b.setPassType(FGPassType::RayTracing);
            b.setManualBarriers();  // RT dispatch，无 read/write 声明，内部管理 barrier
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), vkCmd);
                m_renderer.ndgiPass().record(rhiCmd, m_renderer.ndgi(), m_renderer.frameIndex(),
                    m_renderer.ddgiOrigin(), m_renderer.ddgiSpacing());
                m_renderer.ndgiPass().recordTraining(rhiCmd, m_renderer.ndgi(), m_renderer.frameIndex());
            });
        });
    }

    // --- Lighting ---
    if (!fwd) {
        // FrameGraph auto-barrier: 所有 GBuffer+GI → SR_O（entry），hdrColor → GENERAL（entry+exit）
        m_fg.addPass("Lighting", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.read(m_fgh.gAlbedoMetal);
            b.read(m_fgh.gNormalRough);
            b.read(m_fgh.gEmissiveAO);
            b.read(m_fgh.depth);
            b.read(m_fgh.ssao);
            b.read(m_fgh.ssr);
            b.read(m_fgh.ssgi);
            b.read(m_fgh.shadowMask);
            // Optional GI reads (if GI is active, but framegraph can handle extra deps)
            b.read(m_fgh.rsmGI);
            b.read(m_fgh.rtGI);
            b.read(m_fgh.restir);
            b.read(m_fgh.lumenGI);
            b.write(m_fgh.hdrColor);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), cmd);
                m_renderer.lighting().record(rhiCmd, m_renderer.rt());
            });
        });
    }

    // --- Skybox ---
    // Manual barrier：depth 在 render pass 内作为 DEPTH_ATTACHMENT，与 Lighting
    // 等 compute pass 的 SHADER_READ_ONLY descriptor 冲突，submit-time 无法同时满足
    if (!fwd) {
        m_fg.addPass("Skybox", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Graphics);
            b.setManualBarriers();
            b.read(m_fgh.depth);
            b.write(m_fgh.hdrColor);
            b.setExitLayout(m_fgh.hdrColor, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            b.setExitLayout(m_fgh.depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice());
                rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
                auto& rt = m_renderer.rt();

                rhi::TextureLayout depthOld = (m_renderer.frameIndex() == 0)
                    ? rhi::TextureLayout::Undefined
                    : rhi::TextureLayout::ShaderReadOnly;
                auto depthTex = rhi::VkRHITexture::createNonOwning(vkDev,
                    rt.depth.image(), rhi::Format::D32_SFLOAT,
                    rt.extent.width, rt.extent.height, 1);
                rhiCmd.textureBarrier(*depthTex, depthOld, rhi::TextureLayout::DepthAttachment);

                rhi::TextureLayout hdrOld = (m_renderer.frameIndex() == 0)
                    ? rhi::TextureLayout::Undefined
                    : rhi::TextureLayout::General;
                auto hdrTex = rhi::VkRHITexture::createNonOwning(vkDev,
                    rt.hdrColor.image(), rhi::toRhiFormat(rt.hdrColor.format()),
                    rt.extent.width, rt.extent.height, 1);
                rhiCmd.textureBarrier(*hdrTex, hdrOld, rhi::TextureLayout::ColorAttachment);

                m_renderer.skybox().record(rhiCmd, rt);

                rhiCmd.textureBarrier(*depthTex, rhi::TextureLayout::DepthAttachment,
                    rhi::TextureLayout::ShaderReadOnly);
            });
        });
    }

    // --- Copy hdrPrev ---
    // FrameGraph auto-barrier: hdrColor → TRANSFER_SRC, hdrPrev → TRANSFER_DST（entry）
    //                          hdrPrev → SHADER_READ_ONLY（exit, 由下一帧读触发）
    m_fg.addPass("Copy-hdrPrev", [&](FGBuilder& b) {
        b.setPassType(FGPassType::Compute);
        b.read(m_fgh.hdrColor, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        b.write(m_fgh.hdrPrev, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.extent = {m_renderer.rt().extent.width, m_renderer.rt().extent.height, 1};
            vkCmdCopyImage(cmd,
                m_renderer.rt().hdrColor.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_renderer.rt().hdrPrev.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &region);
        });
    });

    // --- Tonemap（AA 模式下纳入 FrameGraph，非 AA 模式仍由 recordPostProcessing 处理）---
    if (aaEnabled && m_swap->hdrEnabled()) {
        m_fg.addPass("Tonemap", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.read(m_fgh.hdrColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            b.write(m_fgh.aaHdr);
            b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), vkCmd);
                m_renderer.rt().ensureAaResources(*m_device);
                m_renderer.tonemap().bindOutput(m_renderer.rt().aaHdr.view(), m_frameCtx.frameInFlight);
                m_renderer.tonemap().record(rhiCmd, m_renderer.rt(), m_frameCtx.frameInFlight, true, 1.0f);
            });
        });

        // --- TAA / SMAA ---
        // 读取 aaHdr + depth(+aaHistory in TAA)，写入 swapImage
        if (m_aaMethod == AAMethod::TAA) {
            m_fg.addPass("TAA", [&](FGBuilder& b) {
                b.setPassType(FGPassType::Compute);
                b.read(m_fgh.aaHdr, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                b.read(m_fgh.aaHistory, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                b.read(m_fgh.depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                b.write(m_fgh.swapImage, VK_IMAGE_LAYOUT_GENERAL);
                b.setExecute([this](VkCommandBuffer vkCmd, const FGResources&) {
                    rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), vkCmd);
                    m_renderer.taa().bindResources(m_renderer.rt(), m_frameCtx.frameInFlight);
                    m_renderer.taa().bindOutput(m_frameCtx.swapView, m_frameCtx.frameInFlight);
                    m_renderer.taa().record(rhiCmd, m_renderer.rt(), m_jitter, m_prevJitter,
                        m_frameCtx.invViewProj, m_prevViewProj, m_frameCtx.frameInFlight, m_taaBlendAlpha);
                });
            });

            // --- Copy aaHdr → aaHistory（下一帧 TAA 的历史输入）---
            m_fg.addPass("Copy-aaHistory", [&](FGBuilder& b) {
                b.setPassType(FGPassType::Compute);
                b.read(m_fgh.aaHdr, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                b.write(m_fgh.aaHistory, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                    VkImageCopy region{};
                    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    region.extent = {m_renderer.rt().extent.width, m_renderer.rt().extent.height, 1};
                    vkCmdCopyImage(cmd,
                        m_renderer.rt().aaHdr.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        m_renderer.rt().aaHistory.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &region);
                });
            });
        } else {
            // SMAA：只需 aaHdr 输入，内部 m_edgeTex 自行管理 barrier
            m_fg.addPass("SMAA", [&](FGBuilder& b) {
                b.setPassType(FGPassType::Compute);
                b.read(m_fgh.aaHdr, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                b.write(m_fgh.swapImage, VK_IMAGE_LAYOUT_GENERAL);
                b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                    rhi::VkRHICommandBuffer rhiCmd(static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice()), cmd);
                    m_renderer.smaa().bindResources(m_renderer.rt());
                    m_renderer.smaa().bindOutput(m_frameCtx.swapView);
                    m_renderer.smaa().record(rhiCmd, m_renderer.rt());
                });
            });
        }
    }

    // 注：TAA/SMAA/Copy-aaHistory 的 barrier 由 FrameGraph 自动管理。
    //      ImGui 和最终 swapImage→PRESENT 过渡仍由 recordPostProcessing 处理。
}

// ================================================================
// setupFgImports — 导入所有持久纹理资源到 FrameGraph
// ================================================================
void App::setupFgImports(VkExtent3D ext, bool aaEnabled) {
    using namespace somegi::fg;
    auto& rt = m_renderer.rt();

    // GBuffer (resolved, single-sample)
    m_fgh.gAlbedoMetal = m_fg.importTexture("gAlbedoMetal", rt.gAlbedoMetal.image(),
        {ext, rt.gAlbedoMetal.format(), 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.gNormalRough = m_fg.importTexture("gNormalRough", rt.gNormalRough.image(),
        {ext, rt.gNormalRough.format(), 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.gEmissiveAO = m_fg.importTexture("gEmissiveAO", rt.gEmissiveAO.image(),
        {ext, rt.gEmissiveAO.format(), 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.depth = m_fg.importTexture("depth", rt.depth.image(),
        {ext, VK_FORMAT_D32_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // GBuffer MSAA
    m_fgh.gAlbedoMetalMs = m_fg.importTexture("gAlbedoMetalMs", rt.gAlbedoMetalMs.image(),
        {ext, rt.gAlbedoMetalMs.format(), 1, 1, m_msaaSamples,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT}, VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.gNormalRoughMs = m_fg.importTexture("gNormalRoughMs", rt.gNormalRoughMs.image(),
        {ext, rt.gNormalRoughMs.format(), 1, 1, m_msaaSamples,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT}, VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.gEmissiveAOMs = m_fg.importTexture("gEmissiveAOMs", rt.gEmissiveAOMs.image(),
        {ext, rt.gEmissiveAOMs.format(), 1, 1, m_msaaSamples,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT}, VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.depthMs = m_fg.importTexture("depthMs", rt.depthMs.image(),
        {ext, VK_FORMAT_D32_SFLOAT, 1, 1, m_msaaSamples,
         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT}, VK_IMAGE_LAYOUT_UNDEFINED);

    // AO
    m_fgh.ssao = m_fg.importTexture("ssao", rt.ssao.image(),
        {ext, VK_FORMAT_R8_UNORM, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // SSR
    m_fgh.ssr = m_fg.importTexture("ssr", rt.ssr.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // SSGI
    m_fgh.ssgi = m_fg.importTexture("ssgi", rt.ssgi.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.ssgiPrev = m_fg.importTexture("ssgiPrev", rt.ssgiPrev.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // HDR
    m_fgh.hdrColor = m_fg.importTexture("hdrColor", rt.hdrColor.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.hdrPrev = m_fg.importTexture("hdrPrev", rt.hdrPrev.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // LDR
    m_fgh.ldrTonemap = m_fg.importTexture("ldrTonemap", rt.ldrTonemap.image(),
        {ext, VK_FORMAT_B8G8R8A8_UNORM, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // RT GI
    m_fgh.rtGI = m_fg.importTexture("rtGI", rt.rtGI.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // ReSTIR
    m_fgh.restir = m_fg.importTexture("restir", rt.restir.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // RSM GI
    m_fgh.rsmGI = m_fg.importTexture("rsmGI", rt.rsmGI.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // Lumen GI
    m_fgh.lumenGI = m_fg.importTexture("lumenGI", rt.lumenGI.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // Shadow mask
    m_fgh.shadowMask = m_fg.importTexture("shadowMask",
        m_renderer.shadow().shadowMask().image(),
        {ext, VK_FORMAT_R8_UNORM, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // AA intermediates
    if (aaEnabled) {
        m_fgh.aaHdr = m_fg.importTexture("aaHdr", rt.aaHdr.image(),
            {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
            VK_IMAGE_LAYOUT_UNDEFINED);
        m_fgh.aaHistory = m_fg.importTexture("aaHistory", rt.aaHistory.image(),
            {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
            VK_IMAGE_LAYOUT_UNDEFINED);
    }

    // Swapchain（每帧导入，不使用 debugName 避免 persistentState 跨帧错乱）
    m_fgh.swapImage = m_fg.importTexture(nullptr, m_frameCtx.swapImage,
        {ext, m_swap->format(), 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);
}

} // namespace somegi
