#pragma once
#include "core/image.h"
#include <memory>

// 渲染目标集合 —— 所有屏幕分辨率级别的离屏图集中放在这里，create() 一次
// 全部分配，destroy() 一次全部回收。窗口 resize 时由 App 走"销毁 + 重建"。

namespace somegi {

class Device;
namespace rhi {
    class RHIDevice;
    class RHITexture;
    class RHITextureView;
}

struct RenderTargets {
    Image hdrColor;       // R16G16B16A16_SFLOAT，主光照输出
    Image depth;          // D32_SFLOAT，几何深度（GBuffer prepass 写）
    Image ldrTonemap;     // B8G8R8A8_UNORM, STORAGE+TRANSFER_SRC，tonemap 输出

    // M4 GBuffer：GBufferPass 写，LightingPass / SS* passes 采样
    Image gAlbedoMetal;   // RGBA8_UNORM:  rgb=baseColor (线性), a=metallic
    Image gNormalRough;   // RGBA16_SFLOAT: xyz=世界法线归一化, w=roughness
    Image gEmissiveAO;    // RGBA8_UNORM:  rgb=emissive, a=AO（材质 occlusion）

    // MSAA 多重采样版本：GBufferPass 渲染目标，resolve 到 SS 版本
    Image gAlbedoMetalMs; // MSAA RGBA8_UNORM
    Image gNormalRoughMs; // MSAA RGBA16_SFLOAT
    Image gEmissiveAOMs;  // MSAA RGBA8_UNORM
    Image depthMs;        // MSAA D32_SFLOAT

    // M4.1 SSAO：全分辨率，half-res 优化留给后续
    Image ssao;           // R8_UNORM: 1.0=完全无遮蔽, 0.0=完全遮蔽

    // M4.2 SSR + hdrPrev（上一帧 HDR，避免本帧自依赖）
    Image ssr;            // RGBA16F: rgb=反射颜色, a=置信度/fade
    Image hdrPrev;        // RGBA16F: 上一帧 hdrColor 的副本，SSR/SSGI 都从此采样

    // M4.3 SSGI 一次反弹漫反射
    Image ssgi;           // RGBA16F: rgb=半球平均入射 radiance, a=hit fraction
    // B.4 时序累积：上一帧 ssgi 副本（reproject 后 lerp 当 history）
    Image ssgiPrev;       // RGBA16F：与 ssgi 同 layout，每帧 lighting 后 copy 过来

    // M5 RSM gather 输出（RsmSamplePass 写入，LightingPass 读）。
    // RSM 的 4 张几何 RT（depth/position/normal/flux）固定 512² 不跟
    // swapchain，归 RsmGeometryPass 自己持有，不放这里。
    Image rsmGI;          // RGBA16F: rgb=一次反弹间接 radiance, a=覆盖置信度

    // C.4 ReSTIR DI：屏幕空间直接光累加（多 point light reservoir 重采样
    // 后的 shaded radiance）。lighting 端 += 之；关闭时 clear 到 0。
    Image restir;         // RGBA16F: rgb=ReSTIR DI direct radiance, a=valid flag

    // M9 RT GI：硬件光线追踪全局光照（Ray Query 风格）。
    // RGBA16F: rgb=入射 radiance（命中点处 albedo·sunBRDF），a=hit?1:0。
    Image rtGI;           // STORAGE | SAMPLED | TRANSFER_DST

    // L.5 Lumen-lite：screen probe gather 最终间接 diffuse。
    // RGBA16F: rgb=间接 diffuse radiance, a=1.0。
    Image lumenGI;        // STORAGE | SAMPLED | TRANSFER_DST

    // AA intermediate targets (only valid when ensureAaResources has been called)
    Image aaHdr;          // RGBA16F: tonemap output (HDR before LDR conversion)
    Image aaHistory;      // RGBA16F: previous frame for TAA

    void create(Device& d, VkExtent2D ext, VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_4_BIT);
    void destroy();
    void recreateMsaa(Device& d, VkSampleCountFlagBits samples);

    // D3D12 RHI 纹理创建（与 Vulkan Image 并行）
    void createRHI(rhi::RHIDevice& rhiDev, VkExtent2D ext, VkSampleCountFlagBits msaaSamples);

    // Vulkan 路径：从已创建的 Image 成员构造 non-owning RHI 包装，使 RHI 访问器在
    // Vulkan 后端也能返回非 null。
    void populateRHITargetsFromImages(rhi::RHIDevice& rhiDev);

    void ensureAaResources(Device& d);
    void destroyAaResources();

    VkExtent2D extent{};

    // RHI 纹理指针（D3D12 路径使用，Vulkan 为 nullptr）
    struct RHITargets {
        std::unique_ptr<rhi::RHITexture> hdrColor, depth, ldrTonemap;
        std::unique_ptr<rhi::RHITexture> gAlbedoMetal, gNormalRough, gEmissiveAO;
        std::unique_ptr<rhi::RHITexture> gAlbedoMetalMs, gNormalRoughMs, gEmissiveAOMs, depthMs;
        std::unique_ptr<rhi::RHITexture> ssao, ssr, hdrPrev, ssgi, ssgiPrev;
        std::unique_ptr<rhi::RHITexture> rsmGI, restir, rtGI, lumenGI, aaHdr, aaHistory;
        std::unique_ptr<rhi::RHITextureView> hdrColorView, depthView, ldrTonemapView;
        std::unique_ptr<rhi::RHITextureView> gAlbedoMetalView, gNormalRoughView, gEmissiveAOView;
        std::unique_ptr<rhi::RHITextureView> ssaoView, ssrView, ssgiView;
        std::unique_ptr<rhi::RHITextureView> hdrPrevView, ssgiPrevView;
        std::unique_ptr<rhi::RHITextureView> rsmGIView, restirView, rtGIView, lumenGIView;
        std::unique_ptr<rhi::RHITextureView> aaHdrView, aaHistoryView;
    };
    RHITargets rhi{};

    // ════════════════════════════════════════════════════════════════
    // RHI 访问器 —— Vulkan 和 D3D12 双后端均返回非 null（Vulkan 需先调用
    // populateRHITargetsFromImages）。
    // ════════════════════════════════════════════════════════════════

    // 纹理视图访问器（用于绑定为 descriptor）
    rhi::RHITextureView* rhiHdrColorView() const { return rhi.hdrColorView.get(); }
    rhi::RHITextureView* rhiDepthView() const { return rhi.depthView.get(); }
    rhi::RHITextureView* rhiLdrTonemapView() const { return rhi.ldrTonemapView.get(); }
    rhi::RHITextureView* rhiGAlbedoMetalView() const { return rhi.gAlbedoMetalView.get(); }
    rhi::RHITextureView* rhiGNormalRoughView() const { return rhi.gNormalRoughView.get(); }
    rhi::RHITextureView* rhiGEmissiveAOView() const { return rhi.gEmissiveAOView.get(); }
    rhi::RHITextureView* rhiSsaoView() const { return rhi.ssaoView.get(); }
    rhi::RHITextureView* rhiSsrView() const { return rhi.ssrView.get(); }
    rhi::RHITextureView* rhiSsgiView() const { return rhi.ssgiView.get(); }
    rhi::RHITextureView* rhiHdrPrevView() const { return rhi.hdrPrevView.get(); }
    rhi::RHITextureView* rhiSsgiPrevView() const { return rhi.ssgiPrevView.get(); }
    rhi::RHITextureView* rhiRsmGIView() const { return rhi.rsmGIView.get(); }
    rhi::RHITextureView* rhiRestirView() const { return rhi.restirView.get(); }
    rhi::RHITextureView* rhiRtGIView() const { return rhi.rtGIView.get(); }
    rhi::RHITextureView* rhiLumenGIView() const { return rhi.lumenGIView.get(); }
    rhi::RHITextureView* rhiAaHdrView() const { return rhi.aaHdrView.get(); }
    rhi::RHITextureView* rhiAaHistoryView() const { return rhi.aaHistoryView.get(); }

    // 纹理访问器（用于 barrier、clear、copy 等操作）
    rhi::RHITexture* rhiHdrColor() const { return rhi.hdrColor.get(); }
    rhi::RHITexture* rhiDepth() const { return rhi.depth.get(); }
    rhi::RHITexture* rhiLdrTonemap() const { return rhi.ldrTonemap.get(); }
    rhi::RHITexture* rhiGAlbedoMetal() const { return rhi.gAlbedoMetal.get(); }
    rhi::RHITexture* rhiGNormalRough() const { return rhi.gNormalRough.get(); }
    rhi::RHITexture* rhiGEmissiveAO() const { return rhi.gEmissiveAO.get(); }
    rhi::RHITexture* rhiSsao() const { return rhi.ssao.get(); }
    rhi::RHITexture* rhiSsr() const { return rhi.ssr.get(); }
    rhi::RHITexture* rhiHdrPrev() const { return rhi.hdrPrev.get(); }
    rhi::RHITexture* rhiSsgi() const { return rhi.ssgi.get(); }
    rhi::RHITexture* rhiSsgiPrev() const { return rhi.ssgiPrev.get(); }
    rhi::RHITexture* rhiRsmGI() const { return rhi.rsmGI.get(); }
    rhi::RHITexture* rhiRestir() const { return rhi.restir.get(); }
    rhi::RHITexture* rhiRtGI() const { return rhi.rtGI.get(); }
    rhi::RHITexture* rhiLumenGI() const { return rhi.lumenGI.get(); }
    rhi::RHITexture* rhiAaHdr() const { return rhi.aaHdr.get(); }
    rhi::RHITexture* rhiAaHistory() const { return rhi.aaHistory.get(); }
};

}
