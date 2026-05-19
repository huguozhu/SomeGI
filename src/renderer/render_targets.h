#pragma once
#include "core/image.h"

// 渲染目标集合 —— 所有屏幕分辨率级别的离屏图集中放在这里，create() 一次
// 全部分配，destroy() 一次全部回收。窗口 resize 时由 App 走"销毁 + 重建"。

namespace somegi {

class Device;

struct RenderTargets {
    Image hdrColor;       // R16G16B16A16_SFLOAT，主光照输出
    Image depth;          // D32_SFLOAT，几何深度（GBuffer prepass 写）
    Image ldrTonemap;     // B8G8R8A8_UNORM, STORAGE+TRANSFER_SRC，tonemap 输出

    // M4 GBuffer：GBufferPass 写，LightingPass / SS* passes 采样
    Image gAlbedoMetal;   // RGBA8_UNORM:  rgb=baseColor (线性), a=metallic
    Image gNormalRough;   // RGBA16_SFLOAT: xyz=世界法线归一化, w=roughness
    Image gEmissiveAO;    // RGBA8_UNORM:  rgb=emissive, a=AO（材质 occlusion）

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

    void create(Device& d, VkExtent2D ext);
    void destroy();

    VkExtent2D extent{};
};

}
