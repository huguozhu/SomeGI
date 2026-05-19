// 线性屏幕空间反射 —— M4.2。
// 沿反射方向在世界空间步进，在 depth buffer 上比对查找命中点，
// 命中处采样上一帧 hdrPrev 的颜色当反射光。
//
// 设计选择（详见 docs/.../2026-05-18-m4-ssgi.md Task 11）：
//   - 世界空间 march：比 screen-space DDA 简单；M4.2 精度够用，Hi-Z
//     优化留给后续里程碑。
//   - 数据源是 gPrevHdr（上一帧 lighting 输出）：避开"本帧 lighting →
//     SSR → lighting"的同帧依赖循环。代价：反射有 1 帧延迟，相机静止
//     时无视觉影响，快速移动时轻微 ghosting。
//   - 输出 rgba：rgb = 反射颜色（已经是 incoming radiance，不带 BRDF），
//                a = [0,1] 置信度 fade。
//   - LightingPass 用 a 在 IBL specular 与 ssrSpec 之间 lerp，所以
//     高粗糙度像素 SSR 的 a 接近 0 自然回落 IBL；屏幕边沿 a 渐变避免
//     硬切。

import shared_types;
import screen_ray;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] Texture2D<float4> gNormalRough;
[[vk::binding(2, 0)]] Texture2D<float>  gDepth;
[[vk::binding(3, 0)]] Texture2D<float4> gPrevHdr;
[[vk::binding(4, 0)]] SamplerState      gLinearClamp;
[[vk::binding(5, 0)]] RWTexture2D<float4> gOutSsr;

struct SsrPC {
    uint2 outSize;
    float2 invOutSize;
    uint  maxSteps;
    float maxDist;          // total world-space march distance
    float thickness;        // depth-comparison tolerance in NDC z
    float roughThreshold;   // pixels with roughness > this skip SSR (uses IBL)
};
[[vk::push_constant]] SsrPC gPC;

float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gFrame.invViewProj, clip);
    return w.xyz / w.w;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) return;
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) { gOutSsr[pix] = float4(0,0,0,0); return; }

    float4 nr = gNormalRough.Load(int3(pix, 0));
    float roughness = nr.w;
    if (roughness > gPC.roughThreshold) {
        gOutSsr[pix] = float4(0,0,0,0);   // a=0 → IBL spec used as-is
        return;
    }

    float3 wPos = worldFromDepth(pix, depth);
    float3 wN   = normalize(nr.xyz);
    float3 V    = normalize(gFrame.cameraPos.xyz - wPos);
    float3 R    = reflect(-V, wN);

    // Step length scales with maxDist / maxSteps.
    float stepLen = gPC.maxDist / float(max(gPC.maxSteps, 1u));

    // Push the origin off the surface a bit along the reflection direction
    // to avoid self-intersection on the first step.
    float3 origin = wPos + R * stepLen * 0.5;

    ScreenRayHit hit = marchScreenWorld(
        origin, R, stepLen, gPC.maxSteps, gPC.thickness,
        gFrame.viewProj, gDepth, gPC.outSize);

    if (!hit.hit) { gOutSsr[pix] = float4(0,0,0,0); return; }

    float3 reflColor = gPrevHdr.SampleLevel(gLinearClamp, hit.uv, 0).rgb;

    // Confidence fades:
    // 1) screen-edge fade (avoids hard cutoff at viewport boundary)
    float2 e = abs(hit.uv * 2.0 - 1.0);
    float edgeFade = 1.0 - smoothstep(0.85, 1.0, max(e.x, e.y));
    // 2) roughness fade (smooth handoff into IBL above roughThreshold)
    float roughFade = 1.0 - smoothstep(gPC.roughThreshold * 0.5,
                                       gPC.roughThreshold, roughness);
    float fade = edgeFade * roughFade;

    gOutSsr[pix] = float4(reflColor, fade);
}
