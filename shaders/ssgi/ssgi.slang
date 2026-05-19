// 屏幕空间一次反弹全局光照（漫反射）—— M4.3。
//
// 算法：
// 1. 每像素从 GBuffer / depth 重建世界法线 wN 与世界位置 wPos。
// 2. 在 wN 周围的 cosine-weighted 半球内取 sampleCount 条方向（在 tangent
//    空间用 Hammersley 序列 + per-pixel 哈希抖动让邻居走不同方向，
//    弱化 banding）。
// 3. 沿每个方向用 marchScreenWorld 在世界空间线性 march；命中处用
//    线性 sampler 采 gPrevHdr 拿这一帧（实际是上一帧）的 radiance。
// 4. 累加命中 radiance / 计数命中数；输出 rgba16f：
//      rgb = avgL（hits 的 radiance 平均，未上 BRDF）
//      a   = hits / sampleCount（"覆盖比例"）
// 5. LightingPass 用 a 当混合权重把 evalIBLDiffuse 与 ssgi.rgb
//    （applyIBLDiffuseBRDF 后）lerp。命中越多 → 越偏 SSGI 真值；
//    命中少（rays 飞屏外） → 回落 IBL diffuse fallback。
//
// 噪点：8 sample 噪点明显，per-pixel 哈希让噪点呈高频白噪声特性。
// lighting.slang 端 5×5 bilateral 减噪，没有时序累积（M7+）。
//
// 反弹深度：仅 1 bounce，被照亮的表面再反射出去时 hdrPrev 已经包含
// 上一帧的 1-bounce → 多帧后会涌现 multi-bounce 的视觉效果（feedback）。

import shared_types;
import screen_ray;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] Texture2D<float4> gNormalRough;
[[vk::binding(2, 0)]] Texture2D<float>  gDepth;
[[vk::binding(3, 0)]] Texture2D<float4> gPrevHdr;
[[vk::binding(4, 0)]] SamplerState      gLinearClamp;
[[vk::binding(5, 0)]] RWTexture2D<float4> gOutSsgi;
[[vk::binding(6, 0)]] Texture2D<float4> gPrevSsgi;        // B.4 时序：上一帧 SSGI（reproject 后 sample）

struct SsgiPC {
    uint2 outSize;
    float2 invOutSize;
    uint  maxSteps;
    float maxDist;
    float thickness;
    uint  sampleCount;
};
[[vk::push_constant]] SsgiPC gPC;

float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gFrame.invViewProj, clip);
    return w.xyz / w.w;
}

float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

uint hash2(uint2 p) {
    uint h = p.x * 73856093u ^ p.y * 19349663u;
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13;
    return h;
}

// Hammersley sequence with per-pixel offset so adjacent pixels don't trace
// the same directions (reduces obvious banding in low-sample noise).
float2 hammersleyJittered(uint i, uint N, uint2 pix) {
    float jitter = float(hash2(pix)) * (1.0 / 4294967296.0);
    float u = frac(float(i) / float(N) + jitter);
    float v = radicalInverse_VdC(i);
    return float2(u, v);
}

// Cosine-weighted hemisphere sample in tangent space (Z is up). Maps
// [0,1]^2 → unit hemisphere via concentric disk + projection.
float3 cosineHemisphere(float2 xi) {
    float r = sqrt(xi.x);
    float phi = 6.2831853 * xi.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - xi.x)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) return;
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) { gOutSsgi[pix] = float4(0, 0, 0, 0); return; }

    float4 nr = gNormalRough.Load(int3(pix, 0));
    float3 wN = normalize(nr.xyz);
    float3 wPos = worldFromDepth(pix, depth);

    // Tangent basis around N.
    float3 up = abs(wN.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 T = normalize(cross(up, wN));
    float3 B = cross(wN, T);

    float stepLen = gPC.maxDist / float(max(gPC.maxSteps, 1u));
    // Push origin slightly along N to avoid self-intersection on first step.
    float3 origin = wPos + wN * stepLen * 0.5;

    float3 sumColor = float3(0);
    uint   hits = 0;
    for (uint i = 0; i < gPC.sampleCount; ++i) {
        float2 xi = hammersleyJittered(i, gPC.sampleCount, pix);
        float3 dirTan   = cosineHemisphere(xi);
        float3 dirWorld = T * dirTan.x + B * dirTan.y + wN * dirTan.z;

        ScreenRayHit h = marchScreenWorld(
            origin, dirWorld, stepLen, gPC.maxSteps, gPC.thickness,
            gFrame.viewProj, gDepth, gPC.outSize);
        if (h.hit) {
            float3 c = gPrevHdr.SampleLevel(gLinearClamp, h.uv, 0).rgb;
            sumColor += c;
            hits++;
        }
    }

    float3 avgL = (hits > 0) ? sumColor / float(hits) : float3(0);
    float hitFrac = float(hits) / float(gPC.sampleCount);
    float4 newVal = float4(avgL, hitFrac);

    // B.4 时序累积：把 wPos 投到上一帧 NDC，sample 上一帧 SSGI；
    // 在 [0,1] 范围内 lerp 当 history。alpha=0.92 → ~12 帧半衰减，
    // 镜头快速移动时新区域几帧补满，camera 静止时噪点显著降低。
    float3 wPosCur = wPos;
    float4 prevClip = mul(gFrame.prevViewProj, float4(wPosCur, 1.0));
    if (prevClip.w > 0.0) {
        float2 prevNdc = prevClip.xy / prevClip.w;
        if (all(abs(prevNdc) <= 1.0)) {
            float2 prevUv = prevNdc * 0.5 + 0.5;
            float4 prevVal = gPrevSsgi.SampleLevel(gLinearClamp, prevUv, 0);
            // 简单 history 累积；不做 variance clamping（够 noise 了，
            // 但可能 ghosting 重运动物体；目前场景静态可接受）。
            newVal = lerp(newVal, prevVal, 0.92);
        }
    }
    gOutSsgi[pix] = newVal;
}
