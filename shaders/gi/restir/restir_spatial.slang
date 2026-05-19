// ReSTIR DI spatial reuse pass —— C.4 第二步。
//
// 每像素读自身 reservoir + K 个邻居 reservoir，按邻居在当前像素位置的
// targetPdf 重新加权，合并 → 写第二张 reservoir image。
//
// 邻居挑选：屏幕空间 radius 内随机抖动；用 normal/depth 阈值剔除明显
// 不同表面（避免穿墙复用）。
//
// 重要：合并时的 targetPdfAtCenter 是 "把邻居选中的 light 用到当前像素
// 位置算出的 p_hat"（不是邻居那时候算的）。这样 RIS 估计在当前位置无偏。

import shared_types;
import restir_common;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] Texture2D<float4>             gAlbedoMetal;
[[vk::binding(2, 0)]] Texture2D<float4>             gNormalRough;
[[vk::binding(3, 0)]] Texture2D<float>              gDepth;
[[vk::binding(4, 0)]] StructuredBuffer<PointLight>  gLights;
[[vk::binding(5, 0)]] Texture2D<uint4>              gReservoirInA;
[[vk::binding(6, 0)]] RWTexture2D<uint4>            gReservoirOutB;

struct SpatialPC {
    uint2  outSize;
    float2 invOutSize;
    uint   numLights;
    uint   numNeighbors;     // K（典型 4）
    float  radiusPixels;
    uint   frameIndex;
};
[[vk::push_constant]] SpatialPC gPC;

float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc.x, -ndc.y, depth, 1.0);
    float4 v = mul(gFrame.invViewProj, clip);
    return v.xyz / v.w;
}

float targetPdf(float3 worldPos, float3 N, float3 albedo,
                PointLight L) {
    float3 toL = L.pos - worldPos;
    float dist2 = max(dot(toL, toL), 1e-4);
    float invDist = rsqrt(dist2);
    float3 Ldir = toL * invDist;
    float NdotL = max(0.0, dot(N, Ldir));
    if (NdotL <= 0.0) return 0.0;
    float3 Li = L.color * L.intensity;
    float3 contribution = albedo * Li * NdotL / dist2;
    return dot(contribution, float3(0.2126, 0.7152, 0.0722));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (any(dt.xy >= gPC.outSize)) return;
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    Reservoir r = unpackReservoir(gReservoirInA.Load(int3(pix, 0)));

    if (depth >= 1.0) {
        gReservoirOutB[pix] = packReservoir(r);
        return;
    }

    float4 nr = gNormalRough.Load(int3(pix, 0));
    float3 N = normalize(nr.xyz);
    float4 am = gAlbedoMetal.Load(int3(pix, 0));
    float3 albedo = am.rgb * (1.0 - am.a);
    float3 wp = worldFromDepth(pix, depth);

    uint base = pcgHash(pix.x * 1973u + pix.y * 9277u + gPC.frameIndex * 11369u);

    // Reservoir 合并需要保留 wSum / M，新建一个 combined reservoir 从 r 起步
    Reservoir combined = r;

    for (uint i = 0u; i < gPC.numNeighbors; ++i) {
        // 随机邻居偏移（圆盘内）
        float xi1 = hashFloat(base + i * 2u);
        float xi2 = hashFloat(base + i * 2u + 1u);
        float ang = xi1 * 6.2831853;
        float rad = sqrt(xi2) * gPC.radiusPixels;
        int2  off = int2(int(round(cos(ang) * rad)), int(round(sin(ang) * rad)));
        int2  np  = clamp(int2(pix) + off, int2(0), int2(gPC.outSize) - 1);
        if (all(np == int2(pix))) continue;

        // depth / normal 相似性剔除
        float ndepth = gDepth.Load(int3(np, 0));
        if (ndepth >= 1.0) continue;
        if (abs(ndepth - depth) > 0.05) continue;
        float3 Nn = normalize(gNormalRough.Load(int3(np, 0)).xyz);
        if (dot(Nn, N) < 0.9) continue;

        Reservoir rn = unpackReservoir(gReservoirInA.Load(int3(np, 0)));
        if (rn.lightIdx == 0xFFFFFFFFu || rn.M == 0u) continue;

        // 邻居的 picked light 用 *本像素* 的 p_hat 重算
        PointLight Ln = gLights[rn.lightIdx];
        float p_hat_here = targetPdf(wp, N, albedo, Ln);
        float xi3 = hashFloat(base + i * 2u + 100u);
        reservoirCombine(combined, rn, p_hat_here, xi3);
    }

    // 合并完后重算 W = wSum / (M * p_hat(picked))
    if (combined.lightIdx != 0xFFFFFFFFu && combined.M > 0u) {
        PointLight Lp = gLights[combined.lightIdx];
        float p_hat = targetPdf(wp, N, albedo, Lp);
        combined.W = (p_hat > 0.0) ? combined.wSum / (float(combined.M) * p_hat) : 0.0;
    }
    gReservoirOutB[pix] = packReservoir(combined);
}
