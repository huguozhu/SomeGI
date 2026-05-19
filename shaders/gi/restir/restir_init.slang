// ReSTIR DI initial pass —— C.4 第一步。
//
// 每像素 M 个 uniform 候选，按 unshadowed targetPdf 做 RIS 收敛到 1 个
// reservoir。targetPdf = brdf_lambert × Le × max(0, dot(N, L)) / dist²。
//
// visibility 不在此处算（shade 阶段统一做）；RIS 权重 = targetPdf / sourcePdf
// （sourcePdf = 1/N_lights，因为均匀采样）。
//
// 输入：GBuffer + 光源 SSBO；输出：reservoir image A。
// 输出 layout：RWTexture2D<uint4>，pack 见 restir_common。

import shared_types;
import restir_common;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] Texture2D<float4>            gAlbedoMetal;
[[vk::binding(2, 0)]] Texture2D<float4>            gNormalRough;
[[vk::binding(3, 0)]] Texture2D<float>             gDepth;
[[vk::binding(4, 0)]] StructuredBuffer<PointLight> gLights;
[[vk::binding(5, 0)]] RWTexture2D<uint4>           gReservoirOutA;

struct InitPC {
    uint2  outSize;
    float2 invOutSize;
    uint   numLights;
    uint   numCandidates;     // M（论文里典型 32；demo 8）
    uint   frameIndex;
    uint   _pad0;
};
[[vk::push_constant]] InitPC gPC;

float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc.x, -ndc.y, depth, 1.0);
    float4 v = mul(gFrame.invViewProj, clip);
    return v.xyz / v.w;
}

// 计算给定像素 + 给定 light 的 unshadowed target pdf p_hat（标量）。
float targetPdf(float3 worldPos, float3 N, float3 albedo,
                PointLight L) {
    float3 toL = L.pos - worldPos;
    float dist2 = max(dot(toL, toL), 1e-4);
    float invDist = rsqrt(dist2);
    float3 Ldir = toL * invDist;
    float NdotL = max(0.0, dot(N, Ldir));
    if (NdotL <= 0.0) return 0.0;
    // brdf_lambert = albedo / PI；归并 albedo 进 p_hat 即可（后续 shade 也乘 albedo）
    float3 Li = L.color * L.intensity;
    // p_hat 用 luminance 缩成标量（RIS 通常如此）
    float3 contribution = albedo * Li * NdotL / dist2;
    return dot(contribution, float3(0.2126, 0.7152, 0.0722));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (any(dt.xy >= gPC.outSize)) return;
    uint2 pix = dt.xy;

    Reservoir r = emptyReservoir();

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) {
        gReservoirOutA[pix] = packReservoir(r);
        return;
    }

    float4 nr = gNormalRough.Load(int3(pix, 0));
    float3 N = normalize(nr.xyz);
    float4 am = gAlbedoMetal.Load(int3(pix, 0));
    float3 albedo = am.rgb * (1.0 - am.a);   // metallic 部分扣掉（粗略 diffuse）
    float3 wp = worldFromDepth(pix, depth);

    if (gPC.numLights == 0u) {
        gReservoirOutA[pix] = packReservoir(r);
        return;
    }

    uint base = pcgHash(pix.x * 1973u + pix.y * 9277u + gPC.frameIndex * 26699u);
    float invSrcPdf = float(gPC.numLights);   // sourcePdf = 1/N → 1/sourcePdf = N

    for (uint i = 0u; i < gPC.numCandidates; ++i) {
        uint h = pcgHash(base + i * 2654435761u);
        uint cand = h % gPC.numLights;
        float xi = hashFloat(base + i * 2654435761u + 7u);
        PointLight L = gLights[cand];
        float p_hat = targetPdf(wp, N, albedo, L);
        float w = p_hat * invSrcPdf;          // = p_hat / (1/N)
        reservoirUpdate(r, cand, w, xi);
    }

    // 计算最终 W = wSum / (M * p_hat(picked))
    if (r.lightIdx != 0xFFFFFFFFu && r.M > 0u) {
        PointLight Lp = gLights[r.lightIdx];
        float p_hat_picked = targetPdf(wp, N, albedo, Lp);
        if (p_hat_picked > 0.0) {
            r.W = r.wSum / (float(r.M) * p_hat_picked);
        } else {
            r.W = 0.0;
        }
    }
    gReservoirOutA[pix] = packReservoir(r);
}
