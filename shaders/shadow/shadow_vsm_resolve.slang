// shadow_vsm_resolve.slang — VSM Chebyshev resolve compute shader。
//
// 从 VSM 双分量纹理中读取 (depth, depth²) 两个矩，使用 Chebyshev 不等式
// 估算阴影值。这是纯概率方法，无需多点采样即可产生柔和阴影。
//
// VSM 算法：
//   均值 μ = M1 = E[depth]（滤波后）
//   方差 σ² = E[depth²] - E[depth]² = M2 - μ²
//   切比雪夫不等式：P(occluderDepth ≥ receiverDepth) ≤ σ² / (σ² + Δ²)
//     其中 Δ = receiverDepth - μ（正数 = 接收点在光照物体之后）
//
//   Δ ≤ 0 → receiver 在平均 occluder 前面 → fully lit (shadow = 1.0)
//   Δ > 0 → 应用 Chebyshev 上界作为阴影概率
//
//   min variance（少量 variance bias）防止过锐阴影（light bleeding 平衡）。
//
// 管线约定：
//   set=0:
//     binding 0: ConstantBuffer<ShadowUbo>       — sunViewProj
//     binding 1: Texture2D + SamplerState         — vsm map (R32G32_SFLOAT)
//     binding 2: RWTexture2D<float>               — shadowMask
//   set=1:
//     binding 0: ConstantBuffer<FrameUniforms>    — invViewProj
//     binding 1: Texture2D<float>                 — gDepth
//   push constant: ResolvePC

import shared_types;

struct ShadowUbo {
    float4x4 sunViewProj;
};
[[vk::binding(0, 0)]] ConstantBuffer<ShadowUbo>      gShadow;
[[vk::binding(1, 0)]] Texture2D<float2>              gVsmMap;
[[vk::binding(2, 0)]] SamplerState                   gShadowSampler;
[[vk::binding(3, 0)]] RWTexture2D<float>             gOutMask;

[[vk::binding(0, 1)]] ConstantBuffer<FrameUniforms>   gFrame;
[[vk::binding(1, 1)]] Texture2D<float>                gDepth;

struct ResolvePC {
    uint2 outSize;
    float2 invOutSize;
    float bias;
    uint _pad;
    float2 invShadowMapSize;  // PCF 用；VSM resolve 不使用
};
[[vk::push_constant]] ResolvePC gPC;

float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gFrame.invViewProj, clip);
    return w.xyz / w.w;
}

// ── Chebyshev 上界计算 ──
// moment.x = E[depth] (mean)，moment.y = E[depth²]
float chebyshevUpperBound(float2 moments, float receiverDepth, float minVariance) {
    float mean = moments.x;
    float variance = moments.y - mean * mean;
    variance = max(variance, minVariance);

    float d = receiverDepth - mean;
    if (d <= 0.0) return 1.0;   // 接收点在平均 occluder 前面 → fully lit

    float pMax = variance / (variance + d * d);

    // 减少 light bleeding：当方差很大时上界过于保守，施加 clamp
    // 对应原始 VSM 论文中的 "light bleeding reduction" 缩放到 [0,1]
    return saturate(pMax);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) return;
    uint2 pix = dt.xy;

    // ── 1. 读取 GBuffer depth ──
    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) {
        gOutMask[pix] = 1.0;
        return;
    }

    // ── 2. 重建世界坐标 ──
    float3 worldPos = worldFromDepth(pix, depth);

    // ── 3. 变换到 sun light clip space ──
    float4 sunClip = mul(gShadow.sunViewProj, float4(worldPos, 1.0));
    float3 sunNdc = sunClip.xyz / sunClip.w;

    if (any(sunNdc.xy < -1.0) || any(sunNdc.xy > 1.0) || sunNdc.z < 0.0 || sunNdc.z > 1.0) {
        gOutMask[pix] = 1.0;
        return;
    }

    // ── 4. UV 变换 ──
    float2 uv = sunNdc.xy * 0.5 + 0.5;
    float receiverDepth = sunNdc.z - gPC.bias;

    // ── 5. 采样 VSM 双矩 ──
    float2 moments = gVsmMap.SampleLevel(gShadowSampler, uv, 0).rg;

    // ── 6. Chebyshev 阴影估算 ──
    // minVariance 防止过锐阴影边缘（允许少量光渗透）
    float minVariance = 0.00002;
    float shadow = chebyshevUpperBound(moments, receiverDepth, minVariance);

    gOutMask[pix] = shadow;
}
