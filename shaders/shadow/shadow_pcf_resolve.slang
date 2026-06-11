// shadow_pcf_resolve.slang — PCF soft shadow resolve compute shader。
//
// 使用 Poisson disk 16 采样替代原始 3×3 网格 PCF：
//   - 无网格走样伪影
//   - 更宽的软阴影过渡带（kernel 半径 ≈ 3 texels）
//
// 每像素：
//   1. 从 GBuffer depth 重建世界坐标 worldPos
//   2. 变换到 sun light clip space → NDC → UV
//   3. Poisson disk PCF 采样（手动深度比较）
//   4. 输出样本平均值：1.0 = full lit, 0.0 = full shadow
//
// 管线约定：
//   set=0:
//     binding 0: ConstantBuffer<ShadowUbo>       — sunViewProj
//     binding 1: Texture2D                       — shadow map
//     binding 2: SamplerState                     — shadow sampler
//     binding 3: RWTexture2D<float>               — shadowMask
//   set=1:
//     binding 0: ConstantBuffer<FrameUniforms>    — invViewProj
//     binding 1: Texture2D<float>                 — gDepth
//   push constant: ResolvePC

import shared_types;

struct ShadowUbo {
    float4x4 sunViewProj;
};
[[vk::binding(0, 0)]] ConstantBuffer<ShadowUbo>      gShadow;
[[vk::binding(1, 0)]] Texture2D<float>               gShadowMap;
[[vk::binding(2, 0)]] SamplerState                   gShadowSampler;
[[vk::binding(3, 0)]] RWTexture2D<float>             gOutMask;

[[vk::binding(0, 1)]] ConstantBuffer<FrameUniforms>   gFrame;
[[vk::binding(1, 1)]] Texture2D<float>                gDepth;

struct ResolvePC {
    uint2 outSize;
    float2 invOutSize;
    float bias;
    uint _pad;
    float2 invShadowMapSize;  // 1/width, 1/height —— texel 步长
};
[[vk::push_constant]] ResolvePC gPC;

// ── Poisson disk 16 采样点 ──
static const float2 kPoisson16[16] = {
    float2(-0.3262, -0.2358), float2(-0.0913,  0.3674),
    float2( 0.1935, -0.3961), float2( 0.3741,  0.0821),
    float2(-0.4638,  0.0883), float2( 0.0127, -0.1339),
    float2( 0.2319,  0.2978), float2(-0.1876, -0.4062),
    float2( 0.4193, -0.1851), float2(-0.4014,  0.3487),
    float2( 0.0598,  0.4615), float2(-0.2722, -0.0564),
    float2( 0.1489, -0.2593), float2(-0.3594, -0.2965),
    float2( 0.3052, -0.0418), float2(-0.0831,  0.2043),
};

// ── 从 GBuffer depth 重建世界坐标 ──
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

    // ── 4. Poisson disk PCF ──
    float2 uv = sunNdc.xy * 0.5 + 0.5;
    float receiverDepth = sunNdc.z - gPC.bias;
    float2 texelSize = gPC.invShadowMapSize;

    // kernel 半径（texels）：3.0 产生中等宽度的软过渡
    // 可调大（5～8）获得更宽的 penumbra，或调小（1～2）接近 Hard SM
    float kernelRadius = 3.0;

    float shadow = 0.0;
    for (int i = 0; i < 16; ++i) {
        float2 sampleUv = uv + kPoisson16[i] * texelSize * kernelRadius;
        float sampleDepth = gShadowMap.SampleLevel(gShadowSampler, sampleUv, 0).r;
        if (receiverDepth <= sampleDepth) {
            shadow += 1.0;
        }
    }
    shadow /= 16.0;

    gOutMask[pix] = shadow;
}
