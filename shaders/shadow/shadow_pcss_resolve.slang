// shadow_pcss_resolve.slang — PCSS (Percentage Closer Soft Shadows) resolve。
//
// 三步算法：
//   1. Blocker search：在搜索区域内采样 shadow map，找出遮挡物平均深度
//   2. Penumbra 估算：根据遮挡物距离和光源大小计算半影半径
//   3. 自适应 kernel PCF：kernel 大小与半影成正比，接触面=硬边，远离=软边
//
// 管线约定：
//   set=0:
//     binding 0: ConstantBuffer<ShadowUbo>       — sunViewProj
//     binding 1: Texture2D (SAMPLED_IMAGE)       — shadow map
//     binding 2: SamplerState                     — shadow sampler
//     binding 3: RWTexture2D<float>               — shadowMask
//   set=1:
//     binding 0: ConstantBuffer<FrameUniforms>    — invViewProj
//     binding 1: Texture2D<float>                 — gDepth
//   push constant: PCSS_PC

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

// PCSS 专用 push constant（比 ResolvePC 多 lightSize + maxKernel）
struct PCSS_PC {
    uint2  outSize;
    float2 invOutSize;
    float  bias;
    uint   _pad;
    float2 invShadowMapSize;
    float  lightSize;         // 太阳角半径（radians），典型值 0.01～0.05
    float  maxKernelRadius;   // 最大 PCF kernel（texels），典型值 10～25
};
[[vk::push_constant]] PCSS_PC gPC;

// ── Poisson disk 采样（16 点），用于 blocker search 和 PCF ──
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

    // ── 1. GBuffer depth → worldPos → sun space ──
    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) { gOutMask[pix] = 1.0; return; }

    float3 worldPos = worldFromDepth(pix, depth);
    float4 sunClip = mul(gShadow.sunViewProj, float4(worldPos, 1.0));
    float3 sunNdc = sunClip.xyz / sunClip.w;

    if (any(sunNdc.xy < -1.0) || any(sunNdc.xy > 1.0) || sunNdc.z < 0.0 || sunNdc.z > 1.0) {
        gOutMask[pix] = 1.0;
        return;
    }

    float2 uv = sunNdc.xy * 0.5 + 0.5;
    float receiverDepth = sunNdc.z - gPC.bias;

    // ── 2. Blocker search ──
    //     搜索半径与光源大小 / receiverDepth 成正比（近处搜小，远处搜大）
    float searchRadius = gPC.lightSize * gPC.maxKernelRadius / max(receiverDepth, 0.001);
    searchRadius = clamp(searchRadius, 2.0, gPC.maxKernelRadius);
    float2 searchStep = searchRadius * gPC.invShadowMapSize;

    float blockerSum = 0.0;
    uint  blockerCount = 0;

    for (int i = 0; i < 16; ++i) {
        float2 sampleUv = uv + kPoisson16[i] * searchStep;
        float smDepth = gShadowMap.SampleLevel(gShadowSampler, sampleUv, 0).r;

        // receiverDepth <= smDepth 表示接收点在遮挡面之前 → 非遮挡物
        // receiverDepth >  smDepth 表示接收点在遮挡面之后 → 遮挡物
        if (receiverDepth > smDepth) {
            blockerSum += smDepth;
            blockerCount++;
        }
    }

    // ── 3. Penumbra 估算 ──
    float shadow;
    if (blockerCount == 0) {
        // 无遮挡物 → 完全 lit
        shadow = 1.0;
        gOutMask[pix] = shadow;
        return;
    }

    float avgBlockerDepth = blockerSum / float(blockerCount);

    // 半影大小：d_blocker / d_receiver 比例 × 光源角半径
    // 遮挡物离接收点越近 → kernel 越小 → 阴影越硬（接触硬化）
    float penumbraRatio = (receiverDepth - avgBlockerDepth) / max(avgBlockerDepth, 0.001);
    float kernelRadius = max(1.0, penumbraRatio * gPC.lightSize * gPC.maxKernelRadius * 100.0);
    kernelRadius = min(kernelRadius, gPC.maxKernelRadius);
    float2 pcfStep = kernelRadius * gPC.invShadowMapSize;

    // ── 4. 自适应 kernel PCF ──
    float litCount = 0.0;
    // 根据 kernel 大小选择采样数：大kernel用16，小kernel用8
    int sampleCount = (kernelRadius > 5.0) ? 16 : 8;

    for (int i = 0; i < sampleCount; ++i) {
        float2 sampleUv = uv + kPoisson16[i] * pcfStep;
        float smDepth = gShadowMap.SampleLevel(gShadowSampler, sampleUv, 0).r;
        if (receiverDepth <= smDepth) {
            litCount += 1.0;
        }
    }
    shadow = litCount / float(sampleCount);

    gOutMask[pix] = shadow;
}
