// shadow_pcf_resolve.slang — PCF soft shadow resolve compute shader。
//
// 与 shadow_hard_resolve 相似，但用 3×3 的 Percentage-Closer Filtering
// 替代单点深度比较，产生软阴影边缘过渡。
//
// 每像素：
//   1. 从 GBuffer depth 重建世界坐标 worldPos
//   2. 变换到 sun light clip space → NDC → UV
//   3. 在 shadow map 上做 3×3 PCF 采样（每采样做手动深度比较）
//   4. 输出 9 样本平均值：1.0 = full lit, 0.0 = full shadow
//
// 管线约定：
//   set=0:
//     binding 0: ConstantBuffer<ShadowUbo>       — sunViewProj
//     binding 1: Texture2D + SamplerState         — shadow map
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

    // ── 4. UV + 3×3 PCF ──
    float2 uv = sunNdc.xy * 0.5 + 0.5;
    float receiverDepth = sunNdc.z - gPC.bias;

    // texel 大小从 push constant 传入，与 shadow map 实际分辨率一致
    float2 texelSize = gPC.invShadowMapSize;

    float shadow = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 sampleUv = uv + float2(x, y) * texelSize;
            float sampleDepth = gShadowMap.SampleLevel(gShadowSampler, sampleUv, 0).r;
            if (receiverDepth <= sampleDepth) {
                shadow += 1.0;
            }
        }
    }
    shadow /= 9.0;

    gOutMask[pix] = shadow;
}
