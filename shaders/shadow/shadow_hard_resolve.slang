// shadow_hard_resolve.slang — Hard shadow map resolve compute shader。
//
// 每像素：
//   1. 从 GBuffer depth 重建世界坐标 worldPos
//   2. 将 worldPos 变换到 sun light clip space
//   3. 采样 shadow map，手动比较深度（与 gPC.bias 偏移）
//   4. 输出 R8 shadow mask：1.0 = lit, 0.0 = shadowed
//
// 管线约定：
//   set=0: ShadowPass 拥有的资源
//     binding 0: ConstantBuffer<ShadowUbo>       — sunViewProj
//     binding 1: Texture2D + SamplerState         — shadow map (COMBINED_IMAGE_SAMPLER)
//     binding 2: RWTexture2D<float>               — shadowMask (STORAGE_IMAGE)
//   set=1: GBuffer 资源（由调用方传入）
//     binding 0: ConstantBuffer<FrameUniforms>    — invViewProj for world reconstruction
//     binding 1: Texture2D<float>                 — gDepth
//   push constant: ResolvePC

import shared_types;

// ── Shadow UBO ──
struct ShadowUbo {
    float4x4 sunViewProj;
};
[[vk::binding(0, 0)]] ConstantBuffer<ShadowUbo>      gShadow;
[[vk::binding(1, 0)]] Texture2D<float>               gShadowMap;
[[vk::binding(2, 0)]] SamplerState                   gShadowSampler;
[[vk::binding(3, 0)]] RWTexture2D<float>             gOutMask;

// ── GBuffer 资源（set=1）──
[[vk::binding(0, 1)]] ConstantBuffer<FrameUniforms>   gFrame;
[[vk::binding(1, 1)]] Texture2D<float>                gDepth;

// ── Push constants：与 C++ 端 ResolvePC 对齐 ──
struct ResolvePC {
    uint2 outSize;
    float2 invOutSize;
    float bias;
    uint _pad;
    float2 invShadowMapSize;  // PCF 用；hard resolve 不使用
};
[[vk::push_constant]] ResolvePC gPC;

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
        gOutMask[pix] = 1.0;   // 天空/无限远 → lit
        return;
    }

    // ── 2. 重建世界坐标 ──
    float3 worldPos = worldFromDepth(pix, depth);

    // ── 3. 变换到 sun light clip space ──
    float4 sunClip = mul(gShadow.sunViewProj, float4(worldPos, 1.0));
    float3 sunNdc = sunClip.xyz / sunClip.w;

    // 检查是否在 shadow map NDC 范围内
    if (any(sunNdc.xy < -1.0) || any(sunNdc.xy > 1.0) || sunNdc.z < 0.0 || sunNdc.z > 1.0) {
        gOutMask[pix] = 1.0;   // 超出 shadow map 范围 → lit
        return;
    }

    // ── 4. 转换为 shadow map UV（NDC [-1,1] → UV [0,1]）──
    float2 uv = sunNdc.xy * 0.5 + 0.5;

    // ── 5. 手动深度比较（带 bias 偏移）──
    float shadowDepth = gShadowMap.SampleLevel(gShadowSampler, uv, 0).r;
    float receiverDepth = sunNdc.z - gPC.bias;
    float shadow = (receiverDepth <= shadowDepth) ? 1.0 : 0.0;

    gOutMask[pix] = shadow;
}
