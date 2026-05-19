// GTGI（Ground-Truth Global Illumination，Sucker Punch 2024 / Ghost of
// Tsushima 2）—— GTAO 的 GI 升级。同 horizon-search 骨架，但每个"raise
// horizon"的 sample 不仅记录角度变化，还从 hdrPrev 取 radiance 加权累积，
// 输出一次反弹间接光。
//
// 与 SSGI（cosine 半球 ray march）的差别：
//   SSGI：每像素 N 条 ray，沿 ray march 屏幕找命中点，取 hdrPrev radiance。
//         半球均匀采样、噪点高、依赖 sampleCount。
//   GTGI：每像素 N slice × M sample 沿 slice 方向屏幕扫水平角，"raise
//         horizon"的 sample 当作可见间接光源，取 hdrPrev radiance 加权积分。
//         空间相干性强（slice 内 sample 链）→ 噪点显著低，sampleCount
//         可降到 4×4=16 总样本。
//
// 输出 RGBA16F：rgb = GI radiance，a = coverage。与 SSGI 共享 rt.ssgi。
// lighting.slang 读 ssgi 不区分来源 —— SSGI 与 GTGI 互斥。
//
// 与 SSGI 的 binding / 时序累积一致，复用 ssgiPrev reproject + lerp。

import shared_types;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] Texture2D<float4> gNormalRough;
[[vk::binding(2, 0)]] Texture2D<float>  gDepth;
[[vk::binding(3, 0)]] Texture2D<float4> gPrevHdr;
[[vk::binding(4, 0)]] SamplerState      gLinearClamp;
[[vk::binding(5, 0)]] RWTexture2D<float4> gOutGtgi;
[[vk::binding(6, 0)]] Texture2D<float4> gPrevGtgi;       // 时序 history（同 SSGI）

struct GtgiPC {
    uint2  outSize;
    float2 invOutSize;
    float  radiusPixels;
    float  falloff;          // view-space 单位的远 occluder 衰减
    uint   sliceCount;
    uint   samplesPerSlice;
};
[[vk::push_constant]] GtgiPC gPC;

float3 viewFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 v = mul(gFrame.invViewProj, clip);
    float3 wp = v.xyz / v.w;
    return mul(gFrame.view, float4(wp, 1.0)).xyz;
}

float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gFrame.invViewProj, clip);
    return w.xyz / w.w;
}

uint hash2(uint2 p) {
    uint h = p.x * 73856093u ^ p.y * 19349663u;
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) return;
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) { gOutGtgi[pix] = float4(0); return; }

    float3 vP = viewFromDepth(pix, depth);
    float3 V = normalize(-vP);

    float rotJitter = float(hash2(pix)) * (1.0 / 4294967296.0);

    float3 sumRadiance = float3(0);
    float coverageSum = 0.0;

    for (uint s = 0; s < gPC.sliceCount; ++s) {
        float phi = 3.14159265 * (float(s) / float(gPC.sliceCount) + rotJitter);
        float2 sliceDir2D = float2(cos(phi), sin(phi));

        // horizon search 双向。每次"提升水平角"时记录 radiance + 权重。
        float cosH1 = -1.0;
        float cosH2 = -1.0;
        float3 radPos = float3(0);  float wPos = 0.0;
        float3 radNeg = float3(0);  float wNeg = 0.0;

        for (uint i = 1; i <= gPC.samplesPerSlice; ++i) {
            float t = float(i) / float(gPC.samplesPerSlice);
            float pixelOffset = t * gPC.radiusPixels;

            // +dir
            int2 p1 = int2(pix) + int2(sliceDir2D * pixelOffset);
            if (all(p1 >= int2(0)) && all(p1 < int2(gPC.outSize))) {
                float d1 = gDepth.Load(int3(p1, 0));
                if (d1 < 1.0) {
                    float3 vS = viewFromDepth(uint2(p1), d1);
                    float3 D = vS - vP;
                    float lenD = length(D);
                    if (lenD > 1e-3) {
                        float cosT = dot(D / lenD, V);
                        float fall = saturate(1.0 - lenD / gPC.falloff);
                        cosT = lerp(-1.0, cosT, fall);
                        if (cosT > cosH1) {
                            // 提升 horizon → 这个 sample 是可见的间接光源
                            float dCos = cosT - cosH1;
                            float2 uv = (float2(p1) + 0.5) * gPC.invOutSize;
                            float3 rad = gPrevHdr.SampleLevel(gLinearClamp, uv, 0).rgb;
                            radPos += rad * dCos;
                            wPos += dCos;
                            cosH1 = cosT;
                        }
                    }
                }
            }
            // -dir
            int2 p2 = int2(pix) - int2(sliceDir2D * pixelOffset);
            if (all(p2 >= int2(0)) && all(p2 < int2(gPC.outSize))) {
                float d2 = gDepth.Load(int3(p2, 0));
                if (d2 < 1.0) {
                    float3 vS = viewFromDepth(uint2(p2), d2);
                    float3 D = vS - vP;
                    float lenD = length(D);
                    if (lenD > 1e-3) {
                        float cosT = dot(D / lenD, V);
                        float fall = saturate(1.0 - lenD / gPC.falloff);
                        cosT = lerp(-1.0, cosT, fall);
                        if (cosT > cosH2) {
                            float dCos = cosT - cosH2;
                            float2 uv = (float2(p2) + 0.5) * gPC.invOutSize;
                            float3 rad = gPrevHdr.SampleLevel(gLinearClamp, uv, 0).rgb;
                            radNeg += rad * dCos;
                            wNeg += dCos;
                            cosH2 = cosT;
                        }
                    }
                }
            }
        }

        // 每 slice：水平角积分 ≈ (1 - aoSlice)，作为这片"开放"立体角占比；
        // radiance 取此片内"raise horizon"的加权平均。
        // 简化（vs 完整 GTAO 解析积分）：用 (1+cosH1)/2 + (1+cosH2)/2 当
        // 该 slice 的"水平角覆盖量"近似（GTAO 论文 Equation 1 的低阶项）。
        float coverage = ((1.0 - cosH1) + (1.0 - cosH2)) * 0.5;   // 0=全无遮挡, 2=全遮挡
        coverage = saturate(1.0 - coverage * 0.5);                 // 反向 → 0=遮蔽, 1=开放

        float wTot = wPos + wNeg;
        float3 sliceRad = (wTot > 1e-4) ? (radPos + radNeg) / wTot : float3(0);
        sumRadiance += sliceRad * coverage;
        coverageSum += coverage;
    }

    float invS = 1.0 / float(gPC.sliceCount);
    float3 avgRad = sumRadiance * invS;
    float avgCov  = coverageSum * invS;
    float4 newVal = float4(avgRad, avgCov);

    // 时序累积（与 SSGI 同：reproject 上一帧 → lerp）
    float3 wPos3 = worldFromDepth(pix, depth);
    float4 prevClip = mul(gFrame.prevViewProj, float4(wPos3, 1.0));
    if (prevClip.w > 0.0) {
        float2 prevNdc = prevClip.xy / prevClip.w;
        if (all(abs(prevNdc) <= 1.0)) {
            float2 prevUv = prevNdc * 0.5 + 0.5;
            float4 prevVal = gPrevGtgi.SampleLevel(gLinearClamp, prevUv, 0);
            newVal = lerp(newVal, prevVal, 0.92);
        }
    }
    gOutGtgi[pix] = newVal;
}
