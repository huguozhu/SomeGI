// DDGI atlas blend —— 把 ray buffer 里 64 rays/probe 的命中数据 blend 到
// 两张 octahedral atlas（irradiance + distance）。
//
// 一张 shader 配 entry point cs_irradiance（每 texel 8×8 atlas，cosine
// kernel）和 cs_distance（每 texel 16×16 atlas，sharper Chebyshev kernel）。
// 共享 ray buffer 但写到不同 RWTexture2D 输出。
//
// dispatch：
//   cs_irradiance: (atlasW/8, atlasH/8) groups of 8×8 thread；每 thread = 1 atlas texel
//   cs_distance:   同上，但 atlas 维度更大
// 每 texel 内部循环 raysPerProbe，按方向 alignment 加权累积。
// 结果与上帧 atlas 做 hysteresis blend (alpha=0.05) 平滑变化。

import octahedral;

[[vk::binding(0, 0)]] StructuredBuffer<float4> gRayBuf;     // ray data
[[vk::binding(1, 0)]] RWTexture2D<float4>      gIrrAtlas;   // RGBA16F
[[vk::binding(2, 0)]] RWTexture2D<float2>      gDistAtlas;  // RG16F (mean, mean²)

struct BlendPC {
    uint probesX, probesY, probesZ;
    uint raysPerProbe;
    uint octaIrr;
    uint octaDist;
    float hysteresis;     // 时序 blend 因子；新数据贡献 = 1 - hysteresis
    float maxRayDistance;
};
[[vk::push_constant]] BlendPC gPC;

// atlas texel (x,y) → (probeIdx, octaUv) 反查
//   atlasW = probesX * octa
//   atlasH = probesY * probesZ * octa
//   probeXIdx = x / octa；row = y / octa；probeY = row % probesY；probeZ = row / probesY
void atlasTexelToProbe(uint2 pix, uint octa,
                       out uint probeIdx, out float2 octaUv) {
    uint2 inProbe = pix % octa;
    uint2 probeXY = pix / octa;
    uint probeXIdx = probeXY.x;
    uint row = probeXY.y;
    uint probeYIdx = row % gPC.probesY;
    uint probeZIdx = row / gPC.probesY;
    probeIdx = (probeZIdx * gPC.probesY + probeYIdx) * gPC.probesX + probeXIdx;
    // texel 中心采样
    octaUv = (float2(inProbe) + 0.5) / float(octa);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_irradiance(uint3 dt : SV_DispatchThreadID) {
    uint atlasW = gPC.probesX * gPC.octaIrr;
    uint atlasH = gPC.probesY * gPC.probesZ * gPC.octaIrr;
    if (dt.x >= atlasW || dt.y >= atlasH) return;

    uint probeIdx;
    float2 octaUv;
    atlasTexelToProbe(dt.xy, gPC.octaIrr, probeIdx, octaUv);
    float3 texelDir = octUvToDir(octaUv);

    float3 acc = float3(0);
    float weightSum = 0.0;
    uint base = probeIdx * gPC.raysPerProbe * 2;
    for (uint i = 0; i < gPC.raysPerProbe; ++i) {
        float3 rayDir = gRayBuf[base + i*2 + 0].xyz;
        float4 hit = gRayBuf[base + i*2 + 1];
        // 命中 alignment：>0 才贡献。指数 cosine 加权（论文用 max(0, dot)^k）。
        float w = max(0.0, dot(texelDir, rayDir));
        if (w > 1e-3) {
            acc += hit.rgb * w;
            weightSum += w;
        }
    }
    float3 newIrr = (weightSum > 0.0) ? acc / weightSum : float3(0);

    // hysteresis blend with prev atlas
    float3 prev = gIrrAtlas[dt.xy].rgb;
    float3 blended = lerp(newIrr, prev, gPC.hysteresis);
    gIrrAtlas[dt.xy] = float4(blended, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_distance(uint3 dt : SV_DispatchThreadID) {
    uint atlasW = gPC.probesX * gPC.octaDist;
    uint atlasH = gPC.probesY * gPC.probesZ * gPC.octaDist;
    if (dt.x >= atlasW || dt.y >= atlasH) return;

    uint probeIdx;
    float2 octaUv;
    atlasTexelToProbe(dt.xy, gPC.octaDist, probeIdx, octaUv);
    float3 texelDir = octUvToDir(octaUv);

    float meanSum = 0.0;
    float meanSqSum = 0.0;
    float weightSum = 0.0;
    uint base = probeIdx * gPC.raysPerProbe * 2;
    for (uint i = 0; i < gPC.raysPerProbe; ++i) {
        float3 rayDir = gRayBuf[base + i*2 + 0].xyz;
        float dist = gRayBuf[base + i*2 + 1].w;
        // distance 用更尖的 cosine power（论文里 power 50）—— 让 Chebyshev
        // VSM 更精确（避免漏光）。
        float w = pow(max(0.0, dot(texelDir, rayDir)), 50.0);
        if (w > 1e-4) {
            float clamped = min(dist, gPC.maxRayDistance);
            meanSum += clamped * w;
            meanSqSum += clamped * clamped * w;
            weightSum += w;
        }
    }
    float2 newDist = float2(0);
    if (weightSum > 0.0) {
        newDist.x = meanSum / weightSum;
        newDist.y = meanSqSum / weightSum;
    }
    float2 prev = gDistAtlas[dt.xy];
    float2 blended = lerp(newDist, prev, gPC.hysteresis);
    gDistAtlas[dt.xy] = blended;
}
