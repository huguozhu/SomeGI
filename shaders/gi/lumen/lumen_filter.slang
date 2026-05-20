// L.4 Lumen-lite Probe Filtering — spatial cross-bilateral on SH9.
//
// 对每个 probe 的 SH9 coefficient 做 3×3 spatial bilateral filter：
//   weight = depth_similarity × normal_similarity × distance_falloff
// 目的：消除邻 probe 间亮度跳变（discontinuity），不跨几何边沿模糊。
//
// Dispatch: (probeCount + 63) / 64 groups × 64 threads

import shared_types;

struct FilterPC {
    float2 invScreenSize;
    float  probeTileSize;
    uint   probeGridW;
    uint   probeGridH;
    float  sigmaDepth;      // depth diff tolerance, ~0.1
    float  normalPower;     // sharpness of normal cutoff, ~32
    float  sigmaDist;       // world-space distance falloff, ~100
    uint   pad;
};
[[vk::push_constant]] FilterPC gPC;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] Texture2D<float4>              gNormalRough;
[[vk::binding(2, 0)]] Texture2D<float>               gDepth;
[[vk::binding(3, 0)]] Texture2D<float4>              gProbeAtlas;     // input
[[vk::binding(4, 0)]] RWTexture2D<float4>            gFilteredAtlas;  // output

// Reconstruct world position from depth
float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invScreenSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gFrame.invViewProj, clip);
    return w.xyz / w.w;
}

// Read probe center world position and normal
void probeWorldPosNormal(int2 gridPos, out float3 wPos, out float3 wN) {
    float2 uv = (float2(gridPos) + 0.5) * gPC.probeTileSize * gPC.invScreenSize;
    uint2 pix = uint2(uv * (1.0 / gPC.invScreenSize));
    pix = min(pix, uint2(1.0 / gPC.invScreenSize) - 1);

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) {
        wPos = float3(0); wN = float3(0, 1, 0);
    } else {
        wPos = worldFromDepth(pix, depth);
        wN = normalize(gNormalRough.Load(int3(pix, 0)).xyz);
    }
}

// Read SH9 coefficients for a probe from atlas
void readProbeSH(int2 gridPos, out float shR[9], out float shG[9], out float shB[9]) {
    uint baseX = uint(gridPos.x) * 9u;
    uint baseY = uint(gridPos.y);
    for (int c = 0; c < 9; ++c) {
        float4 v = gProbeAtlas.Load(int3(int(baseX) + c, int(baseY), 0));
        shR[c] = v.r; shG[c] = v.g; shB[c] = v.b;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_spatialFilter(uint3 dt : SV_DispatchThreadID) {
    uint probeIdx = dt.x;
    uint total = gPC.probeGridW * gPC.probeGridH;
    if (probeIdx >= total) return;

    int2 center = int2(int(probeIdx % gPC.probeGridW),
                       int(probeIdx / gPC.probeGridW));

    // Center probe world pos / normal
    float3 centerPos, centerN;
    probeWorldPosNormal(center, centerPos, centerN);
    float centerDepth = dot(centerN, centerPos);  // proxy: depth along normal

    // Accumulate
    float shR[9] = {0}, shG[9] = {0}, shB[9] = {0};
    float totalW = 0.0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 neighbor = center + int2(dx, dy);
            if (neighbor.x < 0 || neighbor.y < 0 ||
                neighbor.x >= int(gPC.probeGridW) || neighbor.y >= int(gPC.probeGridH))
                continue;

            float3 nPos, nN;
            probeWorldPosNormal(neighbor, nPos, nN);
            float nDepth = dot(nN, nPos);

            // depth similarity
            float dDiff = abs(nDepth - centerDepth);
            float wDepth = exp(-dDiff / gPC.sigmaDepth);

            // normal similarity
            float wNormal = pow(max(0.0, dot(centerN, nN)), gPC.normalPower);

            // world-space distance
            float wDist = 1.0 / (1.0 + length(nPos - centerPos) * gPC.sigmaDist);

            float w = wDepth * wNormal * wDist;

            float nR[9], nG[9], nB[9];
            readProbeSH(neighbor, nR, nG, nB);
            for (int c = 0; c < 9; ++c) {
                shR[c] += nR[c] * w;
                shG[c] += nG[c] * w;
                shB[c] += nB[c] * w;
            }
            totalW += w;
        }
    }

    // Normalize → write filtered atlas
    float invW = totalW > 0.0 ? 1.0 / totalW : 0.0;
    uint baseX = uint(center.x) * 9u;
    uint baseY = uint(center.y);
    for (int c = 0; c < 9; ++c) {
        float3 coeff = float3(shR[c], shG[c], shB[c]) * invW;
        gFilteredAtlas[uint2(baseX + c, baseY)] = float4(coeff, 1.0);
    }
}
