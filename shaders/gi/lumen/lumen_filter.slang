// L.4 Lumen-lite Probe Filtering — spatial + temporal on SH9.
//
// cs_spatialFilter:
//   5×5 spatial bilateral + temporal reprojection from prevAtlas.
//   For each probe: worldPos → reproject to prev NDC → sample prevAtlas
//   → alpha-blend with spatial result → write filteredAtlas.
//
// Dispatch: (probeCount + 63) / 64 groups × 64 threads

import shared_types;

struct FilterPC {
    float2 invScreenSize;
    float  probeTileSize;
    uint   probeGridW;
    uint   probeGridH;
    float  sigmaDepth;
    float  normalPower;
    float  sigmaDist;
    float  temporalAlpha;   // 0=all new, 1=all prev
};
[[vk::push_constant]] FilterPC gPC;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] Texture2D<float4>              gNormalRough;
[[vk::binding(2, 0)]] Texture2D<float>               gDepth;
[[vk::binding(3, 0)]] Texture2D<float4>              gProbeAtlas;     // raw input
[[vk::binding(4, 0)]] RWTexture2D<float4>            gFilteredAtlas;  // output
[[vk::binding(5, 0)]] Texture2D<float4>              gPrevAtlas;      // prev frame
[[vk::binding(6, 0)]] SamplerState                   gPointClamp;

float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invScreenSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gFrame.invViewProj, clip);
    return w.xyz / w.w;
}

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

void readProbeSH(Texture2D<float4> atlas, int2 gridPos,
                 out float shR[9], out float shG[9], out float shB[9]) {
    uint baseX = uint(gridPos.x) * 9u;
    uint baseY = uint(gridPos.y);
    for (int c = 0; c < 9; ++c) {
        float4 v = atlas.Load(int3(int(baseX) + c, int(baseY), 0));
        shR[c] = v.r; shG[c] = v.g; shB[c] = v.b;
    }
}

void writeProbeSH(RWTexture2D<float4> atlas, int2 gridPos,
                  float shR[9], float shG[9], float shB[9]) {
    uint baseX = uint(gridPos.x) * 9u;
    uint baseY = uint(gridPos.y);
    for (int c = 0; c < 9; ++c) {
        atlas[uint2(baseX + c, baseY)] = float4(shR[c], shG[c], shB[c], 1.0);
    }
}

// Sample prev atlas at a reprojected position (bilinear-like via Load)
void samplePrevAtlas(float3 worldPos, out float shR[9], out float shG[9], out float shB[9]) {
    // Project worldPos to previous frame's NDC
    float4 clip = mul(gFrame.prevViewProj, float4(worldPos, 1.0));
    if (abs(clip.w) < 1e-6) { shR[0] = -1; shG[0] = 0; shB[0] = 0; return; } // 无效标记

    float3 ndc = clip.xyz / clip.w;
    if (any(abs(ndc) > 1.001)) { shR[0] = -1; shG[0] = 0; shB[0] = 0; return; }

    // NDC → screen UV → probe grid coordinate
    float2 screenUV = ndc.xy * 0.5 + 0.5;
    float2 probeCoord = screenUV * (1.0 / gPC.invScreenSize) / gPC.probeTileSize - 0.5;
    int2 bp = clamp(int2(floor(probeCoord)), 0, int2(gPC.probeGridW - 1, gPC.probeGridH - 1));

    readProbeSH(gPrevAtlas, bp, shR, shG, shB);
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
    float centerDepth = dot(centerN, centerPos);

    // 1. Spatial accumulate (5×5)
    float shR[9] = {0}, shG[9] = {0}, shB[9] = {0};
    float totalW = 0.0;

    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int2 neighbor = center + int2(dx, dy);
            if (neighbor.x < 0 || neighbor.y < 0 ||
                neighbor.x >= int(gPC.probeGridW) || neighbor.y >= int(gPC.probeGridH))
                continue;

            float3 nPos, nN;
            probeWorldPosNormal(neighbor, nPos, nN);
            float nDepth = dot(nN, nPos);

            float dDiff = abs(nDepth - centerDepth);
            float wDepth = exp(-dDiff / max(gPC.sigmaDepth, 0.001));
            float wNormal = pow(max(0.0, dot(centerN, nN)), gPC.normalPower);
            float wDist = 1.0 / (1.0 + length(nPos - centerPos) * gPC.sigmaDist);
            float w = wDepth * wNormal * wDist;

            float nR[9], nG[9], nB[9];
            readProbeSH(gProbeAtlas, neighbor, nR, nG, nB);
            for (int c = 0; c < 9; ++c) {
                shR[c] += nR[c] * w;
                shG[c] += nG[c] * w;
                shB[c] += nB[c] * w;
            }
            totalW += w;
        }
    }

    if (totalW > 0.0) {
        float invW = 1.0 / totalW;
        for (int c = 0; c < 9; ++c) {
            shR[c] *= invW; shG[c] *= invW; shB[c] *= invW;
        }
    }

    // 2. Temporal blend: reproject worldPos to prev frame → sample prevAtlas
    if (gPC.temporalAlpha > 0.001) {
        float prevR[9], prevG[9], prevB[9];
        samplePrevAtlas(centerPos, prevR, prevG, prevB);

        if (prevR[0] > -0.5) {  // valid reprojection
            float a = gPC.temporalAlpha;
            float b = 1.0 - a;
            for (int c = 0; c < 9; ++c) {
                shR[c] = shR[c] * b + prevR[c] * a;
                shG[c] = shG[c] * b + prevG[c] * a;
                shB[c] = shB[c] * b + prevB[c] * a;
            }
        }
    }

    writeProbeSH(gFilteredAtlas, center, shR, shG, shB);
}
