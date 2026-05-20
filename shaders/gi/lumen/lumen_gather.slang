// L.5 Lumen-lite Final Gather pass.
//
// 每像素从 screen probe atlas bilinear 插值 SH9 → SH cosine convolution
// → diffuse irradiance → 乘 albedo/π → 写 lumenGI。
//
// Dispatch: (screenW + 7) / 8 × (screenH + 7) / 8 × 1

import shared_types;

// SH cosine-convolution coefficients for diffuse irradiance
static const float kA0 = 3.14159265;   // π   (l=0)
static const float kA1 = 2.09439510;   // 2π/3 (l=1)
static const float kA2 = 0.78539816;   // π/4  (l=2)

// SH9 basis constants (same as lumen_probe.slang)
static const float kSH_Y00 = 0.282094792;
static const float kSH_Y1  = 0.488602512;
static const float kSH_Y20 = 0.315391565;
static const float kSH_Y21 = 0.546274215;
static const float kSH_Y22 = 1.092548431;

struct GatherPC {
    float2 invScreenSize;   // 8
    float  probeTileSize;   // 4
    uint   probeGridW;      // 4
    uint   probeGridH;      // 4
    uint   pad;             // 4  = 24 bytes total
};
[[vk::push_constant]] GatherPC gPC;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] Texture2D<float4>              gProbeAtlas;   // SH9 atlas 2D
[[vk::binding(2, 0)]] SamplerState                   gPointClamp;   // nearest
[[vk::binding(3, 0)]] Texture2D<float4>              gAlbedoMetal;
[[vk::binding(4, 0)]] Texture2D<float4>              gNormalRough;
[[vk::binding(5, 0)]] Texture2D<float>               gDepth;
[[vk::binding(6, 0)]] RWTexture2D<float4>            gLumenGI;      // output

// Eval SH9 basis at direction d. Same ordering as lumen_probe.slang.
// Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22
void evalSH9(float3 d, out float sh[9]) {
    float x = d.x, y = d.y, z = d.z;
    sh[0] = kSH_Y00;
    sh[1] = kSH_Y1 * y;
    sh[2] = kSH_Y1 * z;
    sh[3] = kSH_Y1 * x;
    sh[4] = kSH_Y21 * x * y;
    sh[5] = kSH_Y21 * y * z;
    sh[6] = kSH_Y20 * (3.0 * z * z - 1.0);
    sh[7] = kSH_Y21 * x * z;
    sh[8] = kSH_Y22 * (x * x - y * y);
}

// Read SH9 coefficients for probe (px, py) from atlas.
// Atlas: (probeGridW * 9) × probeGridH, RGBA16F, per-coeff texel.
void readProbeSH(int2 probeIdx, out float shR[9], out float shG[9], out float shB[9]) {
    uint baseX = uint(probeIdx.x) * 9u;
    uint baseY = uint(probeIdx.y);
    for (int c = 0; c < 9; ++c) {
        float4 v = gProbeAtlas.Load(int3(int(baseX) + c, int(baseY), 0));
        shR[c] = v.r; shG[c] = v.g; shB[c] = v.b;
    }
}

// Reconstruct diffuse irradiance at normal n from SH9 radiance coefficients.
float3 shIrradiance(float shR[9], float shG[9], float shB[9], float3 n) {
    float basis[9];
    evalSH9(n, basis);

    // Apply cosine convolution: E_lm = A_l * L_lm
    // l=0: A0 = π, l=1: A1 = 2π/3, l=2: A2 = π/4
    float ir = shR[0] * basis[0] * kA0;
    float ig = shG[0] * basis[0] * kA0;
    float ib = shB[0] * basis[0] * kA0;

    for (int c = 1; c <= 3; ++c) {   // l=1
        ir += shR[c] * basis[c] * kA1;
        ig += shG[c] * basis[c] * kA1;
        ib += shB[c] * basis[c] * kA1;
    }
    for (int c = 4; c <= 8; ++c) {   // l=2
        ir += shR[c] * basis[c] * kA2;
        ig += shG[c] * basis[c] * kA2;
        ib += shB[c] * basis[c] * kA2;
    }
    return max(float3(0), float3(ir, ig, ib));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_gather(uint3 dt : SV_DispatchThreadID) {
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) { gLumenGI[pix] = float4(0, 0, 0, 0); return; }

    float4 a = gAlbedoMetal.Load(int3(pix, 0));
    float4 n = gNormalRough.Load(int3(pix, 0));
    float3 albedo = a.rgb;
    float3 N = normalize(n.xyz);

    // Bilinear probe coordinates
    float2 probeCoord = float2(pix) / gPC.probeTileSize - 0.5;
    int2 bp0 = clamp(int2(floor(probeCoord)), 0, int2(gPC.probeGridW - 1, gPC.probeGridH - 1));
    int2 bp1 = clamp(bp0 + 1, 0, int2(gPC.probeGridW - 1, gPC.probeGridH - 1));
    float2 frac = probeCoord - float2(bp0);

    // Bilinear weights
    float w00 = (1.0 - frac.x) * (1.0 - frac.y);
    float w10 = frac.x * (1.0 - frac.y);
    float w01 = (1.0 - frac.x) * frac.y;
    float w11 = frac.x * frac.y;

    // Accumulate interpolated SH coeffs
    float shR[9] = {0}, shG[9] = {0}, shB[9] = {0};
    {
        float r[9], g[9], b[9];

        readProbeSH(int2(bp0.x, bp0.y), r, g, b);
        for (int c = 0; c < 9; ++c) { shR[c] += r[c] * w00; shG[c] += g[c] * w00; shB[c] += b[c] * w00; }

        readProbeSH(int2(bp1.x, bp0.y), r, g, b);
        for (int c = 0; c < 9; ++c) { shR[c] += r[c] * w10; shG[c] += g[c] * w10; shB[c] += b[c] * w10; }

        readProbeSH(int2(bp0.x, bp1.y), r, g, b);
        for (int c = 0; c < 9; ++c) { shR[c] += r[c] * w01; shG[c] += g[c] * w01; shB[c] += b[c] * w01; }

        readProbeSH(int2(bp1.x, bp1.y), r, g, b);
        for (int c = 0; c < 9; ++c) { shR[c] += r[c] * w11; shG[c] += g[c] * w11; shB[c] += b[c] * w11; }
    }

    // Reconstruct irradiance + apply diffuse BRDF
    float3 irradiance = shIrradiance(shR, shG, shB, N);
    float3 diffuse = irradiance * albedo * (1.0 / 3.14159265);

    gLumenGI[pix] = float4(diffuse, 1.0);
}
