// L.2 Lumen-lite Screen Probe pass.
//
// cs_generateRays — TLAS RayQuery → voxelGrid radiance → ray buffer.
// cs_projectSH   — ray buffer → MC SH9 projection → probe atlas.
//
// Bindings:
//   0  FrameUniforms UBO (invViewProj, vxgiGridMin/Cell/Resolution)
//   1  gNormalRough  sampled 2D
//   2  gDepth        sampled 2D
//   3  gTLAS
//   4  gInstances    SSBO
//   5  gVertices     SSBO
//   6  gIndices      SSBO
//   7  gVoxelGrid    sampled 3D
//   8  sampler       linear clamp
//   9  gRayBuf       RW SSBO
//  10  gProbeAtlas   RW storage 2D

import shared_types;

// ---------------------------------------------------------------------------
// Push constants — per-frame probe params (≤ 128 bytes)
// ---------------------------------------------------------------------------
struct ProbePC {
    float2 screenSize;      // 8
    float2 invScreenSize;   // 8
    uint   probeGridW;      // 4
    uint   probeGridH;      // 4
    uint   probeTileSize;   // 4
    uint   raysPerProbe;    // 4
    uint   totalProbes;     // 4
    float  randomSeed;      // 4
    uint   useSixAxis;      // 4  = 44 bytes total
};
[[vk::push_constant]] ProbePC gPC;

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms>          gFrame;
[[vk::binding(1, 0)]] Texture2D<float4>                      gNormalRough;
[[vk::binding(2, 0)]] Texture2D<float>                       gDepth;
[[vk::binding(3, 0)]] RaytracingAccelerationStructure        gTLAS;
[[vk::binding(4, 0)]] Texture3D<float4>                      gVoxelGrid;
[[vk::binding(5, 0)]] SamplerState                           gLinearClamp;
[[vk::binding(6, 0)]] RWStructuredBuffer<float4>             gRayBuf;
[[vk::binding(7, 0)]] RWTexture2D<float4>                    gProbeAtlas;
[[vk::binding(8, 0)]] Texture3D<float4>                      gSixAxisX;
[[vk::binding(9, 0)]] Texture3D<float4>                      gSixAxisY;
[[vk::binding(10,0)]] Texture3D<float4>                      gSixAxisZ;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invScreenSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gFrame.invViewProj, clip);
    return w.xyz / w.w;
}

// Spherical Fibonacci (same as DDGI)
float3 sphericalFibonacci(uint i, uint N) {
    const float PHI = 1.6180339887498949;
    float phi = 6.2831853 * frac(float(i) * (PHI - 1.0));
    float cosTheta = 1.0 - (2.0 * float(i) + 1.0) / float(N);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

float3 rotateY(float3 v, float angle) {
    float c = cos(angle), s = sin(angle);
    return float3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

// ---------------------------------------------------------------------------
// SH9 helpers (real spherical harmonics, l=0..2)
// ---------------------------------------------------------------------------
static const float kSH_Y00 = 0.282094792;   // 1/(2*sqrt(pi))
static const float kSH_Y1  = 0.488602512;   // sqrt(3/(4*pi))
static const float kSH_Y20 = 0.315391565;   // 1/4*sqrt(5/pi)  -- band l=2, m=0
static const float kSH_Y21 = 0.546274215;   // 1/2*sqrt(15/pi) -- band l=2, m!=0
static const float kSH_Y22 = 1.092548431;   // 1/4*sqrt(15/pi) -- band l=2, m=±2

// Evaluate SH9 basis at direction d. 9 coefficients in standard ordering:
//   Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22
//   [0]  [1]   [2]  [3]  [4]    [5]    [6]   [7]   [8]
void evalSH9(float3 d, out float sh[9]) {
    float x = d.x, y = d.y, z = d.z;
    // l=0
    sh[0] = kSH_Y00;
    // l=1
    sh[1] = kSH_Y1 * y;
    sh[2] = kSH_Y1 * z;
    sh[3] = kSH_Y1 * x;
    // l=2
    sh[4] = kSH_Y21 * x * y;
    sh[5] = kSH_Y21 * y * z;
    sh[6] = kSH_Y20 * (3.0 * z * z - 1.0);
    sh[7] = kSH_Y21 * x * z;
    sh[8] = kSH_Y22 * (x * x - y * y);
}

// ---------------------------------------------------------------------------
// cs_generateRays — 每个 thread 负责一条 ray（probe X 内 ray Y）
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void cs_generateRays(uint3 dt : SV_DispatchThreadID) {
    uint rayIdx = dt.x;
    uint totalRays = gPC.totalProbes * gPC.raysPerProbe;
    if (rayIdx >= totalRays) return;

    uint probeIdx = rayIdx / gPC.raysPerProbe;
    uint rayInProbe = rayIdx % gPC.raysPerProbe;

    // Probe 屏幕坐标 → screen center pixels → depth → world pos
    uint px = probeIdx % gPC.probeGridW;
    uint py = probeIdx / gPC.probeGridW;
    float2 probeCenterUV = (float2(px + 0.5, py + 0.5) * float(gPC.probeTileSize)) * gPC.invScreenSize;
    uint2 probePix = uint2(probeCenterUV * float2(gPC.screenSize));
    probePix = min(probePix, gPC.screenSize - 1);

    float depth = gDepth.Load(int3(probePix, 0));
    if (depth >= 1.0) {
        gRayBuf[rayIdx * 2 + 0] = float4(0, 0, 0, 0);
        gRayBuf[rayIdx * 2 + 1] = float4(0, 0, 0, 1e6);
        return;
    }

    float4 nr = gNormalRough.Load(int3(probePix, 0));
    float3 probeN = normalize(nr.xyz);
    float3 probePos = worldFromDepth(probePix, depth);

    // Uniform hemisphere direction via Fibonacci + per-frame rotation.
    // Reflect full-sphere dirs to upper hemisphere so every ray is valid.
    float3 dirZ = sphericalFibonacci(rayInProbe, gPC.raysPerProbe);
    dirZ = rotateY(dirZ, gPC.randomSeed);
    dirZ.z = abs(dirZ.z);   // only upper hemisphere
    dirZ = normalize(dirZ);

    // Tangent frame around probe normal
    float3 up = abs(probeN.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 T = normalize(cross(up, probeN));
    float3 B = cross(probeN, T);
    float3 rayDir = T * dirZ.x + B * dirZ.y + probeN * dirZ.z;

    float3 hitRgb = float3(0);
    float  hitDist = 1e6;

    if (gPC.useSixAxis >= 2) {
        // Debug bypass: skip TLAS, mark all rays as hit with const radiance
        hitRgb = float3(1.0, 1.0, 1.0);
        hitDist = 0.5;
    } else {
        // TLAS ray query
        RayDesc ray;
        ray.Origin = probePos + probeN * 0.05;
        ray.Direction = rayDir;
        ray.TMin = 0.01;
        ray.TMax = 200.0;

        RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(gTLAS, RAY_FLAG_NONE, 0xFF, ray);
        q.Proceed();

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            float  hitT  = q.CommittedRayT();
            float3 hitPos = ray.Origin + rayDir * hitT;

            // Sample voxel grid radiance at hit point
            float3 gridMin = gFrame.vxgiGridMinCell.xyz;
            float  cellSize = gFrame.vxgiGridMinCell.w;
            uint   gridRes = uint(gFrame.vxgiCounts.x);
            float3 voxelUv = (hitPos - gridMin) / (cellSize * float(gridRes));

            if (gPC.useSixAxis != 0) {
            // 6-axis directional sampling: blend +X/+Y/+Z axis by ray alignment
            float3 ad = abs(rayDir);
            float4 ax = gSixAxisX.SampleLevel(gLinearClamp, voxelUv, 2.0);
            float4 ay = gSixAxisY.SampleLevel(gLinearClamp, voxelUv, 2.0);
            float4 az = gSixAxisZ.SampleLevel(gLinearClamp, voxelUv, 2.0);
            float3 posRgb = ax.rgb * ad.x + ay.rgb * ad.y + az.rgb * ad.z;
            float  negLum = ax.a * ad.x + ay.a * ad.y + az.a * ad.z;
            float  totalW = ad.x + ad.y + ad.z;
            // Negative hemisphere: use luminance only, approximate grey
            float3 negRgb = float3(negLum, negLum, negLum);
            float3 blended = (posRgb + negRgb) / max(totalW, 0.001);
            hitRgb = blended;
        } else {
            hitRgb = gVoxelGrid.SampleLevel(gLinearClamp, voxelUv, 2.0).rgb;
        }
            hitDist = hitT;
        }
    }

    gRayBuf[rayIdx * 2 + 0] = float4(rayDir, 0);
    gRayBuf[rayIdx * 2 + 1] = float4(hitRgb, hitDist);
}

// ---------------------------------------------------------------------------
// cs_projectSH — 每个 thread 处理一个 probe，读其所有 rays 做 SH9 投影
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void cs_projectSH(uint3 dt : SV_DispatchThreadID) {
    uint probeIdx = dt.x;
    if (probeIdx >= gPC.totalProbes) return;

    uint px = probeIdx % gPC.probeGridW;
    uint py = probeIdx / gPC.probeGridW;

    // Probe world normal (for cosine weighting)
    float2 probeCenterUV = (float2(px + 0.5, py + 0.5) * float(gPC.probeTileSize)) * gPC.invScreenSize;
    uint2 probePix = uint2(probeCenterUV * float2(gPC.screenSize));
    probePix = min(probePix, gPC.screenSize - 1);

    float4 nr = gNormalRough.Load(int3(probePix, 0));
    float3 probeN = normalize(nr.xyz);

    // Accumulate SH9: 9 coefs × 3 channels
    float shR[9] = {0,0,0,0,0,0,0,0,0};
    float shG[9] = {0,0,0,0,0,0,0,0,0};
    float shB[9] = {0,0,0,0,0,0,0,0,0};
    uint hitCount = 0;
    uint baseSlot = probeIdx * gPC.raysPerProbe;

    for (uint r = 0; r < gPC.raysPerProbe; ++r) {
        uint slot = baseSlot + r;
        float4 d4 = gRayBuf[slot * 2 + 0];
        float4 v4 = gRayBuf[slot * 2 + 1];
        float3 dir = d4.xyz;
        float3 rad = v4.xyz;
        float dist = v4.w;

        if (dist >= 1e5) continue;  // miss

        hitCount++;
        float shBasis[9];
        evalSH9(dir, shBasis);

        // MC estimate of SH radiance: L_lm = (2π/N_hit)·Σ L·Y_lm
        // Normalize by HIT count (not total rays) so result is independent
        // of the hit ratio, which varies per-frame for edge probes.
        for (int c = 0; c < 9; ++c) {
            shR[c] += rad.r * shBasis[c];
            shG[c] += rad.g * shBasis[c];
            shB[c] += rad.b * shBasis[c];
        }
    }

    // L_lm = (2π / N_hit) * accumulated sum
    float mcNorm = (hitCount > 0) ? (6.2831853 / float(hitCount)) : 0.0;

    // Write SH coefs to probe atlas: (atlasX + coeffIdx, atlasY)
    uint baseX = px * 9;
    uint baseY = py;

    if (gPC.useSixAxis == 3) {
        // Bypass test: fixed SH = DC only, all probes identical
        for (int c = 0; c < 9; ++c) {
            float val = (c == 0) ? (2.0 * 3.14159265 * 0.282094792) : 0.0; // 2π*Y00 ≈ 1.77
            gProbeAtlas[uint2(baseX + c, baseY)] = float4(val, val, val, 1.0);
        }
    } else {
        for (int c = 0; c < 9; ++c) {
            float3 coeff = float3(shR[c], shG[c], shB[c]) * mcNorm;
            gProbeAtlas[uint2(baseX + c, baseY)] = float4(coeff, 1.0);
        }
    }
}
