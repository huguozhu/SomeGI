// Visibility-buffer resolve pass — reconstructs GBuffer from triangleID + depth.

import shared_types;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] Texture2D<uint2> gVisBuffer;
[[vk::binding(2, 0)]] StructuredBuffer<float> gVertices;
[[vk::binding(3, 0)]] StructuredBuffer<uint>  gIndices;
[[vk::binding(4, 0)]] StructuredBuffer<MaterialGpu> gMaterials;
[[vk::binding(5, 0)]] SamplerState gLinear;
[[vk::binding(6, 0)]] Texture2D gTextures[];
[[vk::binding(7, 0)]] RWTexture2D<float4> gOutAlbedoMetal;
[[vk::binding(8, 0)]] RWTexture2D<float4> gOutNormalRough;
[[vk::binding(9, 0)]] RWTexture2D<float4> gOutEmissiveAO;

// Per-vertex layout (matches scene.h Vertex): 12 floats = 48 bytes
static const uint kVStride     = 12;  // floats per vertex
static const uint kPosOff      = 0;   // position (3 floats)
static const uint kNormalOff   = 3;   // normal (3)
static const uint kTangentOff  = 6;   // tangent (4)
static const uint kUvOff       = 10;  // uv0 (2)

struct PC {
    uint2  outSize;
    float2 invOutSize;
};
[[vk::push_constant]] PC gPC;

float3 readVertexPos(uint vertIdx) {
    uint base = vertIdx * kVStride;
    return float3(gVertices[base + 0], gVertices[base + 1], gVertices[base + 2]);
}

float3 readVertexNormal(uint vertIdx) {
    uint base = vertIdx * kVStride;
    return float3(gVertices[base + 3], gVertices[base + 4], gVertices[base + 5]);
}

float4 readVertexTangent(uint vertIdx) {
    uint base = vertIdx * kVStride;
    return float4(gVertices[base + 6], gVertices[base + 7], gVertices[base + 8], gVertices[base + 9]);
}

float2 readVertexUv(uint vertIdx) {
    uint base = vertIdx * kVStride;
    return float2(gVertices[base + 10], gVertices[base + 11]);
}

float4 sampleTex(int idx, float2 uv) {
    if (idx < 0) return float4(1, 1, 1, 1);
    return gTextures[idx].SampleLevel(gLinear, uv, 0);
}

// Barycentric coordinates of point p in triangle (a, b, c)
float3 barycentrics(float3 p, float3 a, float3 b, float3 c) {
    float3 v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    if (abs(denom) < 1e-10) return float3(1.0/3.0, 1.0/3.0, 1.0/3.0); // degenerate
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0 - v - w;
    return float3(u, v, w);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (any(dt.xy >= gPC.outSize)) return;

    uint2 tid = gVisBuffer[dt.xy];
    uint triId = tid.x;
    float depth = asfloat(tid.y);

    // Skip sky pixels (depth == 1.0 or no triangle)
    if (depth >= 1.0 || triId == 0u) {
        gOutAlbedoMetal[dt.xy] = float4(0, 0, 0, 0);
        gOutNormalRough[dt.xy] = float4(0, 0, 1, 0);
        gOutEmissiveAO[dt.xy] = float4(0, 0, 0, 0);
        return;
    }

    uint matIdx  = triId >> 20u;
    uint primIdx = triId & 0xFFFFFu;

    // Read 3 vertex indices for this triangle
    uint i0 = gIndices[primIdx * 3u + 0u];
    uint i1 = gIndices[primIdx * 3u + 1u];
    uint i2 = gIndices[primIdx * 3u + 2u];

    // Read vertex positions
    float3 p0 = readVertexPos(i0);
    float3 p1 = readVertexPos(i1);
    float3 p2 = readVertexPos(i2);

    // Reconstruct world position from depth
    float2 uv = (float2(dt.xy) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc.x, -ndc.y, depth, 1.0);
    float4 wp4 = mul(gFrame.invViewProj, clip);
    float3 worldPos = wp4.xyz / wp4.w;

    // Compute barycentric coordinates in world space
    float3 bc = barycentrics(worldPos, p0, p1, p2);

    // Interpolate vertex attributes
    float3 N = readVertexNormal(i0) * bc.x + readVertexNormal(i1) * bc.y + readVertexNormal(i2) * bc.z;
    N = normalize(N);
    float4 T = readVertexTangent(i0) * bc.x + readVertexTangent(i1) * bc.y + readVertexTangent(i2) * bc.z;
    float2 uv0 = readVertexUv(i0) * bc.x + readVertexUv(i1) * bc.y + readVertexUv(i2) * bc.z;

    // Sample materials (same logic as gbuffer.slang)
    MaterialGpu m = gMaterials[matIdx];
    float4 baseTex = sampleTex(m.baseColorTex, uv0);
    float4 base = m.baseColorFactor * baseTex;

    float metallic = m.metallicFactor;
    float roughness = m.roughnessFactor;
    if (m.mrTex >= 0) {
        float3 t = gTextures[m.mrTex].SampleLevel(gLinear, uv0, 0).rgb;
        roughness *= t.g;
        metallic  *= t.b;
    }
    metallic  = saturate(metallic);
    roughness = saturate(roughness);

    if (m.normalTex >= 0) {
        float3 nm = gTextures[m.normalTex].SampleLevel(gLinear, uv0, 0).rgb * 2.0 - 1.0;
        nm.xy *= m.normalScale;
        float3 T3 = normalize(T.xyz);
        float3 B = normalize(cross(N, T3) * T.w);
        N = normalize(T3 * nm.x + B * nm.y + N * nm.z);
    }

    float ao = 1.0;
    if (m.occlusionTex >= 0) {
        float t = gTextures[m.occlusionTex].SampleLevel(gLinear, uv0, 0).r;
        ao = lerp(1.0, t, m.occlusionStrength);
    }

    float3 emi = m.emissiveFactor;
    if (m.emissiveTex >= 0) emi *= gTextures[m.emissiveTex].SampleLevel(gLinear, uv0, 0).rgb;

    // Write reconstructed GBuffer
    gOutAlbedoMetal[dt.xy] = float4(base.rgb, metallic);
    gOutNormalRough[dt.xy] = float4(N, roughness);
    gOutEmissiveAO[dt.xy] = float4(saturate(emi), ao);
}
