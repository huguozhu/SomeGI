// Nanite Phase 4 — compute SW rasterizer for small triangles.
// Each thread rasters one triangle, writing triangleID to visibility buffer.

import shared_types;

// Vertex layout (matches nanite_types.h PackedVertex: 16 bytes)
struct PackedVertex {
    float px, py, pz;   // 12 bytes
    uint  normal_uv;    //  4 bytes (oct normal + half UV)
};

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] StructuredBuffer<PackedVertex> gVertices;
[[vk::binding(2, 0)]] RWTexture2D<uint> gVisBuffer;  // triangleID

struct RasterPC {
    uint  triangleCount;
    uint2 outSize;
    float2 invOutSize;
    uint  pad;
};
[[vk::push_constant]] RasterPC gPC;

float edge(float2 a, float2 b, float2 p) {
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint triIdx = dt.x;
    if (triIdx >= gPC.triangleCount) return;

    uint vBase = triIdx * 3u;
    PackedVertex v0 = gVertices[vBase + 0u];
    PackedVertex v1 = gVertices[vBase + 1u];
    PackedVertex v2 = gVertices[vBase + 2u];

    float3 p0 = float3(v0.px, v0.py, v0.pz);
    float3 p1 = float3(v1.px, v1.py, v1.pz);
    float3 p2 = float3(v2.px, v2.py, v2.pz);

    // Transform to clip space
    float4 c0 = mul(gFrame.viewProj, float4(p0, 1.0));
    float4 c1 = mul(gFrame.viewProj, float4(p1, 1.0));
    float4 c2 = mul(gFrame.viewProj, float4(p2, 1.0));

    // Frustum cull: all vertices outside same plane → skip
    if (c0.w <= 0.0 && c1.w <= 0.0 && c2.w <= 0.0) return;
    if (all(c0.xy < -c0.w) && all(c1.xy < -c1.w) && all(c2.xy < -c2.w)) return;
    if (all(c0.xy >  c0.w) && all(c1.xy >  c1.w) && all(c2.xy >  c2.w)) return;

    // Perspective divide → NDC → screen UV
    float2 s0 = (c0.xy / max(c0.w, 1e-5)) * 0.5 + 0.5;
    float2 s1 = (c1.xy / max(c1.w, 1e-5)) * 0.5 + 0.5;
    float2 s2 = (c2.xy / max(c2.w, 1e-5)) * 0.5 + 0.5;

    float2 dim = float2(gPC.outSize);
    float2 p0p = s0 * dim;
    float2 p1p = s1 * dim;
    float2 p2p = s2 * dim;

    // Bounding box
    int2 bbMin = int2(floor(min(min(p0p, p1p), p2p)));
    int2 bbMax = int2(ceil (max(max(p0p, p1p), p2p)));
    bbMin = max(bbMin, int2(0, 0));
    bbMax = min(bbMax, int2(gPC.outSize) - 1);

    // Backface cull
    float area = edge(p0p, p1p, p2p);
    if (area <= 0.0) return;

    // Rasterize
    for (int y = bbMin.y; y <= bbMax.y; ++y) {
        for (int x = bbMin.x; x <= bbMax.x; ++x) {
            float2 px = float2(float(x) + 0.5, float(y) + 0.5);

            float e0 = edge(p1p, p2p, px);
            float e1 = edge(p2p, p0p, px);
            float e2 = edge(p0p, p1p, px);

            if (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) {
                gVisBuffer[uint2(x, y)] = triIdx;
            }
        }
    }
}
