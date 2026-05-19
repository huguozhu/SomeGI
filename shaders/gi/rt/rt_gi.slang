// M9 — 硬件光线追踪全局光照（Ray Query 风格）。
//
// 每像素从 GBuffer 重建世界坐标/法线 → 在法线 cosine 半球内发 1 条 ray
// → RayQuery 命中后计算该点的直接太阳光（含 shadow ray）。
//
// 输出：
//   rgb = 入射 radiance（命中点处 albedo · sunBRDF，未上 BRDF）
//   a   = 1.0（命中）/ 0.0（未命中，IBL fallback）
//
// Lighting.slang 端据此 lerp IBL diffuse → RT diffuse。
// 命中点时完整替换 IBL；未命中时回落到 IBL。
//
// 用 KHR_ray_query 而非完整 RT pipeline（无 SBT 模板），一切在 compute
// shader 内完成。

import shared_types;

struct RtPC {
    uint2  outSize;
    float2 invOutSize;
};
[[vk::push_constant]] RtPC gPC;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms>          gFrame;
[[vk::binding(1, 0)]] Texture2D<float4>                      gNormalRough;
[[vk::binding(2, 0)]] Texture2D<float>                       gDepth;
[[vk::binding(3, 0)]] RaytracingAccelerationStructure        gTLAS;
[[vk::binding(4, 0)]] StructuredBuffer<RtInstanceData>        gInstances;
[[vk::binding(5, 0)]] StructuredBuffer<Vertex>               gVertices;
[[vk::binding(6, 0)]] StructuredBuffer<uint>                 gIndices;
[[vk::binding(7, 0)]] StructuredBuffer<MaterialGpu>           gMaterials;
[[vk::binding(8, 0)]] SamplerState                            gLinearClamp;
[[vk::binding(9, 0)]] Texture2D<float4>                       gTextures[];
[[vk::binding(10, 0)]] RWTexture2D<float4>                    gOutRt;

struct RtInstanceData {
    int materialIndex;
    int vertexOffset;
    uint firstIndex;
    int _pad0;
};

struct Vertex {
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uv0;
};

float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gFrame.invViewProj, clip);
    return w.xyz / w.w;
}

// Fetch material baseColor at UV, handling the optional texture (-1 = use factor).
float3 sampleBaseColor(int texIdx, MaterialGpu mat, float2 uv) {
    if (texIdx >= 0 && texIdx < 128) {
        return gTextures[texIdx].SampleLevel(gLinearClamp, uv, 0).rgb * mat.baseColorFactor.rgb;
    }
    return mat.baseColorFactor.rgb;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) return;
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) { gOutRt[pix] = float4(0, 0, 0, 0); return; }

    float4 nr = gNormalRough.Load(int3(pix, 0));
    float3 wN = normalize(nr.xyz);
    float3 wPos = worldFromDepth(pix, depth);

    // Cosine-weighted hemisphere direction in tangent space (same as SSGI).
    float3 up = abs(wN.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 T = normalize(cross(up, wN));
    float3 B = cross(wN, T);

    // Single cosine-hemisphere sample. Use a deterministic per-pixel direction
    // (Hammersley i=0 + pixel hash offset) so neighbors diverge.
    // For simplicity: pick a fixed direction in the hemisphere.
    float2 h = float2(0.25, 0.5);   // deterministic, d = (0.5, 0.5, 0.707) in tangent space
    float r = sqrt(h.x);
    float phi = 6.2831853 * h.y;
    float3 dirT = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - h.x)));
    float3 dirW = T * dirT.x + B * dirT.y + wN * dirT.z;

    // Shoot the ray from a small offset along N to avoid self-intersection.
    float3 origin = wPos + wN * 0.01;
    float tMax = 100.0;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dirW;
    ray.TMin = 0.01;
    ray.TMax = tMax;

    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(gTLAS, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    float3 rtRgb = float3(0);
    float rtHit = 0.0;

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        // Geometry data from the hit.
        uint  instIdx    = q.CommittedInstanceID();
        uint  primIdx    = q.CommittedPrimitiveIndex();
        float2 bary      = q.CommittedTriangleBarycentrics();
        float  hitT      = q.CommittedRayT();

        RtInstanceData inst = gInstances[instIdx];

        // Fetch triangle vertices.
        uint idx0 = gIndices[inst.firstIndex + primIdx * 3 + 0];
        uint idx1 = gIndices[inst.firstIndex + primIdx * 3 + 1];
        uint idx2 = gIndices[inst.firstIndex + primIdx * 3 + 2];

        Vertex v0 = gVertices[idx0];
        Vertex v1 = gVertices[idx1];
        Vertex v2 = gVertices[idx2];

        // Barycentric interpolation.
        float u = bary.x;
        float v = bary.y;
        float w = 1.0 - u - v;

        float3 hitPos   = v0.position * w + v1.position * u + v2.position * v;
        float3 hitN     = normalize(v0.normal * w + v1.normal * u + v2.normal * v);
        float2 hitUv    = v0.uv0 * w + v1.uv0 * u + v2.uv0 * v;

        // Sample material.
        MaterialGpu mat = gMaterials[inst.materialIndex];
        float3 baseColor = sampleBaseColor(mat.baseColorTex, mat, hitUv);

        // Sun direct light at hit point (shadow ray).
        float3 L = normalize(-gFrame.sunDir.xyz);
        float NdotL = dot(hitN, L);

        float3 sunContrib = float3(0);
        if (NdotL > 0.0) {
            // Shadow ray from hit point to sun.
            RayDesc shadowRay;
            shadowRay.Origin = hitPos + hitN * 0.005;
            shadowRay.Direction = L;
            shadowRay.TMin = 0.005;
            shadowRay.TMax = tMax;

            RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
                     RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
            sq.TraceRayInline(gTLAS, RAY_FLAG_NONE, 0xFF, shadowRay);
            sq.Proceed();

            bool inShadow = (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
            if (!inShadow) {
                float3 sunColor = gFrame.sunColor_intensity.rgb;
                float  sunInt   = gFrame.sunColor_intensity.w;
                sunContrib = baseColor * (1.0 / 3.14159265) * NdotL * sunColor * sunInt;
            }
        }

        // Result: incoming radiance at the hit point = sun direct (one bounce).
        // No diffuse BRDF applied here — lighting.slang does applyIBLDiffuseBRDF.
        // But for a diffuse bounce, the BRDF at the receiver is albedo/π, and
        // the BRDF at the hit point is also applied. For simplicity, output
        // the 'flux' equivalent: baseColor * NdotL * sun (matching RSM flux).
        rtRgb = sunContrib;
        rtHit = 1.0;
    }

    // When no hit (rtHit = 0), lighting.slang falls back to IBL diffuse.
    gOutRt[pix] = float4(rtRgb, rtHit);
}
