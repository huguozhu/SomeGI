// ReSTIR DI shade pass —— M10 硬件 RT 可见性变体。
//
// 与 restir_shade.slang 同算法，但 visibility 测试使用硬件 RayQuery
// （VK_KHR_ray_query）替代 VXGI 体素网格 alpha ray march。
//
// 输入：GBuffer + 光源 SSBO + 最终 reservoir（来自 spatial pass 输出 B）
// 输出：rt.restir RGBA16F（rgb = 直接光辐射度，a = 1.0 标志"已写入"）

import shared_types;

struct ShadePC {
    uint2  outSize;
    float2 invOutSize;
    uint   numLights;
    float  shadowSteps;     // unused (RT shadow is a single RayQuery)
    float  intensityScale;
    uint   _pad0;
};
[[vk::push_constant]] ShadePC gPC;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms>          gFrame;
[[vk::binding(1, 0)]] Texture2D<float4>                      gAlbedoMetal;
[[vk::binding(2, 0)]] Texture2D<float4>                      gNormalRough;
[[vk::binding(3, 0)]] Texture2D<float>                       gDepth;
[[vk::binding(4, 0)]] StructuredBuffer<PointLight>           gLights;
[[vk::binding(5, 0)]] Texture2D<uint4>                       gReservoir;
[[vk::binding(6, 0)]] RaytracingAccelerationStructure        gTLAS;
[[vk::binding(7, 0)]] RWTexture2D<float4>                    gOutRestir;

// ---- ReSTIR common types + functions (inlined to avoid import issues) ----

struct PointLight {
    float3 pos;
    float  radius;
    float3 color;
    float  intensity;
};

struct Reservoir {
    uint  lightIdx;
    float wSum;
    float W;
    uint  M;
};

uint4 packReservoir(Reservoir r) {
    return uint4(r.lightIdx, asuint(r.wSum), asuint(r.W), r.M);
}
Reservoir unpackReservoir(uint4 p) {
    Reservoir r;
    r.lightIdx = p.x;
    r.wSum = asfloat(p.y);
    r.W = asfloat(p.z);
    r.M = p.w;
    return r;
}

// ---- World position reconstruction ----

float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc.x, -ndc.y, depth, 1.0);
    float4 v = mul(gFrame.invViewProj, clip);
    return v.xyz / v.w;
}

// ---- HW RT shadow visibility (replaces voxel alpha marching) ----

bool visibilityRt(float3 surfPos, float3 lightPos, float3 N) {
    float3 dir = lightPos - surfPos;
    float dist = length(dir);
    float3 L = dir / dist;

    // Offset along N to avoid self-intersection
    float3 origin = surfPos + N * 0.02;

    RayDesc shadowRay;
    shadowRay.Origin = origin;
    shadowRay.Direction = L;
    shadowRay.TMin = 0.01;
    shadowRay.TMax = dist - 0.02;  // stop just before the light

    RayQuery<RAY_FLAG_CULL_NON_OPAQUE |
             RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(gTLAS, RAY_FLAG_NONE, 0xFF, shadowRay);
    q.Proceed();

    return q.CommittedStatus() != COMMITTED_TRIANGLE_HIT;
}

// ---- Main compute ----

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (any(dt.xy >= gPC.outSize)) return;
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) {
        gOutRestir[pix] = float4(0, 0, 0, 0);
        return;
    }

    Reservoir r = unpackReservoir(gReservoir.Load(int3(pix, 0)));
    if (r.lightIdx == 0xFFFFFFFFu || r.W <= 0.0) {
        gOutRestir[pix] = float4(0, 0, 0, 0);
        return;
    }

    float4 nr = gNormalRough.Load(int3(pix, 0));
    float3 N = normalize(nr.xyz);
    float4 am = gAlbedoMetal.Load(int3(pix, 0));
    float3 albedo = am.rgb * (1.0 - am.a);
    float3 wp = worldFromDepth(pix, depth);

    PointLight L = gLights[r.lightIdx];

    float3 toL = L.pos - wp;
    float dist2 = max(dot(toL, toL), 1e-4);
    float invDist = rsqrt(dist2);
    float3 Ldir = toL * invDist;
    float NdotL = max(0.0, dot(N, Ldir));

    // 唯一一次 RT 可见性测试（覆盖整个 reservoir 的有效估计）
    float vis = (NdotL > 0.0) ? (visibilityRt(wp, L.pos, N) ? 1.0 : 0.0) : 0.0;

    float3 Li = L.color * L.intensity;
    float3 contrib = albedo * (1.0 / 3.14159265) * Li * NdotL / dist2;
    float3 result = contrib * vis * r.W * gPC.intensityScale;

    gOutRestir[pix] = float4(result, 1.0);
}
