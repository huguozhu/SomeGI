// shadow_rt_hard.slang — RT 硬阴影 resolve compute shader。
//
// 使用 VK_KHR_ray_query 从每像素发一条 shadow ray 向太阳方向，
// 首次命中即停止。替代 shadow map 渲染 + depth compare 管线。
//
// 管线约定：
//   set=0: RT 专用资源
//     binding 0: RaytracingAccelerationStructure  — TLAS
//     binding 1: RWTexture2D<float>               — shadowMask 输出
//     binding 2: Texture2D<float4>                 — gNormalRough(.xyz=法线)
//   set=1: 复用 m_frameSet
//     binding 0: ConstantBuffer<FrameUniforms>     — invViewProj + sunDir
//     binding 1: Texture2D<float>                  — gDepth
//   push constant: RtHardPC { uint2 outSize; float2 invOutSize; }

import shared_types;

// ── Push constant ──
struct RtHardPC {
    uint2  outSize;
    float2 invOutSize;
};
[[vk::push_constant]] RtHardPC gPC;

// ── set=0 ──
[[vk::binding(0, 0)]] RaytracingAccelerationStructure gTLAS;
[[vk::binding(1, 0)]] RWTexture2D<float>              gOutMask;
[[vk::binding(2, 0)]] Texture2D<float4>               gNormalRough;

// ── set=1 ──
[[vk::binding(0, 1)]] ConstantBuffer<FrameUniforms>   gFrame;
[[vk::binding(1, 1)]] Texture2D<float>                gDepth;

// ── 从 GBuffer depth 重建世界坐标 ──
float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gFrame.invViewProj, clip);
    return w.xyz / w.w;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) return;
    uint2 pix = dt.xy;

    // ── 1. 读取 GBuffer depth ──
    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) {
        gOutMask[pix] = 1.0;   // 天空 / 无限远 → lit
        return;
    }

    // ── 2. 重建世界坐标 + 法线 ──
    float3 worldPos = worldFromDepth(pix, depth);
    float3 N = normalize(gNormalRough.Load(int3(pix, 0)).xyz);
    float3 L = normalize(-gFrame.sunDir.xyz);     // 从表面指向太阳

    // ── 3. 沿法线 + 光线方向偏移起点避免自相交 ──
    //     偏移量需足够大（世界单位），否则 ray 仍会命中起始面
    float3 origin = worldPos + N * 0.05 + L * 0.02;

    // ── 4. 发射 shadow ray ──
    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = L;
    ray.TMin      = 0.01;
    ray.TMax      = 100.0;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
             RAY_FLAG_CULL_NON_OPAQUE |
             RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(gTLAS, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    // ── 5. 命中 → shadow，未命中 → lit ──
    bool inShadow = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
    gOutMask[pix] = inShadow ? 0.0 : 1.0;
}
