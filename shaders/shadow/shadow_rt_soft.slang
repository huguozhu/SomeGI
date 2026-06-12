// shadow_rt_soft.slang — RT 软阴影 resolve compute shader。
//
// 每像素发射 8 条 shadow ray，在太阳方向锥体内随机采样，
// 模拟面光源产生半影（penumbra）。射线越多越平滑。
//
// 管线约定：
//   set=0:
//     binding 0: RaytracingAccelerationStructure  — TLAS
//     binding 1: RWTexture2D<float>               — shadowMask 输出
//     binding 2: Texture2D<float4>                 — gNormalRough
//   set=1:
//     binding 0: ConstantBuffer<FrameUniforms>     — invViewProj + sunDir
//     binding 1: Texture2D<float>                  — gDepth
//   push constant: RtSoftPC { outSize, invOutSize, frameIndex, sunRadius }

import shared_types;

// ── Push constant ──
struct RtSoftPC {
    uint2  outSize;
    float2 invOutSize;
    uint   frameIndex;      // 用于随机种子变化
    float  sunRadius;        // 太阳角半径（radians），0.01~0.10
    uint   rayCount;         // 每像素采样数，4~32
    uint   _pad;
};
[[vk::push_constant]] RtSoftPC gPC;

// ── set=0 ──
[[vk::binding(0, 0)]] RaytracingAccelerationStructure gTLAS;
[[vk::binding(1, 0)]] RWTexture2D<float>              gOutMask;
[[vk::binding(2, 0)]] Texture2D<float4>               gNormalRough;

// ── set=1 ──
[[vk::binding(0, 1)]] ConstantBuffer<FrameUniforms>   gFrame;
[[vk::binding(1, 1)]] Texture2D<float>                gDepth;

// ── 简单伪随机数生成器（PCG 变体）──
uint pcg(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float rand01(inout uint seed) {
    seed = pcg(seed);
    return float(seed) / 4294967296.0;
}

// ── 从 GBuffer depth 重建世界坐标 ──
float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gFrame.invViewProj, clip);
    return w.xyz / w.w;
}

// ── 在方向 L 周围锥角 angle 内均匀采样 ──
float3 coneSample(float3 L, float angle, float2 xi) {
    float cosA = cos(angle);
    float z = 1.0 - xi.x * (1.0 - cosA);
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 6.2831853 * xi.y;
    // 局部坐标系：以 L 为 Z 轴
    float3 u = (abs(L.x) > 0.99) ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t = normalize(cross(u, L));
    float3 b = cross(L, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + L * z);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) return;
    uint2 pix = dt.xy;

    // ── 1. 读取 GBuffer depth ──
    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) {
        gOutMask[pix] = 1.0;
        return;
    }

    // ── 2. 重建世界坐标 + 法线 ──
    float3 worldPos = worldFromDepth(pix, depth);
    float3 N = normalize(gNormalRough.Load(int3(pix, 0)).xyz);
    float3 L = normalize(-gFrame.sunDir.xyz);

    // ── 3. 随机种子 ──
    uint seed = (pix.y * gPC.outSize.x + pix.x) * 1664525u + gPC.frameIndex * 22695477u;

    // ── 4. N 条 ray 在锥体内采样 ──
    float shadowSum = 0.0;
    int n = int(clamp(gPC.rayCount, 4u, 32u));
    for (int i = 0; i < n; ++i) {
        float2 xi = float2(rand01(seed), rand01(seed));
        float3 dir = coneSample(L, gPC.sunRadius, xi);

        // 沿随机方向 + 法线偏移（增大偏移量 + 背面剔除避免自阴影）
        float3 rayOrigin = worldPos + N * 0.1 + dir * 0.05;

        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = dir;
        ray.TMin      = 0.01;
        ray.TMax      = 100.0;

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
                 RAY_FLAG_CULL_BACK_FACING_TRIANGLES |
                 RAY_FLAG_CULL_NON_OPAQUE |
                 RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
        q.TraceRayInline(gTLAS, RAY_FLAG_NONE, 0xFF, ray);
        q.Proceed();

        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            shadowSum += 1.0;   // 未命中 → lit
        }
    }
    gOutMask[pix] = shadowSum / float(n);
}
