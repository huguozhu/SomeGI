// SDFGI trace —— C.3 第四步：屏幕空间的 N 射线 sphere-trace。
//
// 思路（Godot 4 SDFGI 简化版）：每像素从 GBuffer 还原 worldPos+normal，
// 在表面半球（cosine-weighted）射 N 条光线，用 SDF 做 sphere-stepping
// （每步走 SDF[当前点] 个 cell，保证不穿入 occupied 区域）。命中时取
// voxelGrid mip 0 的 radiance（已含 direct sun 注入 + C.2 多 bounce 反弹）。
//
// 与 VXGI cone trace 的对比：
//   - cone trace 是固定步进 + mip 锥度积分，不能精确到表面 → 厚墙 leaking、
//     thin geometry 漏光。
//   - sphere-trace 每步等于 SDF 值，自适应步长，逼近表面到 sub-cell；
//     thin geometry 的可见性更准。
//
// 输出 rt.ssgi（与 SSGI/GTGI 共享），lighting.slang 不区分来源。
//
// 简化点（vs Godot 4）：
//   - 单 cascade（128³，跟随场景 AABB 静态）
//   - 不用 light probes，直接采 voxelGrid radiance 作出射 radiance
//   - 不做时序累积（首版；可作后续 polish）

import shared_types;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] Texture2D<float4>   gNormalRough;
[[vk::binding(2, 0)]] Texture2D<float>    gDepth;
[[vk::binding(3, 0)]] Texture3D<float>    gSdf;          // R16F UDF（cell 单位）
[[vk::binding(4, 0)]] Texture3D<float4>   gVoxelGrid;    // RGBA16F radiance + alpha
[[vk::binding(5, 0)]] SamplerState        gLinearClamp;
[[vk::binding(6, 0)]] RWTexture2D<float4> gOutSdfgi;

struct SdfgiPC {
    uint2  outSize;
    float2 invOutSize;
    uint   numRays;       // 4 / 8 / 16
    uint   maxSteps;      // sphere-step 上限（默认 48）
    float  rayMaxCells;   // 单射线最大行进 cell 距离（默认 96）
    float  hitEpsCells;   // 命中阈值（cell 单位，默认 0.6）
    uint   frameIndex;    // 用于 hash 抖动
    uint   _pad0, _pad1, _pad2;
};
[[vk::push_constant]] SdfgiPC gPC;

// ---------- 简单 hash（PCG-style 32-bit）-------------------------------
uint pcgHash(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float hash01(uint v) { return float(pcgHash(v) & 0x00FFFFFFu) / float(0x01000000u); }
float2 hash2(uint a, uint b) { return float2(hash01(a), hash01(b)); }

// ---------- TBN：法线到 world basis ------------------------------------
void buildTBN(float3 N, out float3 T, out float3 B) {
    float3 up = abs(N.y) < 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

// cosine-weighted 半球采样（Malley 方法）
float3 sampleCosineHemi(float2 xi) {
    float r = sqrt(xi.x);
    float phi = 6.2831853 * xi.y;
    float x = r * cos(phi);
    float z = r * sin(phi);
    float y = sqrt(max(0.0, 1.0 - xi.x));
    return float3(x, y, z);   // local: (T, N, B) 用 y 当 N 通量
}

// ---------- worldPos 重建（从 depth）-----------------------------------
float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    // Vulkan：clip y 翻转。NormalRough/Depth 写入 framebuffer 时 y 已"屏幕向下"，
    // 还原 worldPos 时直接用 ndc.y（与 GBuffer 写法保持一致）即可。
    float4 clip = float4(ndc.x, -ndc.y, depth, 1.0);
    float4 v = mul(gFrame.invViewProj, clip);
    return v.xyz / v.w;
}

// ---------- voxelGrid / SDF 采样（cell 坐标 → UV）---------------------
float3 cellToUv(float3 cellPos) {
    return cellPos / float(gFrame.vxgiCounts.x);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (any(dt.xy >= gPC.outSize)) return;
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) {
        gOutSdfgi[pix] = float4(0, 0, 0, 0);
        return;
    }
    float4 nr = gNormalRough.Load(int3(pix, 0));
    float3 N = normalize(nr.xyz);
    float3 wp = worldFromDepth(pix, depth);

    // 把 worldPos 转 cell 坐标（仍是浮点）
    float3 gridMin = gFrame.vxgiGridMinCell.xyz;
    float  cellSize = gFrame.vxgiGridMinCell.w;
    int    R = gFrame.vxgiCounts.x;
    float3 cellPos = (wp - gridMin) / cellSize;

    // 表面外推半 cell，避免起点就在 occupied 内（自交）
    float3 startCell = cellPos + N * 1.0;

    float3 T, B;
    buildTBN(N, T, B);

    float3 acc = float3(0);
    float coverage = 0.0;
    uint base = pcgHash(pix.x * 1973u + pix.y * 9277u + gPC.frameIndex * 26699u);

    for (uint i = 0; i < gPC.numRays; ++i) {
        float2 xi = hash2(base + i * 2u, base + i * 2u + 1u);
        float3 local = sampleCosineHemi(xi);
        // local frame：x=T, y=N, z=B
        float3 dir = normalize(local.x * T + local.y * N + local.z * B);

        // sphere-trace through SDF
        float traveled = 0.0;
        float3 p = startCell;
        bool hit = false;
        for (uint s = 0; s < gPC.maxSteps; ++s) {
            float3 uv = cellToUv(p);
            // 出 grid → miss（sky / IBL fallback：暂记为 0；后续 polish 加 sky）
            if (any(uv < 0.0) || any(uv > 1.0)) break;
            float d = gSdf.SampleLevel(gLinearClamp, uv, 0).x;
            if (d <= gPC.hitEpsCells) {
                // 命中：取此处 voxelGrid radiance
                float4 c = gVoxelGrid.SampleLevel(gLinearClamp, uv, 0);
                acc += c.rgb;
                coverage += 1.0;
                hit = true;
                break;
            }
            // 步进 d cell（最少推进 1 cell 防 stuck）
            float step = max(d, 0.5);
            p += dir * step;
            traveled += step;
            if (traveled >= gPC.rayMaxCells) break;
        }
        // 没命中 → 这条 ray 不贡献（black/0）；coverage 不增。
    }
    float invN = 1.0 / float(gPC.numRays);
    acc *= invN;
    coverage *= invN;

    // 输出与 SSGI/GTGI 一致：rgb=入射 radiance（PI 已隐含在 cosine 采样中），
    // a=hit fraction（lighting 端可用作渐变）。
    gOutSdfgi[pix] = float4(acc, coverage);
}
