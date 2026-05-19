// ReSTIR DI shade pass —— C.4 第三步。
//
// 读最终 reservoir → 计算 picked light 的实际 shaded radiance × W。
// 此处做唯一一次 visibility 测试（voxelGrid alpha-only ray march）：把
// 这个开销摊到全屏 1 次而非 RIS 的 M 次，是 ReSTIR 的核心成本节省。
//
// 输出 rt.restir RGBA16F：rgb = 直接光辐射度（lighting 端 += 之），
// a = 1.0 标志"已写入"。

import shared_types;
import restir_common;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] Texture2D<float4>             gAlbedoMetal;
[[vk::binding(2, 0)]] Texture2D<float4>             gNormalRough;
[[vk::binding(3, 0)]] Texture2D<float>              gDepth;
[[vk::binding(4, 0)]] StructuredBuffer<PointLight>  gLights;
[[vk::binding(5, 0)]] Texture2D<uint4>              gReservoir;     // 取自 spatial 输出 B
[[vk::binding(6, 0)]] Texture3D<float4>             gVoxelGrid;     // alpha-only ray march
[[vk::binding(7, 0)]] SamplerState                  gLinearClamp;
[[vk::binding(8, 0)]] RWTexture2D<float4>           gOutRestir;

struct ShadePC {
    uint2  outSize;
    float2 invOutSize;
    uint   numLights;
    float  shadowSteps;
    float  intensityScale;   // ImGui 总体调节
    uint   _pad0;
};
[[vk::push_constant]] ShadePC gPC;

float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc.x, -ndc.y, depth, 1.0);
    float4 v = mul(gFrame.invViewProj, clip);
    return v.xyz / v.w;
}

// 把 worldPos → voxel grid uvw（[0,1]）
float3 worldToVoxelUv(float3 wp) {
    float3 p = (wp - gFrame.vxgiGridMinCell.xyz) /
               (float(gFrame.vxgiCounts.x) * gFrame.vxgiGridMinCell.w);
    return p;
}

// Beer-style transmittance 沿 surface→light 路径采 voxelGrid alpha 累乘
float visibilityVoxel(float3 surfPos, float3 lightPos, float3 N, int steps) {
    if (gFrame.vxgiCounts.x <= 0) return 1.0;   // VXGI 没建：默认可见
    // 起点沿 N offset 半 cell 防自交
    float cellSize = gFrame.vxgiGridMinCell.w;
    float3 origin = surfPos + N * (cellSize * 0.75);
    float3 dir = lightPos - origin;
    float trans = 1.0;
    for (int s = 1; s <= steps; ++s) {
        float t = float(s) / float(steps + 1);   // 中点段，避开两端 surface
        float3 p = origin + dir * t;
        float3 uv = worldToVoxelUv(p);
        if (any(uv < 0.0) || any(uv > 1.0)) continue;
        float a = gVoxelGrid.SampleLevel(gLinearClamp, uv, 0).a;
        trans *= max(0.0, 1.0 - a);
        if (trans < 0.01) break;
    }
    return trans;
}

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

    // 唯一一次 visibility 测试（覆盖整个 reservoir 的有效估计）
    float vis = (NdotL > 0.0)
        ? visibilityVoxel(wp, L.pos, N, int(gPC.shadowSteps))
        : 0.0;

    float3 Li = L.color * L.intensity;
    // 完整 contribution = (albedo / PI) × Li × NdotL / dist²
    float3 contrib = albedo * (1.0 / 3.14159265) * Li * NdotL / dist2;
    float3 result = contrib * vis * r.W * gPC.intensityScale;

    gOutRestir[pix] = float4(result, 1.0);
}
