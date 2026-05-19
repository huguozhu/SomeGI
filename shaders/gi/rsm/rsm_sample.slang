// RSM 采样 compute —— 把 RSM 当 VPL 数组读，每屏幕像素抽 N 个 disk 样本
// 累加一次反弹漫反射间接光（Dachsbacher & Stamminger 2005）。
//
// 算法（标准 RSM gathering）：
// 1. 屏幕像素 pix 从 GBuffer / depth 重建 worldPos `p` 与 worldNormal `N`。
// 2. 把 p 投到 sun light space → RSM UV `centerUv`。屏幕外 / 落在 RSM 之外
//    的像素直接 0（rsmGI.a=0，下游 lerp 不会污染 IBL diffuse）。
// 3. 在 centerUv 周围 disk 上抽 sampleCount 个 Hammersley + per-pixel
//    哈希抖动样本：
//      r = sqrt(xi.x)          // 均匀 disk
//      phi = 2π · xi.y
//      offset = (r·cosφ, r·sinφ) · radius
//      uv = centerUv + offset
// 4. 每个样本读 (p_vpl, N_vpl, flux_vpl) 三张 RSM 图。
// 5. 累加 RSM 论文权重：
//      E += flux_vpl
//           · max(0, dot(N_vpl, p - p_vpl))     // VPL 朝接收点
//           · max(0, dot(N,    p_vpl - p))      // 接收点朝 VPL
//           · |offset|² / |p - p_vpl|⁴
//      其中 |offset|² 是均匀 disk → 论文 importance sampling 补偿因子；
//      |p - p_vpl|⁴ 是几何衰减（form factor 的 1/r²，一对法线点积已带）。
// 6. 输出：
//      rgb = sumE / sampleCount · intensity   （未乘 albedo，留给 lighting.slang）
//      a   = hits / sampleCount                （覆盖率 / 置信度，给 LightingPass lerp 用）
//
// 注意：
// - RSM 输入纹理在 RsmGeometryPass 收尾后已是 SHADER_READ_ONLY。
// - sunViewProj 含 Vulkan Y-flip（与主 proj 一致），所以 ndc.xy*0.5+0.5
//   能直接喂给 SampleLevel 拿 RSM 像素。
// - 输出 rsmGI 是 swapchain-sized RGBA16F。SSGI 同模式（rgba: irradiance/cov）。

import shared_types;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms>    gFrame;
[[vk::binding(1, 0)]] Texture2D<float4>                gNormalRough;     // GBuffer
[[vk::binding(2, 0)]] Texture2D<float>                 gDepth;           // GBuffer depth
[[vk::binding(3, 0)]] Texture2D<float4>                gRsmPosition;     // RSM RT0
[[vk::binding(4, 0)]] Texture2D<float4>                gRsmNormal;       // RSM RT1
[[vk::binding(5, 0)]] Texture2D<float4>                gRsmFlux;         // RSM RT2
[[vk::binding(6, 0)]] SamplerState                     gLinearClamp;
[[vk::binding(7, 0)]] ConstantBuffer<RsmFrameUniforms> gRsm;
[[vk::binding(8, 0)]] RWTexture2D<float4>              gOutRsmGI;

struct RsmSamplePC {
    uint2 outSize;
    float2 invOutSize;
    float radius;       // disk 半径（sun light space UV 单位，[0,1] 范围内）
    uint  sampleCount;
    float intensity;    // ImGui 调的整体强度倍率
    uint  _pad;
};
[[vk::push_constant]] RsmSamplePC gPC;

// 与 GBuffer 的 worldFromDepth 对齐（与 ssgi.slang 同实现）。
float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gFrame.invViewProj, clip);
    return w.xyz / w.w;
}

float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

uint hash2(uint2 p) {
    uint h = p.x * 73856093u ^ p.y * 19349663u;
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13;
    return h;
}

// per-pixel 抖动让相邻像素抽不同 disk 偏移，弱化低 sampleCount 时的 banding。
float2 hammersleyJittered(uint i, uint N, uint2 pix) {
    float jitter = float(hash2(pix)) * (1.0 / 4294967296.0);
    float u = frac(float(i) / float(N) + jitter);
    float v = radicalInverse_VdC(i);
    return float2(u, v);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) return;
    uint2 pix = dt.xy;

    // 跳过天空（depth==1）—— 没有 surface，没法 gather。
    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) { gOutRsmGI[pix] = float4(0, 0, 0, 0); return; }

    float3 N = normalize(gNormalRough.Load(int3(pix, 0)).xyz);
    float3 p = worldFromDepth(pix, depth);

    // p → sun light space → RSM UV。sunViewProj 含 Y-flip，所以与
    // gRsmPosition.SampleLevel 的 UV 约定（origin top-left）一致。
    float4 lp = mul(gRsm.sunViewProj, float4(p, 1.0));
    if (lp.w <= 0.0) { gOutRsmGI[pix] = float4(0, 0, 0, 0); return; }
    float3 lndc = lp.xyz / lp.w;
    float2 centerUv = lndc.xy * 0.5 + 0.5;

    // sun frustum 之外（centerUv 出 [0,1]）的像素，给 a=0 让下游 fallback 到 IBL。
    if (any(saturate(centerUv) != centerUv)) {
        gOutRsmGI[pix] = float4(0, 0, 0, 0);
        return;
    }

    float3 sumE = float3(0);
    uint hits = 0;
    for (uint i = 0; i < gPC.sampleCount; ++i) {
        float2 xi = hammersleyJittered(i, gPC.sampleCount, pix);
        // 均匀 disk 采样：r = sqrt(xi.x)，phi = 2π·xi.y。
        float r = sqrt(xi.x);
        float phi = 6.2831853 * xi.y;
        float2 offset = float2(r * cos(phi), r * sin(phi)) * gPC.radius;
        float2 uv = centerUv + offset;
        if (any(saturate(uv) != uv)) continue;

        float3 p_vpl = gRsmPosition.SampleLevel(gLinearClamp, uv, 0).xyz;
        float3 N_vpl = normalize(gRsmNormal.SampleLevel(gLinearClamp, uv, 0).xyz);
        float3 flux_vpl = gRsmFlux.SampleLevel(gLinearClamp, uv, 0).rgb;

        float3 dir = p - p_vpl;
        float dist2 = max(dot(dir, dir), 1e-4);
        float invDist = rsqrt(dist2);
        float3 dirN = dir * invDist;

        float c1 = max(0.0, dot(N_vpl, dirN));   // VPL 法线朝向接收点（cos）
        float c2 = max(0.0, dot(N, -dirN));      // 接收点法线朝向 VPL

        // RSM 论文权重：weight = c1·c2 · |xi_offset|² / dist⁴。
        // |xi_offset|² 是均匀 disk → 论文 importance sampling 补偿因子
        // （中心样本 dist 小贡献大，靠 |offset|² 降权使估计无偏）。
        float distSqr2 = dist2 * dist2;
        float w = c1 * c2 * dot(offset, offset) / distSqr2;
        sumE += flux_vpl * w;
        hits++;
    }

    // 平均到 sampleCount（不是 hits）—— 与 SSGI 同：a 编码覆盖率，rgb
    // 是 unbiased estimator。这样 hits=0 时 rgb=0 不会出 NaN。
    float3 avgE = sumE / float(gPC.sampleCount);
    float hitFrac = float(hits) / float(gPC.sampleCount);
    gOutRsmGI[pix] = float4(avgE * gPC.intensity, hitFrac);
}
