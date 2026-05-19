// DDGI probe classification —— B.5：每 probe 看自己 64 rays 的命中距离，
// 如果"近距离命中比例 > 阈值"判定为 inactive（probe 卡在墙体内部 / 角落，
// 提供的 irradiance 是无效的几乎全黑数据）。
//
// dispatch (probeCount + 63) / 64，每 thread = 1 个 probe。
//
// 比例 > closeHitFrac（默认 0.7）→ state = 0 (inactive)；否则 1 (active)。
// lighting.slang 在 8-probe blend 时把 inactive probe 权重清 0，避免它
// 拉低周围 active probe 的 contribution。

[[vk::binding(0, 0)]] StructuredBuffer<float4>   gRayBuf;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gProbeStates;

struct ClassifyPC {
    uint  probeCount;
    uint  raysPerProbe;
    float closeHitDist;     // 距离阈值（小于这个 = 近距离命中）
    float closeHitFrac;     // 比例阈值（超过 = 标 inactive）
};
[[vk::push_constant]] ClassifyPC gPC;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint p = dt.x;
    if (p >= gPC.probeCount) return;

    uint base = p * gPC.raysPerProbe * 2;
    uint closeCnt = 0;
    for (uint i = 0; i < gPC.raysPerProbe; ++i) {
        // ray buffer slot 1 = (hitRgb, hitDist)
        float dist = gRayBuf[base + i*2 + 1].w;
        if (dist < gPC.closeHitDist) closeCnt++;
    }
    float frac = float(closeCnt) / float(gPC.raysPerProbe);
    gProbeStates[p] = (frac > gPC.closeHitFrac) ? 0u : 1u;
}
