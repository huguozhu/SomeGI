// Octahedral 编解码 —— DDGI / Lumen 等技术用来把球面方向映射到 2D UV，
// 比 cube map 更 cache-friendly（一张 2D atlas 就能存所有方向）。
//
// 数学：单位方向 d → 2D 八面体投影 → (u,v) ∈ [0,1]²。回来反一下。
// 核心是 |x|+|y|+|z| = 1 的八面体 → flatten 到平面，下半球翻折。

float2 signNotZero(float2 v) {
    return float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

// 单位方向 → [-1,1]² octahedral coord
float2 dirToOct(float3 d) {
    float l1 = abs(d.x) + abs(d.y) + abs(d.z);
    float2 p = d.xy * (1.0 / l1);
    if (d.z < 0.0) {
        // 下半球翻折
        p = (1.0 - abs(p.yx)) * signNotZero(p);
    }
    return p;
}

// [-1,1]² octahedral coord → 单位方向
float3 octToDir(float2 p) {
    float3 d = float3(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (d.z < 0.0) {
        d.xy = (1.0 - abs(d.yx)) * signNotZero(d.xy);
    }
    return normalize(d);
}

// 单位方向 → [0,1]² UV（atlas 用）。
float2 dirToOctUv(float3 d) {
    return dirToOct(d) * 0.5 + 0.5;
}

// [0,1]² UV → 单位方向
float3 octUvToDir(float2 uv) {
    return octToDir(uv * 2.0 - 1.0);
}
