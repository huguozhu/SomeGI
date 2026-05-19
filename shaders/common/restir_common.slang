// ReSTIR DI 公共类型 / 工具 —— C.4。
//
// 参考：Bitterli et al. 2020 "Spatiotemporal reservoir resampling for
// real-time ray tracing with dynamic direct lighting"。
//
// 核心思想：每像素维护一个 reservoir，持有"当前选中"的光源 + RIS 权重 W。
// initial 阶段对 M 个 uniform 候选做 weighted-resampling 收敛到 1；
// spatial / temporal 阶段把邻居 / 上帧的 reservoir 合并进来 → 等价于
// 看了更多样本，方差大幅降。
//
// 简化点（vs 论文）：
//   - 屏幕空间 reservoir，无 GBuffer-aware reproject（不做 temporal）
//   - 单 spatial pass，K=4 邻居
//   - RIS targetPdf 不含 visibility（unshadowed）→ shade 阶段才做 voxel
//     grid alpha-only 可视性近似（biased estimator，常用 demo 变体）
//   - 光源仅 point lights（color + intensity + radius=0 的 δ 光）

struct PointLight {
    float3 pos;
    float  radius;
    float3 color;
    float  intensity;
};

// reservoir 紧凑打包：u32×4 = 16 字节。
//   r = lightIdx (uint, 0..N-1；0xFFFFFFFF = invalid)
//   g = asuint(wSum)
//   b = asuint(W)
//   a = M (uint)
//
// W = 1 / p_hat(picked) × wSum / M（RIS 标准权重）。shade 阶段直接乘 W。

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
Reservoir emptyReservoir() {
    Reservoir r;
    r.lightIdx = 0xFFFFFFFFu;
    r.wSum = 0.0;
    r.W = 0.0;
    r.M = 0;
    return r;
}

// Bitterli reservoir update（"streaming RIS"）：
//   r_new.M += 1
//   r_new.wSum += w
//   if rand() < w / r_new.wSum:  r_new.lightIdx = candidate
bool reservoirUpdate(inout Reservoir r, uint candidate, float w, float xi) {
    r.wSum += w;
    r.M += 1u;
    if (r.wSum > 0.0 && xi < (w / r.wSum)) {
        r.lightIdx = candidate;
        return true;
    }
    return false;
}

// 合并两个 reservoir（spatial reuse）：
//   r_a.M_a 个样本 + r_b.M_b 个样本 → 等价 r_combined.M = M_a + M_b。
//   每边按其当前权重投票 r.W * r.M * targetPdf(picked at center)
//   论文 Algorithm 4。
//
// 这里 targetPdfAtCenter 是当前像素位置上对 r_b 选中的 light 算出的
// p_hat（含 brdf × Le × cos / dist²，但 unshadowed）。
void reservoirCombine(inout Reservoir r,
                      Reservoir other,
                      float targetPdfAtCenterForOther,
                      float xi) {
    if (other.M == 0u) return;
    float w = targetPdfAtCenterForOther * other.W * float(other.M);
    r.wSum += w;
    r.M += other.M;
    if (r.wSum > 0.0 && xi < (w / r.wSum)) {
        r.lightIdx = other.lightIdx;
    }
}

// ---------- Hash / RNG -----------------------------------------------------
uint pcgHash(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float hashFloat(uint v) {
    return float(pcgHash(v) & 0x00FFFFFFu) / float(0x01000000u);
}
