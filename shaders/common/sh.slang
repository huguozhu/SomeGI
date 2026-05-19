// 一阶 (order=1) Spherical Harmonics —— LPV 用：每格 4 个 SH 系数即可
// 表达 cosine-weighted 入射 radiance 的低频项。
//
// Basis（real spherical harmonics）：
//   Y00  = 1 / (2·sqrt(π))           ≈ 0.282094792
//   Y1-1 = sqrt(3 / (4π)) · y        ≈ 0.488602512 · y
//   Y10  = sqrt(3 / (4π)) · z        ≈ 0.488602512 · z
//   Y11  = sqrt(3 / (4π)) · x        ≈ 0.488602512 · x
//
// "Cosine lobe" SH（Crytek/Kaplanyan LPV slides 标准系数）：
// 把方向 d 上 max(0, dot(N,d)) 的 cosine kernel 投到 SH，结果是
//   c0 = π · Y00            ≈ 0.886226925
//   c1..c3 = (2π/3) · Y1m   ≈ 1.023326708 · (d.y, d.z, d.x)
//
// 顺序约定 ShCoeffs.c = (c00, c1-1, c10, c11)。所有 LPV 文件共用此约定。

struct ShCoeffs {
    float4 c;   // (Y00, Y1-1, Y10, Y11)
};

static const float kShBasis0 = 0.282094792;   // Y00
static const float kShBasis1 = 0.488602512;   // sqrt(3/(4π))
static const float kShCosine0 = 0.886226925;  // π · Y00
static const float kShCosine1 = 1.023326708;  // (2π/3) · sqrt(3/(4π))

// 把方向 d 处的 clamped cosine 投到 SH。d 应已 normalize。
ShCoeffs cosineLobeSh(float3 d) {
    ShCoeffs sh;
    sh.c = float4(kShCosine0,
                  kShCosine1 * d.y,
                  kShCosine1 * d.z,
                  kShCosine1 * d.x);
    return sh;
}

// 沿方向 N 评估 SH（即给定 N，恢复方向 N 上的 radiance）。
// 用 SH basis 直接 dot —— 不再 cosine 加权，因为 SH 已经在投影时被
// cosine kernel 加权过；但取 max(0,·) 防止负数尾巴造成黑斑。
float evalSh(ShCoeffs sh, float3 N) {
    float4 basis = float4(kShBasis0,
                          kShBasis1 * N.y,
                          kShBasis1 * N.z,
                          kShBasis1 * N.x);
    return max(0.0, dot(sh.c, basis));
}

ShCoeffs addSh(ShCoeffs a, ShCoeffs b)   { ShCoeffs r; r.c = a.c + b.c; return r; }
ShCoeffs scaleSh(ShCoeffs a, float s)    { ShCoeffs r; r.c = a.c * s;   return r; }
