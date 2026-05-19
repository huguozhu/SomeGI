// LPV 注入 compute —— 把 RSM 当 VPL 数组，逐 RSM texel 投到 LPV 3D 网格
// 对应 cell 的 SH 系数。
//
// 输入：M5 RsmGeometryPass 输出的 3 张 RSM 图（worldPos / worldNormal / flux）。
// 输出：LpvGrid current() 的 3 张 RGBA16F 3D image（lpvR/G/B），每张
//       的 RGBA = 该色彩通道的 4 个 SH 系数。
// dispatch：(rsmSize/8, rsmSize/8, 1)，每 thread 一个 RSM texel。
//
// 算法：
// 1. 读 RSM texel：worldPos p, worldNormal n, flux f。flux=0 的 texel
//    （shadow / 背光面）跳过。
// 2. p → grid cell index：(p - gridMin) / cellSize。出界则跳过。
// 3. cosineLobeSh(n) × f.r → lpvR[idx]。同理 G/B。
//
// last-writer-wins：多个 RSM texel 投到同一 cell 时只留最后一次写入。
// 对 32³ grid + 512² RSM，平均每 cell ~256 个 texel 竞争 → 严重欠估能量。
// M6 起步先这样跑通；M6.5+ 改 atomic add（VK_FORMAT_R32_UINT pack 或
// SHADER_ATOMIC_FLOAT 扩展）。
//
// 影响：
// - 密集 RSM 区域的能量比真值低；但因为 SH 是低频表达 + propagate 多次
//   迭代后光会 "扩散"，视觉效果还是看得到。
// - 这也是 Crytek 论文里他们用 32×32×32 = 32k 个 VPL（一张小 RSM）的
//   原因 —— RSM texel 数 ≈ grid cell 数时碰撞少。

import shared_types;
import sh;

[[vk::binding(0,0)]] Texture2D<float4> gRsmPosition;
[[vk::binding(1,0)]] Texture2D<float4> gRsmNormal;
[[vk::binding(2,0)]] Texture2D<float4> gRsmFlux;
[[vk::binding(3,0)]] RWTexture3D<float4> gLpvR;
[[vk::binding(4,0)]] RWTexture3D<float4> gLpvG;
[[vk::binding(5,0)]] RWTexture3D<float4> gLpvB;
[[vk::binding(6,0)]] RWTexture3D<float4> gGv;        // B.8 Geometry Volume：cosineLobeSh(-N)

struct InjectPC {
    uint2  rsmSize;
    uint   gridResolution;
    uint   _pad0;
    float3 gridMin;
    float  cellSize;
};
[[vk::push_constant]] InjectPC gPC;

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.rsmSize.x || dt.y >= gPC.rsmSize.y) return;
    uint2 rpix = dt.xy;

    float3 wp = gRsmPosition.Load(int3(rpix, 0)).xyz;
    float3 wn = gRsmNormal.Load(int3(rpix, 0)).xyz;
    float3 fx = gRsmFlux.Load(int3(rpix, 0)).rgb;

    // 全 0 的 RSM texel（sun frustum 之外，或 RSM 还没渲过的初始 0）。
    if (dot(fx, fx) <= 0.0) return;
    if (dot(wn, wn) <= 0.0) return;
    wn = normalize(wn);

    // worldPos → grid index。VPL 沿 wn 推半个 cell 防自交（投到 cell 外
    // 表面而不是发光面所在的 cell —— 经典 LPV 技巧）。
    float3 vplPos = wp + wn * (0.5 * gPC.cellSize);
    int3 idx = int3(floor((vplPos - gPC.gridMin) / gPC.cellSize));
    if (any(idx < int3(0)) || any(idx >= int(gPC.gridResolution))) return;

    // VPL 的 cosine-weighted SH 投影。flux 已含 NdotL（M5 RSM PS 已乘），
    // 所以这里直接 cosineLobeSh(wn) × flux 即可，不再额外乘 cosine。
    ShCoeffs lobe = cosineLobeSh(wn);

    // last-writer-wins。把 SH·flux 直接 store 到 cell。
    gLpvR[idx] = lobe.c * fx.r;
    gLpvG[idx] = lobe.c * fx.g;
    gLpvB[idx] = lobe.c * fx.b;

    // B.8 Geometry Volume：在 VPL 所在 cell（不沿 N 推半 cell —— GV 描
    // 述的是 surface 自身而非"发光面后方"）写 cosineLobeSh(-N)：lobe 朝
    // 表面背面方向。propagate 时 evalSh(GV, d_light) 高 → d_light 与 -N
    // 同向 → 光线从前方射入表面 → 应被遮挡。
    int3 gvIdx = int3(floor((wp - gPC.gridMin) / gPC.cellSize));
    if (all(gvIdx >= int3(0)) && all(gvIdx < int(gPC.gridResolution))) {
        ShCoeffs gvLobe = cosineLobeSh(-wn);
        gGv[gvIdx] = gvLobe.c;   // last-writer-wins，多 VPL 落同 cell 时取最后一个 N
    }
}
