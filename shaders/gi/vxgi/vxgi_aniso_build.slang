// VXGI 各向异性 alpha mipmap 构建 —— B.6。
//
// Crassin 2011 的核心优化：cone trace 走斜方向时，简单 average alpha
// 会低估遮挡（"穿墙"瑕疵）。各向异性 mipmap 给每 cell 在 mip 1+ 存
// 沿 X/Y/Z 三个方向的 Beer-composite alpha，cone trace 按 dir 加权选。
//
// 简化（vs 论文 6 方向）：方向性 alpha 关于轴对称（沿 +X 与 -X 同样的
// composite），所以仅 3 个独立方向，存到 RGBA 的 RGB；A 留 isotropic。
//
// 算法（per output cell，每 thread）：
//   读 src level 的 8 child cells 的 isotropic alpha（src image 是 voxelGrid
//   的 mip i-1 或 anisotropic 的 mip i-1，alpha channel 一致）。
//   对每个轴 axis：把 8 cells 视为 2 (沿 axis) × 4 (横截面) 排列：
//     pathAlpha[k] = 1 - (1 - childAlpha[axis=0, k]) · (1 - childAlpha[axis=1, k])
//                    （沿 axis 串行 2 cell 的 Beer's law 复合）
//     axisAlpha    = avg over k=0..3 of pathAlpha[k]
//   isotropic alpha = avg over 8 children
//   写 dst[i,j,k] = (axisX, axisY, axisZ, isotropic)
//
// 第 0 级输入是 voxelGrid mip 0（直接 alpha = isotropic）；后续 mip 用
// 上一级的 anisotropic alpha 当 child（self-recursive）。论文里有更精
// 细的 mip 0 处理；本里程碑统一用 mip 0 alpha 作为基。

[[vk::binding(0, 0)]] Texture3D<float4>   gSrcVoxel;        // voxel grid full mipchain
[[vk::binding(1, 0)]] Texture3D<float4>   gSrcAniso;        // aniso 上一级 mip 单层 view（per-set bind）
[[vk::binding(2, 0)]] RWTexture3D<float4> gDstAniso;

struct AnisoPC {
    uint dstSize;
    uint useAnisoSrc;   // 0=从 voxelGrid 读（mip 1 build）；1=从 aniso 单 mip view 读
    uint srcLevel;      // 仅 useAnisoSrc=0 时用：voxel 的 mip level；=1 时 view 已是单 mip → mip 0
    uint _pad;
};
[[vk::push_constant]] AnisoPC gPC;

float fetchChildAlpha(int3 idx) {
    if (gPC.useAnisoSrc != 0u) {
        // gSrcAniso 已经是单 mip view，Load mip=0
        return gSrcAniso.Load(int4(idx, 0)).a;
    }
    return gSrcVoxel.Load(int4(idx, gPC.srcLevel)).a;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (any(dt >= uint3(gPC.dstSize, gPC.dstSize, gPC.dstSize))) return;

    int3 src0 = int3(dt) * 2;
    // 取 8 child alphas
    float a000 = fetchChildAlpha(src0 + int3(0,0,0));
    float a100 = fetchChildAlpha(src0 + int3(1,0,0));
    float a010 = fetchChildAlpha(src0 + int3(0,1,0));
    float a110 = fetchChildAlpha(src0 + int3(1,1,0));
    float a001 = fetchChildAlpha(src0 + int3(0,0,1));
    float a101 = fetchChildAlpha(src0 + int3(1,0,1));
    float a011 = fetchChildAlpha(src0 + int3(0,1,1));
    float a111 = fetchChildAlpha(src0 + int3(1,1,1));

    // 沿 X 方向：每 (y,z) 把 x=0 与 x=1 cells 复合，取 4 个截面平均
    float xRow0 = 1.0 - (1.0 - a000) * (1.0 - a100);
    float xRow1 = 1.0 - (1.0 - a010) * (1.0 - a110);
    float xRow2 = 1.0 - (1.0 - a001) * (1.0 - a101);
    float xRow3 = 1.0 - (1.0 - a011) * (1.0 - a111);
    float aX = (xRow0 + xRow1 + xRow2 + xRow3) * 0.25;

    // 沿 Y 方向
    float yRow0 = 1.0 - (1.0 - a000) * (1.0 - a010);
    float yRow1 = 1.0 - (1.0 - a100) * (1.0 - a110);
    float yRow2 = 1.0 - (1.0 - a001) * (1.0 - a011);
    float yRow3 = 1.0 - (1.0 - a101) * (1.0 - a111);
    float aY = (yRow0 + yRow1 + yRow2 + yRow3) * 0.25;

    // 沿 Z 方向
    float zRow0 = 1.0 - (1.0 - a000) * (1.0 - a001);
    float zRow1 = 1.0 - (1.0 - a100) * (1.0 - a101);
    float zRow2 = 1.0 - (1.0 - a010) * (1.0 - a011);
    float zRow3 = 1.0 - (1.0 - a110) * (1.0 - a111);
    float aZ = (zRow0 + zRow1 + zRow2 + zRow3) * 0.25;

    // 各向同性 = 简单平均（与现有 vxgi_mipmap 一致）
    float aIso = (a000 + a100 + a010 + a110 + a001 + a101 + a011 + a111) * 0.125;

    gDstAniso[dt] = float4(aX, aY, aZ, aIso);
}
