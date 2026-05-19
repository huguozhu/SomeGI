// SDFGI seed —— C.3 第一步：从 voxelGrid alpha 生成 JFA 种子图。
//
// 思路：JFA（Jump Flooding Algorithm）算 UDF（unsigned distance field）。
// JFA 的状态是"每 cell 记录最近 seed 的位置"；初始时只有 seed 自己有
// 自己的位置，其余 cell 写 sentinel（用 a=0 标记）。后续 jfa 步用 a 标志
// 区分有效条目。
//
// seed 判定：voxelGrid alpha >= threshold（默认 0.05）即认为 cell 占用
//（与 lighting cone trace 的 alpha 阈值一致）。seed 的存储是 cell 中心
// 的 cell 坐标（int3 → float3），方便后续 finalize 直接 length 计算。

[[vk::binding(0, 0)]] Texture3D<float4>   gVoxelGrid;     // mip 0：alpha = occupancy
[[vk::binding(1, 0)]] RWTexture3D<float4> gSeedOut;       // rgb=cellPos, a=isSeed

struct SeedPC {
    uint  resolution;
    float threshold;
    float _pad0, _pad1;
};
[[vk::push_constant]] SeedPC gPC;

[shader("compute")]
[numthreads(4, 4, 4)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (any(dt >= uint3(gPC.resolution, gPC.resolution, gPC.resolution))) return;

    float a = gVoxelGrid.Load(int4(int3(dt), 0)).a;
    if (a >= gPC.threshold) {
        // 该 cell 自己是 seed —— 记录自身坐标，标志 1.0
        gSeedOut[dt] = float4(float3(dt), 1.0);
    } else {
        // sentinel：a=0 表示尚无有效 seed
        gSeedOut[dt] = float4(0, 0, 0, 0);
    }
}
