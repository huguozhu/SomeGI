// SDFGI finalize —— C.3 第三步：把 JFA 收敛后的 "最近 seed 位置" 转成
// scalar UDF（unsigned distance field），单位是 cell。
//
// d = length(cellPos - seedPos)，存到 R16F 3D image。空 cell（无 seed
// 命中，理论上 JFA 收敛后不该出现）写一个大值当 fallback。
//
// 注：这里不做 sign（in/out）—— 我们只需 UDF 用作 sphere-tracing 步长。
// trace 的射线起点在 voxelGrid 外部边界（视点方向），一路向场景，UDF 保证
// 永不步进进 occupied cell 之内，sphere-stepping 收敛到表面。

[[vk::binding(0, 0)]] Texture3D<float4>   gSeed;          // 收敛后的 seed 图
[[vk::binding(1, 0)]] RWTexture3D<float>  gUdfOut;        // R16F：UDF in cell units

struct FinalizePC {
    uint  resolution;
    float maxDist;        // 写入上限（避免 16F 溢出 / 标记 "no seed"）
    float _pad0, _pad1;
};
[[vk::push_constant]] FinalizePC gPC;

[shader("compute")]
[numthreads(4, 4, 4)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (any(dt >= uint3(gPC.resolution, gPC.resolution, gPC.resolution))) return;

    float4 s = gSeed.Load(int4(int3(dt), 0));
    float d;
    if (s.a < 0.5) {
        d = gPC.maxDist;
    } else {
        float3 here = float3(dt);
        d = length(s.xyz - here);
    }
    gUdfOut[dt] = min(d, gPC.maxDist);
}
