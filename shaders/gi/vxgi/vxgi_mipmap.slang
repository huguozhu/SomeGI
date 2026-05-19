// VXGI mipmap reduction compute —— 把 src mip 的每 2×2×2 = 8 voxel 平均
// 写到 dst mip 一个 voxel。各向同性（无方向性 mip）。
//
// 算法：
//   dispatch (dstSize/4)³。每 thread = 一个 dst voxel。
//   读 src mip 8 个 voxel；按 opacity-weighted 平均累加到 dst：
//     dst.rgb = sum(src_i.rgb · src_i.a) / max(sumAlpha, eps)
//     dst.a   = sum(src_i.a) / 8
//   这样 alpha=0 的"空" cell 不污染颜色，alpha=1 的实心 cell 主导聚合。
//
// 调用方 ping-pong：
//   for (level = 1; level < mipLevels; ++level)
//     bindResources(srcMip = level-1, dstMip = level)
//     barrier(srcMip → SHADER_READ_ONLY, dstMip → GENERAL)
//     record(dstSize)
//
// 注：用 SampleLevel 读 src 时，src 必须是带 sampler 的 SAMPLED_IMAGE。
// 这里用 .Load 直接整数取 voxel 即可（dst 视角逐一访问 src 的 8 个邻居），
// 避免 sampler 要求。

[[vk::binding(0, 0)]] Texture3D<float4>   gSrc;
[[vk::binding(1, 0)]] RWTexture3D<float4> gDst;

struct MipmapPC {
    uint dstSize;   // dst mip 边长
    uint _p0, _p1, _p2;
};
[[vk::push_constant]] MipmapPC gPC;

[shader("compute")]
[numthreads(4, 4, 4)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (any(dt >= uint3(gPC.dstSize, gPC.dstSize, gPC.dstSize))) return;

    int3 src0 = int3(dt) * 2;
    float3 sumRgb = float3(0);
    float  sumA   = 0.0;
    [unroll]
    for (int z = 0; z < 2; ++z)
    [unroll]
    for (int y = 0; y < 2; ++y)
    [unroll]
    for (int x = 0; x < 2; ++x) {
        float4 c = gSrc.Load(int4(src0 + int3(x, y, z), 0));
        sumRgb += c.rgb * c.a;
        sumA   += c.a;
    }

    float3 avgRgb = (sumA > 1e-4) ? (sumRgb / sumA) : float3(0);
    gDst[dt] = float4(avgRgb, sumA / 8.0);
}
