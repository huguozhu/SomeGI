// SDFGI JFA step —— C.3 第二步：单次 jump-flooding 迭代。
//
// k = step size（128³ 用 64,32,16,8,4,2,1 共 7 步收敛）。每步对当前 cell
// 取 27 个偏移 ±k 的邻居（含自身），收集这些邻居中已有的 seed 位置，挑
// 距离当前 cell 最近的写入输出。
//
// 27 邻居形式：dz∈{-k,0,+k}, dy同, dx同。出界邻居跳过；空邻居（a=0）跳过。
// 自己也算一个邻居（保持已有最优解）。
//
// 输入 / 输出 ping-pong：上一步写入 gSeedSrc，本步从 gSeedSrc 读、写 gSeedDst；
// 调用方在两步之间交换两张 image 的角色 + 走 layout barrier。

[[vk::binding(0, 0)]] Texture3D<float4>   gSeedSrc;
[[vk::binding(1, 0)]] RWTexture3D<float4> gSeedDst;

struct JfaPC {
    uint resolution;
    int  k;
    uint _pad0, _pad1;
};
[[vk::push_constant]] JfaPC gPC;

[shader("compute")]
[numthreads(4, 4, 4)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    int R = int(gPC.resolution);
    int3 c = int3(dt);
    if (any(c >= R)) return;

    float3 here = float3(c);
    float bestDist2 = 1e30;
    float3 bestPos = float3(0);
    float bestValid = 0.0;

    [unroll]
    for (int dz = -1; dz <= 1; ++dz)
    [unroll]
    for (int dy = -1; dy <= 1; ++dy)
    [unroll]
    for (int dx = -1; dx <= 1; ++dx) {
        int3 nb = c + int3(dx, dy, dz) * gPC.k;
        if (any(nb < 0) || any(nb >= R)) continue;
        float4 s = gSeedSrc.Load(int4(nb, 0));
        if (s.a < 0.5) continue;   // 邻居尚未持有 seed
        float3 d = s.xyz - here;
        float d2 = dot(d, d);
        if (d2 < bestDist2) {
            bestDist2 = d2;
            bestPos = s.xyz;
            bestValid = 1.0;
        }
    }

    gSeedDst[uint3(c)] = float4(bestPos, bestValid);
}
