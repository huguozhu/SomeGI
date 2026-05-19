// VXGI 多 bounce relight compute —— C.2 Lumen-lite。
//
// 思路：voxel grid 在 inject 阶段只有 1-bounce direct sun radiance。Lumen
// 的核心 trick 是 "iterative relight" —— 每帧把 surface cache 自身做几次
// 全局 cone trace，把"已经有的 radiance"再传一轮，得到 multi-bounce 间接光。
// 我们没有 Lumen 的 mesh card system，但 voxel grid 就是粗 lit-cell-cache，
// 同样适用。
//
// 算法：每 voxel 当 receiver，朝半球 6 个方向 cone-trace voxelGrid 自身
// （已有 mipchain），在 cone 沿途取 radiance × 透射率累积。结果 += 当前
// voxel 的 RGB → 多一轮 bounce。
//
// 6 方向：±X、±Y、±Z 简单半球替代。完整 Lumen 用 cosine-weighted N 方向，
// 但我们不知道 voxel 的 surface normal（voxelize 时未存）→ 用 6 主轴当
// 各向同性近似。
//
// 步数 / cone aperture 沿用 lighting.slang 的 cone trace 数值。
// alpha 用 vxgiAniso 各向异性 alpha 减穿墙。

[[vk::binding(0, 0)]] Texture3D<float4>   gVoxelGridSrc;   // 上一轮 voxel + mipchain（采读）
[[vk::binding(1, 0)]] Texture3D<float4>   gVxgiAnisoSrc;   // 各向异性 alpha
[[vk::binding(2, 0)]] SamplerState        gLinearClamp;
[[vk::binding(3, 0)]] RWTexture3D<float4> gVoxelGridDst;   // 写新一轮（mip 0）

struct RelightPC {
    uint   gridResolution;
    uint   mipLevels;
    float  cellSize;
    float  bounceStrength;   // 反弹增益（默认 1.0）
    float3 gridMin;
    float  _pad0;
};
[[vk::push_constant]] RelightPC gPC;

static const int3 kDirs[6] = {
    int3(1,0,0), int3(-1,0,0),
    int3(0,1,0), int3(0,-1,0),
    int3(0,0,1), int3(0,0,-1),
};

float3 traceCone(float3 origin, float3 dir, float aperture) {
    float3 acc = float3(0);
    float trans = 1.0;
    float dist = gPC.cellSize * 1.5;
    const int kSteps = 12;
    for (int s = 0; s < kSteps; ++s) {
        float3 pos = origin + dir * dist;
        float3 uv = (pos - gPC.gridMin) / (float(gPC.gridResolution) * gPC.cellSize);
        if (any(uv < 0.0) || any(uv > 1.0)) break;
        float diameter = max(gPC.cellSize, 2.0 * aperture * dist);
        float mip = clamp(log2(diameter / gPC.cellSize), 0.0, float(gPC.mipLevels - 1));
        float4 c = gVoxelGridSrc.SampleLevel(gLinearClamp, uv, mip);
        // 各向异性 alpha 防穿墙
        float a = c.a;
        if (mip >= 1.0) {
            float4 ani = gVxgiAnisoSrc.SampleLevel(gLinearClamp, uv, mip);
            float3 wdir = abs(dir);
            float aDir = max(wdir.x * ani.r, max(wdir.y * ani.g, wdir.z * ani.b));
            a = max(a, aDir);
        }
        acc += trans * c.rgb * a;
        trans *= max(0.0, 1.0 - a);
        if (trans < 0.01) break;
        dist += diameter * 0.5;
    }
    return acc;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (any(dt >= uint3(gPC.gridResolution, gPC.gridResolution, gPC.gridResolution))) return;

    // 读自己 cell 的当前 radiance + alpha
    float4 self = gVoxelGridSrc.Load(int4(int3(dt), 0));
    if (self.a < 0.01) {
        // 空 cell —— 不做 relight，原样写出（bounce 之间空 cell 仍是空）
        gVoxelGridDst[dt] = self;
        return;
    }

    // cell 中心 worldPos（用于 cone origin）
    float3 cellCenter = gPC.gridMin + (float3(dt) + 0.5) * gPC.cellSize;

    // 6 方向 cone gather。aperture ~27°（与 lighting cone trace 一致）。
    float3 gathered = float3(0);
    [unroll]
    for (int i = 0; i < 6; ++i) {
        float3 dir = float3(kDirs[i]);
        // 起点沿 dir 推半 cell 防自交
        float3 ori = cellCenter + dir * (gPC.cellSize * 0.75);
        gathered += traceCone(ori, dir, 0.5);
    }
    gathered *= (1.0 / 6.0);   // 6 方向平均

    // 把 indirect 加回当前 cell 的 RGB（保留 alpha 不变）。bounceStrength
    // 让用户能放大效果观察。下一轮 relight 会再读这个新值。
    float3 newRgb = self.rgb + gathered * gPC.bounceStrength;
    gVoxelGridDst[dt] = float4(newRgb, self.a);
}
