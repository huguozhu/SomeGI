// VXGI 光照注入 compute —— 把 RSM 的 VPL（worldPos, flux）覆写到对应
// voxel cell 的 RGB；alpha（opacity）保留 voxelize 阶段写入的 1.0。
//
// 算法：
//   dispatch (rsmSize/8, rsmSize/8, 1)。每 thread 一个 RSM texel。
//   读 (worldPos, flux) → voxel idx → imageStore RGB = flux（保留 A=1）。
//   注：M5 RSM PS 已经把 flux = albedo · sunColor · sunIntensity · NdotL，
//   是"该 surface 出射到天空的 radiance"。voxelize 阶段写的 RGB 是 albedo
//   （未受光），inject 这里覆盖成 lit_radiance。
//
// 没有 surface 的 voxel（背光面 / 不在 sun frustum 内）保留 voxelize 写的
// albedo + 1.0 —— cone trace 时这些 cell 会贡献 albedo · 0 = 0 间接光，
// 但 opacity=1 仍能挡住光线（视觉效果上：阴影正确）。
//
// last-writer-wins：与 LPV inject 同问题，密集 RSM 区域多 texel 落同 cell
// 时只留最后一次。M7+ 改 atomic add。

[[vk::binding(0,0)]] Texture2D<float4>      gRsmPosition;
[[vk::binding(1,0)]] Texture2D<float4>      gRsmFlux;
[[vk::binding(2,0)]] RWTexture3D<float4>    gVoxelGrid;

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

    float3 wp = gRsmPosition.Load(int3(dt.xy, 0)).xyz;
    float3 fx = gRsmFlux.Load(int3(dt.xy, 0)).rgb;
    if (dot(fx, fx) <= 0.0) return;

    int3 idx = int3(floor((wp - gPC.gridMin) / gPC.cellSize));
    if (any(idx < int3(0)) || any(idx >= int(gPC.gridResolution))) return;

    // 覆写 RGB 为受光 radiance；alpha 保留 voxelize 写的 1.0
    // （RWTexture3D 没有 mask write，需要 read-modify-write）。
    float existing_alpha = gVoxelGrid[idx].a;
    gVoxelGrid[idx] = float4(fx, max(existing_alpha, 1.0));
}
