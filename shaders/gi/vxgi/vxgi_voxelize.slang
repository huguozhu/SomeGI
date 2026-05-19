// VXGI 体素化 compute —— 把场景三角形 scatter 到 128³ voxel 网格。
//
// 算法（compute scatter，简化版的保守光栅化）：
//   dispatch (triangleCount / 64, 1, 1)，每 thread 一个三角形。
//   对每个三角形：
//     1. 读 3 顶点（model 变换到 world）。
//     2. 在三角形上采样 7 个点（3 顶点 + 3 边中点 + 重心）。
//     3. 每点 → voxel idx → imageStore(voxelGrid, (albedo, 1))。
//   多个三角形落同 cell：last-writer-wins（简化 —— 每帧重新 voxelize，
//   稳态视觉差异不可见）。
//
// 漏 cell 可能性：三角形比 voxel 大很多时，7 点采样不够覆盖中部。
// 退路：M7+ 改用 GS 真保守光栅。
//
// 关于 albedo 取值：用 material.baseColorFactor × baseColorTex(uv0_重心)，
// 在三角形重心采一次。voxel 表示低频 → 一次取样足够。

import shared_types;

// 与 C++ 端 struct Vertex 严格对齐（pos+normal+tangent+uv0 = 48B）。
struct Vertex {
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uv0;
};

[[vk::binding(0, 0)]] StructuredBuffer<Vertex>      gVertices;
[[vk::binding(1, 0)]] StructuredBuffer<uint>        gIndices;
[[vk::binding(2, 0)]] StructuredBuffer<MaterialGpu> gMaterials;
[[vk::binding(3, 0)]] SamplerState                  gLinearSampler;
[[vk::binding(4, 0)]] Texture2D                     gTextures[];
[[vk::binding(5, 0)]] RWTexture3D<float4>           gVoxelGrid;

struct VoxelizePC {
    float4x4 model;
    uint     firstIndex;       // 索引缓冲偏移
    uint     indexCount;       // 本 primitive 的索引数（=3·triangle 数）
    int      vertexOffset;     // 加到 index 上得到 vertexBuffer 索引
    int      materialIndex;
    float3   gridMin;
    float    cellSize;
    uint     gridResolution;
    uint     _pad0, _pad1, _pad2;
};
[[vk::push_constant]] VoxelizePC gPC;

float4 sampleAlbedo(int matIdx, float2 uv) {
    if (matIdx < 0) return float4(1, 1, 1, 1);
    MaterialGpu m = gMaterials[matIdx];
    float4 base = m.baseColorFactor;
    if (m.baseColorTex >= 0) {
        base *= gTextures[m.baseColorTex].SampleLevel(gLinearSampler, uv, 0);
    }
    return base;
}

void writeVoxel(float3 worldPos, float4 color) {
    float3 rel = (worldPos - gPC.gridMin) / gPC.cellSize;
    int3 idx = int3(floor(rel));
    if (any(idx < int3(0)) || any(idx >= int(gPC.gridResolution))) return;
    gVoxelGrid[idx] = color;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint triIdx = dt.x;
    uint triCount = gPC.indexCount / 3u;
    if (triIdx >= triCount) return;

    // 读 3 个索引 + 顶点。
    uint i0 = gIndices[gPC.firstIndex + triIdx * 3u + 0u];
    uint i1 = gIndices[gPC.firstIndex + triIdx * 3u + 1u];
    uint i2 = gIndices[gPC.firstIndex + triIdx * 3u + 2u];

    Vertex v0 = gVertices[(int)i0 + gPC.vertexOffset];
    Vertex v1 = gVertices[(int)i1 + gPC.vertexOffset];
    Vertex v2 = gVertices[(int)i2 + gPC.vertexOffset];

    float3 p0 = mul(gPC.model, float4(v0.position, 1.0)).xyz;
    float3 p1 = mul(gPC.model, float4(v1.position, 1.0)).xyz;
    float3 p2 = mul(gPC.model, float4(v2.position, 1.0)).xyz;

    // 重心采 albedo（uv0 重心插值）。alpha cutoff 不在这层处理 —— 保守
    // 把 alpha-mask 三角形也 voxelize 进去；视觉上影响小。
    float2 uvCenter = (v0.uv0 + v1.uv0 + v2.uv0) / 3.0;
    float4 albedo = sampleAlbedo(gPC.materialIndex, uvCenter);
    if (albedo.a < 0.01) return;   // 透明三角形跳过

    // 7-point sample：3 顶点 + 3 边中点 + 重心。三角形小于 voxel 时全落
    // 同一 cell；稍大时落进 ~3-4 cell；非常大的三角形可能漏中段（M7+
    // 改 GS 保守光栅可解）。
    float3 mid01 = (p0 + p1) * 0.5;
    float3 mid12 = (p1 + p2) * 0.5;
    float3 mid20 = (p2 + p0) * 0.5;
    float3 cent  = (p0 + p1 + p2) / 3.0;

    float4 color = float4(albedo.rgb, 1.0);
    writeVoxel(p0,    color);
    writeVoxel(p1,    color);
    writeVoxel(p2,    color);
    writeVoxel(mid01, color);
    writeVoxel(mid12, color);
    writeVoxel(mid20, color);
    writeVoxel(cent,  color);
}
