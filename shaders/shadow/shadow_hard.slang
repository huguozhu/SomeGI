// shadow_hard.slang — GPU-driven shadow map rendering (vertex + fragment).
//
// GPU-driven vertex shader: reads geometry from SSBOs, transforms to sun light space.
// Fragment shader: depth-only (no color output).
//
// Pipeline conventions (set=0):
//   binding 0: ConstantBuffer<ShadowUbo>     — sunViewProj
//   binding 1: StructuredBuffer<Vertex>      — gVertices
//   binding 2: StructuredBuffer<uint>        — gIndices
//   binding 3: StructuredBuffer<DrawData>    — gDrawData

import shared_types;

struct ShadowUbo {
    float4x4 sunViewProj;
};

[[vk::binding(0, 0)]] ConstantBuffer<ShadowUbo>          gShadow;
[[vk::binding(1, 0)]] StructuredBuffer<Vertex>            gVertices;
[[vk::binding(2, 0)]] StructuredBuffer<uint>              gIndices;
[[vk::binding(3, 0)]] StructuredBuffer<DrawData>          gDrawData;

struct Vertex {
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uv0;
};

struct VsOut {
    float4 svPos : SV_Position;
};

[shader("vertex")]
VsOut vs_main(uint vertexIdx : SV_VertexID, uint drawID : SV_InstanceID) {
    DrawData dd = gDrawData[drawID];
    VsOut o;

    uint idx = gIndices[dd.firstIndex + vertexIdx];
    Vertex v = gVertices[int(idx) + dd.vertexOffset];

    float4 worldPos = mul(dd.modelMatrix, float4(v.position, 1.0));
    o.svPos = mul(gShadow.sunViewProj, worldPos);

    return o;
}

[shader("fragment")]
void ps_main() {
    // depth written automatically via depth attachment
}
