// RSM Geometry Mesh Shader —— Task + Mesh + Fragment。
// RSM 不做 cull（sun 视角需要全量几何），Task Shader 直通所有 draw。

import shared_types;
import mesh_common;

// ── set=0 bindings ─────────────────────────────────────────────
[[vk::binding(0, 0)]] StructuredBuffer<DrawData> gDrawData;
[[vk::binding(1, 0)]] ConstantBuffer<CullUniforms> gCull;  // 仅用 drawCount
[[vk::binding(6, 0)]] ConstantBuffer<RsmFrameUniforms> gRsm;
[[vk::binding(7, 0)]] StructuredBuffer<float> gVertexData;
[[vk::binding(8, 0)]] StructuredBuffer<uint> gIndexData;
[[vk::binding(9, 0)]] StructuredBuffer<MaterialGpu> gMaterials;
[[vk::binding(10,0)]] SamplerState gLinear;
[[vk::binding(11,0)]] Texture2D gTextures[];

#define VERTEX_FLOATS 12  // sizeof(Vertex)=48 bytes, 12 floats packed
#define MAX_VERTS 192
#define MAX_TRIS  64

struct VsOut {
    float4 svPos    : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float2 uv       : TEXCOORD2;
    nointerpolation int matIndex : TEXCOORD3;
};

// ── Task Shader（直通，不做 cull）──────────────────────────────
[shader("task")]
[numthreads(64, 1, 1)]
void ts_main(uint tid : SV_DispatchThreadID) {
    if (tid >= gCull.drawCount) return;
    TaskPayload pld = { tid };
    DispatchMesh(1, 1, 1, pld);
}

// ── Mesh Shader ────────────────────────────────────────────────
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void ms_main(
    uint3 tid : SV_GroupThreadID,
    in payload TaskPayload pld,
    out vertices VsOut verts[MAX_VERTS],
    out indices uint3 tris[MAX_TRIS],
    out primitives float4 prims[MAX_TRIS])
{
    DrawData dd = gDrawData[pld.drawIndex];
    uint triCount = dd.indexCount / 3;
    float3x3 n3 = (float3x3)dd.modelMatrix;

    for (uint i = tid.x; i < MAX_TRIS; i += 64) {
        if (i < triCount) {
            uint3 idx;
            idx.x = gIndexData[dd.firstIndex + i * 3 + 0];
            idx.y = gIndexData[dd.firstIndex + i * 3 + 1];
            idx.z = gIndexData[dd.firstIndex + i * 3 + 2];

            for (uint v = 0; v < 3; ++v) {
                uint vi = (v == 0) ? idx.x : ((v == 1) ? idx.y : idx.z);
                uint vo = (dd.vertexOffset + vi) * VERTEX_FLOATS;
                float3 pos = float3(gVertexData[vo+0], gVertexData[vo+1], gVertexData[vo+2]);
                float3 nrm = float3(gVertexData[vo+3], gVertexData[vo+4], gVertexData[vo+5]);
                float2 uv  = float2(gVertexData[vo+10], gVertexData[vo+11]);
                float4 wp = mul(dd.modelMatrix, float4(pos, 1.0));
                uint oi = i * 3 + v;
                verts[oi].svPos = mul(gRsm.sunViewProj, wp);
                verts[oi].worldPos = wp.xyz;
                verts[oi].normal = normalize(mul(n3, nrm));
                verts[oi].uv = uv;
                verts[oi].matIndex = int(dd.materialIndex);
            }
            tris[i] = uint3(i*3+0, i*3+1, i*3+2);
        } else {
            verts[i*3+0].svPos = float4(0,0,0,0);
            verts[i*3+1].svPos = float4(0,0,0,0);
            verts[i*3+2].svPos = float4(0,0,0,0);
            tris[i] = uint3(i*3+0, i*3+1, i*3+2);
        }
    }
}

// ── Fragment Shader ────────────────────────────────────────────
float4 sampleTex(int idx, float2 uv) {
    if (idx < 0) return float4(1, 1, 1, 1);
    return gTextures[idx].Sample(gLinear, uv);
}

struct RsmOut {
    float4 rt0 : SV_Target0;
    float4 rt1 : SV_Target1;
    float4 rt2 : SV_Target2;
};

[shader("pixel")]
RsmOut ps_main(VsOut i) {
    MaterialGpu m = gMaterials[i.matIndex];

    float4 baseTex = sampleTex(m.baseColorTex, i.uv);
    float4 base = m.baseColorFactor * baseTex;
    if (m.alphaMode == 1u && base.a < m.alphaCutoff) discard;

    float3 N = normalize(i.normal);
    float3 L = normalize(-gRsm.sunDir.xyz);
    float NdotL = max(0.0, dot(N, L));
    float3 flux = base.rgb * gRsm.sunColor_intensity.rgb * gRsm.sunColor_intensity.w * NdotL;

    RsmOut o;
    o.rt0 = float4(i.worldPos, 1.0);
    o.rt1 = float4(N, 0.0);
    o.rt2 = float4(flux, 1.0);
    return o;
}
