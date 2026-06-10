// GBuffer Mesh Shader（仅 Mesh + Fragment，无 Task Shader）。
// 用于 taskShader 特性不可用的 GPU。每 draw 一个 mesh work group，
// Mesh Shader 内部不做 culling（由外部 compute cull 处理）。

import shared_types;
import gbuffer_layout;

[[vk::binding(0, 0)]] StructuredBuffer<DrawData> gDrawData;
[[vk::binding(6, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(7, 0)]] StructuredBuffer<float> gVertexData;
[[vk::binding(8, 0)]] StructuredBuffer<uint> gIndexData;
[[vk::binding(9, 0)]] StructuredBuffer<MaterialGpu> gMaterials;
[[vk::binding(10,0)]] SamplerState gLinear;
[[vk::binding(11,0)]] Texture2D gTextures[];

#define VERTEX_FLOATS 14
#define MAX_VERTS 192
#define MAX_TRIS  64

struct VsOut {
    float4 svPos    : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float4 tangent  : TEXCOORD2;
    float2 uv       : TEXCOORD3;
    nointerpolation int matIndex : TEXCOORD4;
};

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void ms_main(
    uint32_t drawIndex : SV_GroupID,
    uint3 tid : SV_GroupThreadID,
    out vertices VsOut verts[MAX_VERTS],
    out indices uint3 tris[MAX_TRIS],
    out primitives GBufferOut prims[MAX_TRIS])
{
    DrawData dd = gDrawData[drawIndex];
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
                float4 tan = float4(gVertexData[vo+6], gVertexData[vo+7], gVertexData[vo+8], gVertexData[vo+9]);
                float2 uv  = float2(gVertexData[vo+10], gVertexData[vo+11]);
                float4 wp = mul(dd.modelMatrix, float4(pos, 1.0));
                uint oi = i * 3 + v;
                verts[oi].svPos = mul(gFrame.viewProj, wp);
                verts[oi].worldPos = wp.xyz;
                verts[oi].normal = normalize(mul(n3, nrm));
                verts[oi].tangent = float4(normalize(mul(n3, tan.xyz)), tan.w);
                verts[oi].uv = uv;
                verts[oi].matIndex = int(dd.materialIndex);
            }
            tris[i] = uint3(i*3+0, i*3+1, i*3+2);
        } else {
            verts[i*3+0].svPos = float4(0,0,0,0); verts[i*3+1].svPos = float4(0,0,0,0); verts[i*3+2].svPos = float4(0,0,0,0);
            tris[i] = uint3(i*3+0, i*3+1, i*3+2);
        }
    }
}

float4 sampleTex(int idx, float2 uv) {
    if (idx < 0) return float4(1, 1, 1, 1);
    return gTextures[idx].Sample(gLinear, uv);
}

[shader("pixel")]
GBufferOut ps_main(VsOut i) {
    MaterialGpu m = gMaterials[i.matIndex];
    float4 baseTex = sampleTex(m.baseColorTex, i.uv);
    float4 base = m.baseColorFactor * baseTex;
    if (m.alphaMode == 1u && base.a < m.alphaCutoff) discard;
    float metallic = m.metallicFactor, roughness = m.roughnessFactor;
    if (m.mrTex >= 0) { float3 t = gTextures[m.mrTex].Sample(gLinear, i.uv).rgb; roughness *= t.g; metallic *= t.b; }
    metallic=saturate(metallic); roughness=saturate(roughness);
    float3 N = normalize(i.normal);
    if (m.normalTex >= 0) { float3 nm=gTextures[m.normalTex].Sample(gLinear,i.uv).rgb*2.0-1.0; nm.xy*=m.normalScale;
        float3 T=normalize(i.tangent.xyz); float3 B=normalize(cross(N,T)*i.tangent.w); N=normalize(T*nm.x+B*nm.y+N*nm.z); }
    float ao=1.0; if (m.occlusionTex>=0){ float t=gTextures[m.occlusionTex].Sample(gLinear,i.uv).r; ao=lerp(1.0,t,m.occlusionStrength); }
    float3 emi=m.emissiveFactor; if (m.emissiveTex>=0) emi*=gTextures[m.emissiveTex].Sample(gLinear,i.uv).rgb;
    return packGBuffer(base.rgb,metallic,N,roughness,emi,ao);
}
