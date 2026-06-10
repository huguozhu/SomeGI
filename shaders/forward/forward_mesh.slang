// Forward Mesh Shader —— Task + Mesh + Fragment。
// 与 gbuffer_mesh 共享 cull 逻辑。Fragment 复用 forward_ibl 计算。
//
// 注意：当前 Slang 版本 Mesh Shader 的输出数由数组大小静态决定。
// 未使用的顶点初始化为退化位置。

import shared_types;
import pbr;
import ibl_eval;
import mesh_common;

// ── set=0: Task + Mesh + Fragment ──────────────────────────────
[[vk::binding(0, 0)]] StructuredBuffer<DrawData> gDrawData;
[[vk::binding(1, 0)]] ConstantBuffer<CullUniforms> gCull;
[[vk::binding(2, 0)]] Texture2D<float> gHiZMip1;
[[vk::binding(3, 0)]] Texture2D<float> gHiZMip2;
[[vk::binding(4, 0)]] Texture2D<float> gHiZMip3;
[[vk::binding(5, 0)]] Texture2D<float> gHiZMip4;
[[vk::binding(6, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(7, 0)]] StructuredBuffer<float> gVertexData;
[[vk::binding(8, 0)]] StructuredBuffer<uint> gIndexData;
[[vk::binding(9, 0)]] StructuredBuffer<MaterialGpu> gMaterials;
[[vk::binding(10,0)]] SamplerState gLinear;
[[vk::binding(11,0)]] Texture2D gTextures[];

// ── set=1: IBL（由 ibl_eval 定义 binding 0-4）───────────────

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

// ── Task Shader ────────────────────────────────────────────────
[shader("task")]
[numthreads(64, 1, 1)]
void ts_main(uint tid : SV_DispatchThreadID) {
    if (tid >= gCull.drawCount) return;
    DrawData dd = gDrawData[tid];
    if (!aabbInFrustum(dd.aabbMin, dd.aabbMax, gCull.frustum)) return;
    if (isOccluded(dd.aabbMin, dd.aabbMax, gCull.viewProj, gCull.screenSize,
                   gCull.hizMaxMip, gHiZMip1, gHiZMip2, gHiZMip3, gHiZMip4)) return;
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

[shader("pixel")]
float4 ps_main(VsOut i) : SV_Target {
    MaterialGpu m = gMaterials[i.matIndex];

    float4 baseTex = sampleTex(m.baseColorTex, i.uv);
    float4 base = m.baseColorFactor * baseTex;
    if (m.alphaMode == 1u && base.a < m.alphaCutoff) discard;

    float metallic = m.metallicFactor;
    float roughness = m.roughnessFactor;
    if (m.mrTex >= 0) {
        float3 t = gTextures[m.mrTex].Sample(gLinear, i.uv).rgb;
        roughness *= t.g;
        metallic  *= t.b;
    }
    metallic = saturate(metallic);
    roughness = saturate(roughness);

    float3 N = normalize(i.normal);
    if (m.normalTex >= 0) {
        float3 nm = gTextures[m.normalTex].Sample(gLinear, i.uv).rgb * 2.0 - 1.0;
        nm.xy *= m.normalScale;
        float3 T = normalize(i.tangent.xyz);
        float3 B = normalize(cross(N, T) * i.tangent.w);
        N = normalize(T*nm.x + B*nm.y + N*nm.z);
    }

    float3 V = normalize(gFrame.cameraPos.xyz - i.worldPos);
    float3 L = normalize(-gFrame.sunDir.xyz);
    float3 sun = directBRDF(N, V, L, base.rgb, metallic, roughness) *
                 gFrame.sunColor_intensity.rgb * gFrame.sunColor_intensity.w;

    float occ = 1.0;
    if (m.occlusionTex >= 0) {
        float t = gTextures[m.occlusionTex].Sample(gLinear, i.uv).r;
        occ = lerp(1.0, t, m.occlusionStrength);
    }

    float3 indirect = evalIBL(N, V, base.rgb, metallic, roughness, (float)gFrame.counts.y);
    indirect *= occ;

    float3 emi = m.emissiveFactor;
    if (m.emissiveTex >= 0) emi *= gTextures[m.emissiveTex].Sample(gLinear, i.uv).rgb;

    return float4(sun + indirect + emi, base.a);
}
