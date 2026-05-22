// GBuffer visibility-buffer variant — writes triangleID + depth only.
// Uses SV_VertexID/3 + push constant primitiveBase to avoid geometryShader dependency.

import shared_types;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] StructuredBuffer<MaterialGpu> gMaterials;
[[vk::binding(2, 0)]] SamplerState gLinear;
[[vk::binding(3, 0)]] Texture2D gTextures[];

struct PushConstsVis {
    float4x4 model;
    int materialIndex;
    uint primitiveBase;  // unique triangle offset for this draw call
    int _p1, _p2;
};
[[vk::push_constant]] PushConstsVis gPC;

struct VsIn {
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float4 tangent : TANGENT;
    float2 uv0     : TEXCOORD0;
};

struct VsOut {
    float4 svPos    : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float4 tangent  : TEXCOORD2;
    float2 uv       : TEXCOORD3;
    nointerpolation int  matIndex   : TEXCOORD4;
    nointerpolation uint triIndex   : TEXCOORD5;  // global triangle index
};

[shader("vertex")]
VsOut vs_main(VsIn i, uint vid : SV_VertexID) {
    VsOut o;
    float4 wp = mul(gPC.model, float4(i.pos, 1.0));
    o.worldPos = wp.xyz;
    o.svPos = mul(gFrame.viewProj, wp);
    float3x3 n3 = (float3x3)gPC.model;
    o.normal  = normalize(mul(n3, i.normal));
    o.tangent = float4(normalize(mul(n3, i.tangent.xyz)), i.tangent.w);
    o.uv = i.uv0;
    o.matIndex = gPC.materialIndex;
    // Primitive index = vertex/3 + base (works for non-indexed or de-duplicated vertex data)
    o.triIndex = vid / 3u + gPC.primitiveBase;
    return o;
}

float4 sampleTex(int idx, float2 uv) {
    if (idx < 0) return float4(1, 1, 1, 1);
    return gTextures[idx].Sample(gLinear, uv);
}

struct VisOut {
    uint2 packed : SV_Target0;  // x=triangleID, y=depth_bits
};

[shader("pixel")]
VisOut ps_main(VsOut i) {
    MaterialGpu m = gMaterials[i.matIndex];

    float4 baseTex = sampleTex(m.baseColorTex, i.uv);
    float4 base = m.baseColorFactor * baseTex;
    if (m.alphaMode == 1u && base.a < m.alphaCutoff) discard;

    // Pack triangleID: materialIndex (12 bits) | triangleIndex (20 bits)
    uint tid = (uint(i.matIndex) << 20u) | (i.triIndex & 0xFFFFFu);

    VisOut o;
    o.packed.x = tid;
    o.packed.y = asuint(i.svPos.z);
    return o;
}
