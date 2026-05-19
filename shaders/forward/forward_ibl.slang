import shared_types;
import pbr;
import ibl_eval;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] StructuredBuffer<MaterialGpu> gMaterials;
[[vk::binding(2, 0)]] SamplerState gLinear;
[[vk::binding(3, 0)]] Texture2D gTextures[];

[[vk::push_constant]] PushConsts gPC;

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
    nointerpolation int matIndex : TEXCOORD4;
};

[shader("vertex")]
VsOut vs_main(VsIn i) {
    VsOut o;
    float4 wp = mul(gPC.model, float4(i.pos, 1.0));
    o.worldPos = wp.xyz;
    o.svPos = mul(gFrame.viewProj, wp);
    float3x3 n3 = (float3x3)gPC.model;
    o.normal  = normalize(mul(n3, i.normal));
    o.tangent = float4(normalize(mul(n3, i.tangent.xyz)), i.tangent.w);
    o.uv = i.uv0;
    o.matIndex = gPC.materialIndex;
    return o;
}

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

    // IBL：specularMipCount 经 gFrame.counts.y 注入（约定 counts: x=matCount, y=specMips）
    float3 ibl = evalIBL(N, V, base.rgb, metallic, roughness,
                         (float)gFrame.counts.y);
    ibl *= occ;

    float3 emi = m.emissiveFactor;
    if (m.emissiveTex >= 0) emi *= gTextures[m.emissiveTex].Sample(gLinear, i.uv).rgb;

    return float4(sun + ibl + emi, base.a);
}
