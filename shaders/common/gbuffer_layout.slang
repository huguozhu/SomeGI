// shaders/common/gbuffer_layout.slang
//
// GBuffer encode/decode contract — single source of truth for both the
// GBufferPass writer and the LightingPass / SS* passes that read it.
//
//   RT0 (RGBA8_UNORM):    rgb = baseColor (linear), a = metallic
//   RT1 (RGBA16_SFLOAT):  xyz = world normal (normalized), w = roughness
//   RT2 (RGBA8_UNORM):    rgb = emissive (clamped to [0,1] in M4),
//                         a   = ambient occlusion (1 = unoccluded)

struct GBufferOut {
    float4 rt0 : SV_Target0;
    float4 rt1 : SV_Target1;
    float4 rt2 : SV_Target2;
};

GBufferOut packGBuffer(float3 baseColor, float metallic,
                       float3 worldNormal, float roughness,
                       float3 emissive, float ao) {
    GBufferOut o;
    o.rt0 = float4(baseColor, metallic);
    o.rt1 = float4(normalize(worldNormal), roughness);
    o.rt2 = float4(saturate(emissive), ao);
    return o;
}
