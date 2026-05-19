// Fullscreen skybox: samples a TextureCube using a view ray reconstructed
// from invViewProj. Drawn after forward; depth test = LESS_OR_EQUAL with
// SV_Position.z = 1.0 so it only fills pixels where depth was cleared.

struct SkyUbo {
    float4x4 invViewProj;
    float4   cameraPos; // xyz used
};

[[vk::binding(0, 0)]] ConstantBuffer<SkyUbo> gSky;
[[vk::binding(1, 0)]] TextureCube<float4>    gEnv;
[[vk::binding(2, 0)]] SamplerState           gLinear;

struct VsOut {
    float4 svPos : SV_Position;
    float2 ndc   : TEXCOORD0;
};

[shader("vertex")]
VsOut vs_main(uint vid : SV_VertexID) {
    // Fullscreen triangle (covers NDC [-1, 3] in x, [-1, 3] in y; clipped to viewport)
    float2 ndc = float2((vid << 1) & 2, vid & 2) * 2.0 - 1.0;
    VsOut o;
    o.svPos = float4(ndc, 1.0, 1.0);   // depth = 1.0 (far plane, Vulkan [0,1])
    o.ndc   = ndc;
    return o;
}

[shader("pixel")]
float4 ps_main(VsOut i) : SV_Target {
    // Unproject the far-plane pixel back to world, then build a ray from camera.
    float4 wp = mul(gSky.invViewProj, float4(i.ndc, 1.0, 1.0));
    float3 dir = normalize(wp.xyz / wp.w - gSky.cameraPos.xyz);
    float3 c = gEnv.SampleLevel(gLinear, dir, 0).rgb;
    return float4(c, 1.0);
}
