[[vk::binding(0, 0)]] Texture2D gHdr;
[[vk::binding(1, 0)]] SamplerState gLinear;
[[vk::binding(2, 0)]] RWTexture2D<float4> gLdr;

float3 acesApprox(float3 c) {
    c *= 0.6;
    float a = 2.51, b = 0.03, c2 = 2.43, d = 0.59, e = 0.14;
    return saturate((c*(a*c+b)) / (c*(c2*c+d)+e));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint2 dim;
    gLdr.GetDimensions(dim.x, dim.y);
    if (any(dt.xy >= dim)) return;
    float2 uv = (float2(dt.xy) + 0.5) / float2(dim);
    float3 hdr = gHdr.SampleLevel(gLinear, uv, 0).rgb;
    float3 ldr = acesApprox(hdr);
    ldr = pow(saturate(ldr), 1.0/2.2);
    gLdr[dt.xy] = float4(ldr, 1.0);
}
