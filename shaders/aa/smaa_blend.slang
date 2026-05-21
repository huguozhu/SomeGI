// SMAA Pass 2 — Edge-guided neighborhood blending
// Reads edge texture to determine blend direction and strength,
// then blends 9 neighbors accordingly.

[[vk::binding(0, 0)]] Texture2D gInput;
[[vk::binding(1, 0)]] Texture2D<float2> gEdges;
[[vk::binding(2, 0)]] RWTexture2D<float4> gOutput;

float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint2 dim;
    gOutput.GetDimensions(dim.x, dim.y);
    if (any(dt.xy >= dim)) return;

    int2 uv = int2(dt.xy);
    int2 uvL = int2(max(uv.x-1, 0), uv.y);
    int2 uvR = int2(min(uv.x+1, dim.x-1), uv.y);
    int2 uvT = int2(uv.x, max(uv.y-1, 0));
    int2 uvB = int2(uv.x, min(uv.y+1, dim.y-1));
    int2 uvTL = int2(max(uv.x-1, 0), max(uv.y-1, 0));
    int2 uvTR = int2(min(uv.x+1, dim.x-1), max(uv.y-1, 0));
    int2 uvBL = int2(max(uv.x-1, 0), min(uv.y+1, dim.y-1));
    int2 uvBR = int2(min(uv.x+1, dim.x-1), min(uv.y+1, dim.y-1));

    float3 c  = gInput[uv].rgb;
    float3 l  = gInput[uvL].rgb;
    float3 r  = gInput[uvR].rgb;
    float3 t  = gInput[uvT].rgb;
    float3 b  = gInput[uvB].rgb;

    float2 e = gEdges[uv];
    float eh = e.x;
    float ev = e.y;

    // No edge at this pixel — output unchanged
    if (eh < 0.001 && ev < 0.001) {
        gOutput[uv] = float4(c, 1.0);
        return;
    }

    // Horizontal edge: blend left/right
    // Vertical edge: blend top/bottom
    // Diagonal / both: blend all four
    float wh = eh / (eh + ev + 0.0001);
    float wv = ev / (eh + ev + 0.0001);

    float3 blend = c;
    blend += l * wh * 0.5 + r * wh * 0.5;
    blend += t * wv * 0.5 + b * wv * 0.5;
    blend /= 1.0 + wh + wv;

    gOutput[uv] = float4(blend, 1.0);
}
