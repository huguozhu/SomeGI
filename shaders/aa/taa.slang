// TAA — temporal anti-aliasing via reprojection + history blend

[[vk::binding(0, 0)]] Texture2D gCurr;
[[vk::binding(1, 0)]] Texture2D gHistory;
[[vk::binding(2, 0)]] Texture2D<float> gDepth;
[[vk::binding(3, 0)]] RWTexture2D<float4> gOutput;

struct PC {
    float2 jitter;
    float2 prevJitter;
    float4x4 invViewProj;
    float4x4 prevViewProj;
    float blendAlpha;
    float2 invRes;
};
[[vk::push_constant]] PC gPC;

float3 rgb2yuv(float3 c) {
    return float3(dot(c, float3(0.2126, 0.7152, 0.0722)),
                  dot(c, float3(-0.09991, -0.33609, 0.436)),
                  dot(c, float3(0.615, -0.55861, -0.05639)));
}
float3 yuv2rgb(float3 c) {
    return float3(dot(c, float3(1.0, 0.0, 1.13983)),
                  dot(c, float3(1.0, -0.39465, -0.58060)),
                  dot(c, float3(1.0, 2.03211, 0.0)));
}

float4 clipAABB(float3 aabbMin, float3 aabbMax, float4 p, float4 q) {
    float3 pClip = 0.5 * (aabbMax + aabbMin);
    float3 eClip = 0.5 * (aabbMax - aabbMin) + 0.001;
    float4 vClip = q - float4(pClip, q.w);
    float3 u = abs(vClip.xyz / eClip);
    float f = max(u.x, max(u.y, u.z));
    if (f > 1.0) {
        return float4(pClip + vClip.xyz / f, q.w);
    }
    return q;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint2 dim;
    gOutput.GetDimensions(dim.x, dim.y);
    if (any(dt.xy >= dim)) return;

    float2 uv = (float2(dt.xy) + 0.5) * gPC.invRes;
    float depth = gDepth[dt.xy];

    // Reproject to previous frame
    float2 ndc = uv * 2.0 - 1.0 - gPC.jitter;
    float4 clip = mul(gPC.invViewProj, float4(ndc, depth, 1.0));
    float4 prevClip = mul(gPC.prevViewProj, float4(clip.xyz / clip.w, 1.0));
    float2 prevNdc = prevClip.xy / prevClip.w + gPC.prevJitter;
    float2 prevUv = prevNdc * 0.5 + 0.5;

    float3 curr = gCurr[dt.xy].rgb;
    float3 history = float3(0, 0, 0);
    if (all(prevUv >= 0.0) && all(prevUv <= 1.0)) {
        float2 prevPx = prevUv * float2(dim) - 0.5;
        int2 prevCoord = int2(floor(prevPx));
        history = gHistory[prevCoord].rgb;
    }

    // 3x3 neighborhood for color box clamp
    float3 neigh[9];
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            int2 sc = int2(dt.xy) + int2(x, y);
            sc = clamp(sc, int2(0,0), int2(dim) - 1);
            neigh[(y+1)*3 + (x+1)] = gCurr[sc].rgb;
        }
    }
    float3 cMin = neigh[0], cMax = neigh[0];
    for (int i = 1; i < 9; ++i) {
        cMin = min(cMin, neigh[i]);
        cMax = max(cMax, neigh[i]);
    }

    float3 currY = rgb2yuv(curr);
    float3 histY = rgb2yuv(history);
    float3 cMinY = rgb2yuv(cMin);
    float3 cMaxY = rgb2yuv(cMax);

    float4 result = float4(lerp(currY, histY, gPC.blendAlpha), 1.0);
    result = clipAABB(cMinY, cMaxY, float4(currY, 1.0), result);

    gOutput[dt.xy] = float4(yuv2rgb(result.rgb), 1.0);
}
