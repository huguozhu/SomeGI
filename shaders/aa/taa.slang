// TAA — temporal anti-aliasing via reprojection + history blend
// Enhancements: disocclusion detection + variance-based color clamping

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

// YUV conversion for perceptual color space clamping
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

// Variance-based clip: clamp history to [mean - gamma*sigma, mean + gamma*sigma]
float3 clipVariance(float3 history, float3 neigh[9], float gamma) {
    float3 mean = float3(0, 0, 0);
    float3 moment2 = float3(0, 0, 0);
    for (int i = 0; i < 9; ++i) {
        mean += neigh[i];
        moment2 += neigh[i] * neigh[i];
    }
    mean /= 9.0;
    moment2 /= 9.0;
    float3 sigma = sqrt(max(float3(0, 0, 0), moment2 - mean * mean));

    float3 cMin = mean - gamma * sigma;
    float3 cMax = mean + gamma * sigma;
    return clamp(history, cMin, cMax);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint2 dim;
    gOutput.GetDimensions(dim.x, dim.y);
    if (any(dt.xy >= dim)) return;

    float2 uv = (float2(dt.xy) + 0.5) * gPC.invRes;
    float depth = gDepth[dt.xy];

    // Sky / no geometry — no history to blend
    if (depth >= 1.0) {
        gOutput[dt.xy] = float4(gCurr[dt.xy].rgb, 1.0);
        return;
    }

    // Reproject to previous frame
    float2 ndc = uv * 2.0 - 1.0 - gPC.jitter;
    float4 clip = mul(gPC.invViewProj, float4(ndc, depth, 1.0));
    float4 prevClip = mul(gPC.prevViewProj, float4(clip.xyz / clip.w, 1.0));
    float2 prevNdc = prevClip.xy / prevClip.w + gPC.prevJitter;
    float2 prevUv = prevNdc * 0.5 + 0.5;

    float3 curr = gCurr[dt.xy].rgb;
    float3 history = float3(0, 0, 0);
    float blendWeight = gPC.blendAlpha;

    if (all(prevUv >= 0.0) && all(prevUv <= 1.0)) {
        float2 prevPx = prevUv * float2(dim) - 0.5;
        int2 prevCoord = int2(floor(prevPx));

        // Disocclusion detection: compare depth at reprojected UV
        // Large depth difference = likely disocclusion → reduce blend
        float depthPrev = gDepth[prevCoord];
        float depthDiff = abs(depth - depthPrev) / max(depth, 1e-5);
        if (depthDiff > 0.02) {
            blendWeight = min(blendWeight, 0.15); // mostly current frame
        }

        history = gHistory[prevCoord].rgb;
    } else {
        // Off-screen reprojection → no valid history
        blendWeight = 0.0;
    }

    // 3x3 neighborhood for variance clamping
    float3 neigh[9];
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            int2 sc = int2(dt.xy) + int2(x, y);
            sc = clamp(sc, int2(0,0), int2(dim) - 1);
            neigh[(y+1)*3 + (x+1)] = gCurr[sc].rgb;
        }
    }

    // Convert to YUV for perceptual clamping
    float3 currY = rgb2yuv(curr);
    float3 histY = rgb2yuv(history);
    float3 neighY[9];
    for (int i = 0; i < 9; ++i) neighY[i] = rgb2yuv(neigh[i]);

    // Variance-based clamp history to neighborhood
    histY = clipVariance(histY, neighY, 1.5);

    // Blend
    float3 resultY = lerp(currY, histY, blendWeight);
    float3 result = yuv2rgb(resultY);

    gOutput[dt.xy] = float4(result, 1.0);
}
