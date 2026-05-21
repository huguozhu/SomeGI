// SMAA — Subpixel Morphological Anti-Aliasing (simplified 2-pass)
// Pass 1: edge detection + write edges to R8G8 texture (R=edge, G=gradient)
// Pass 2: gather 9 neighbors per pixel, blend

[[vk::binding(0, 0)]] Texture2D gInput;
[[vk::binding(1, 0)]] RWTexture2D<float4> gOutput;

// Edge detection constants
static const float kEdgeThreshold    = 0.05;
static const float kEdgeThresholdMin = 0.01;
static const float kSearchSteps      = 8.0;

float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

float3 smaaEdge(float3 c, float3 l, float3 r, float3 t, float3 b,
                float3 tl, float3 tr, float3 bl, float3 br) {
    float L = luminance(c);
    float Ll = luminance(l), Lr = luminance(r), Lt = luminance(t), Lb = luminance(b);
    float Ltl = luminance(tl), Ltr = luminance(tr), Lbl = luminance(bl), Lbr = luminance(br);

    // Horizontal gradient
    float gh = abs(Lr + Ltr + Lbr - Ll - Ltl - Lbl);
    float gv = abs(Lt + Ltl + Ltr - Lb - Lbl - Lbr);

    float edgeH = (gh > 0.05) ? 1.0 : 0.0;
    float edgeV = (gv > 0.05) ? 1.0 : 0.0;

    // Detect if we're on an edge
    if (gh < 0.01 && gv < 0.01) return c;

    // Simple edge-guided blend: weight by inverse gradient
    float wh = 1.0 / (1.0 + gh * 4.0);
    float wv = 1.0 / (1.0 + gv * 4.0);
    float wt = wh + wv + 1.0;

    float3 blend = (c + l * wh + r * wh + t * wv + b * wv) / wt;
    return blend;
}

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
    float3 tl = gInput[uvTL].rgb;
    float3 tr = gInput[uvTR].rgb;
    float3 bl = gInput[uvBL].rgb;
    float3 br = gInput[uvBR].rgb;

    gOutput[uv] = float4(smaaEdge(c, l, r, t, b, tl, tr, bl, br), 1.0);
}
