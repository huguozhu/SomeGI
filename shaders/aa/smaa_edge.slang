// SMAA Pass 1 — Edge detection
// Detects luminance edges and stores horizontal/vertical edge strengths in R8G8.

[[vk::binding(0, 0)]] Texture2D gInput;
[[vk::binding(1, 0)]] RWTexture2D<float2> gEdges;

static const float kEdgeThreshold = 0.05;

float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint2 dim;
    gEdges.GetDimensions(dim.x, dim.y);
    if (any(dt.xy >= dim)) return;

    int2 uv = int2(dt.xy);
    int2 uvL = int2(max(uv.x-1, 0), uv.y);
    int2 uvR = int2(min(uv.x+1, dim.x-1), uv.y);
    int2 uvT = int2(uv.x, max(uv.y-1, 0));
    int2 uvB = int2(uv.x, min(uv.y+1, dim.y-1));

    float L   = luminance(gInput[uv].rgb);
    float Ll  = luminance(gInput[uvL].rgb);
    float Lr  = luminance(gInput[uvR].rgb);
    float Lt  = luminance(gInput[uvT].rgb);
    float Lb  = luminance(gInput[uvB].rgb);

    // 3-tap horizontal / vertical gradients
    float gh = abs(Lr - Ll);
    float gv = abs(Lt - Lb);

    float edgeH = (gh > kEdgeThreshold) ? gh : 0.0;
    float edgeV = (gv > kEdgeThreshold) ? gv : 0.0;

    gEdges[uv] = float2(edgeH, edgeV);
}
