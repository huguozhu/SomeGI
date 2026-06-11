// shadow_vsm_blur.slang — VSM 5-tap gaussian blur compute shader。
//
// 单 pass 5-tap gaussian：每个输出像素对其 ±2 邻域做加权平均。
// 水平+垂直分开两 pass 由 C++ 端控制（先水平 blurDispatch(dir=0)，后垂直 blurDispatch(dir=1)）。
//
// gaussian weights: [1, 4, 6, 4, 1] / 16.0
//
// 管线约定：
//   set=0:
//     binding 0: Texture2D<float2>     — 输入 moments (R32G32_SFLOAT)
//     binding 1: RWTexture2D<float2>   — 输出 moments
//   push constant: VsmBlurPC { uint2 size; float2 invSize; uint dir; }

struct VsmBlurPC {
    uint2  size;
    float2 invSize;
    uint   dir;       // 0 = 水平, 1 = 垂直
    uint   _pad;
};
[[vk::push_constant]] VsmBlurPC gPC;

[[vk::binding(0, 0)]] Texture2D<float2>   gInput;
[[vk::binding(1, 0)]] RWTexture2D<float2>  gOutput;

// 5-tap gaussian weights: [1, 4, 6, 4, 1] / 16.0
static const float kWeight[5] = {
    1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0
};

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.size.x || dt.y >= gPC.size.y) return;
    uint2 pix = dt.xy;
    int2 maxPix = int2(gPC.size) - 1;

    float2 result = float2(0, 0);

    if (gPC.dir == 0) {
        // ── 水平 5-tap ──
        for (int t = -2; t <= 2; ++t) {
            int sx = clamp(int(pix.x) + t, 0, maxPix.x);
            result += gInput.Load(int3(sx, int(pix.y), 0)).rg * kWeight[t + 2];
        }
    } else {
        // ── 垂直 5-tap ──
        for (int t = -2; t <= 2; ++t) {
            int sy = clamp(int(pix.y) + t, 0, maxPix.y);
            result += gInput.Load(int3(int(pix.x), sy, 0)).rg * kWeight[t + 2];
        }
    }

    gOutput[pix] = result;
}
