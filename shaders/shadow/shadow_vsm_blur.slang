// shadow_vsm_blur.slang — VSM 2×2 box blur compute shader。
//
// 对 VSM 生成的双矩纹理做 2×2 box blur 滤波，减少方差计算中的噪声。
// 未滤波的 moments 会导致 Chebyshev 测试产生明显的 light bleeding 伪影。
//
// 每输出像素对其自身及右下邻居共 4 个像素取均值。
//
// 管线约定：
//   set=0:
//     binding 0: Texture2D<float2>     — 输入 VSM moments (R32G32_SFLOAT)
//     binding 1: RWTexture2D<float2>   — 输出 blurred moments
//   push constant: VsmBlurPC

struct VsmBlurPC {
    uint2 size;
    float2 invSize;
};
[[vk::push_constant]] VsmBlurPC gPC;

[[vk::binding(0, 0)]] Texture2D<float2>  gInput;
[[vk::binding(1, 0)]] RWTexture2D<float2> gOutput;

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.size.x || dt.y >= gPC.size.y) return;
    uint2 pix = dt.xy;

    // 2×2 box blur：自身 + 右邻 + 下邻 + 右下邻
    // 边缘像素 clamp 到有效范围内避免越界读取
    int2 maxPix = int2(gPC.size) - 1;
    int x0 = pix.x,            y0 = pix.y;
    int x1 = min(pix.x + 1, maxPix.x);
    int y1 = min(pix.y + 1, maxPix.y);
    float2 m00 = gInput.Load(int3(x0, y0, 0)).rg;
    float2 m10 = gInput.Load(int3(x1, y0, 0)).rg;
    float2 m01 = gInput.Load(int3(x0, y1, 0)).rg;
    float2 m11 = gInput.Load(int3(x1, y1, 0)).rg;

    gOutput[pix] = (m00 + m10 + m01 + m11) * 0.25;
}
