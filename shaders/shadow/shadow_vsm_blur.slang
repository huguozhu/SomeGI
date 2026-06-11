// shadow_vsm_blur.slang — VSM 可分離 5-tap gaussian blur compute shader。
//
// 替代原始 2×2 box blur，用水平+垂直两 pass 5-tap gaussian 近似产生
// 更高质量的平滑 moments，进一步减少 light bleeding。
//
// gaussian weights: [1, 4, 6, 4, 1] / 16.0
//
// 管线约定（每 pass 单独绑定输入/输出）：
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

// 每线程组共享内存：处理 8×8 输出像素，需要加载 8×(8+4) 或 (8+4)×8 的输入
// 简化：不使用共享内存，直接用 Load（5 次 texture fetch 仍比采样快）
groupshared float2 gCache[8][12];  // 水平: 8 rows × (8+4) cols；垂直: 8 cols × (8+4) rows

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 gt : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    // ── 计算输出像素坐标 ──
    uint2 outPix = gid.xy * 8 + gt.xy;
    if (outPix.x >= gPC.size.x || outPix.y >= gPC.size.y) return;

    int2 maxPix = int2(gPC.size) - 1;

    if (gPC.dir == 0) {
        // ── 水平 pass ──
        // 预加载共享内存：每个线程加载自己负责的输入像素 + 左侧 2 个额外像素
        int2 base = int2(gid.x * 8, outPix.y);
        for (int x = int(gt.x) - 2; x <= int(gt.x) + 2; x += 8) {
            int sx = clamp(base.x + x, 0, maxPix.x);
            gCache[gt.y][gt.x + 2] = gInput.Load(int3(sx, base.y, 0)).rg;
        }
        GroupMemoryBarrierWithGroupSync();

        // 5-tap gaussian
        float2 result = float2(0, 0);
        for (int t = 0; t < 5; ++t) {
            result += gCache[gt.y][gt.x + t] * kWeight[t];
        }
        gOutput[outPix] = result;
    } else {
        // ── 垂直 pass ──
        // 预加载共享内存：每线程加载自己负责的输入像素 + 上方 2 个额外像素
        int2 base = int2(outPix.x, gid.y * 8);
        for (int y = int(gt.y) - 2; y <= int(gt.y) + 2; y += 8) {
            int sy = clamp(base.y + y, 0, maxPix.y);
            gCache[gt.x][gt.y + 2] = gInput.Load(int3(base.x, sy, 0)).rg;
        }
        GroupMemoryBarrierWithGroupSync();

        // 5-tap gaussian
        float2 result = float2(0, 0);
        for (int t = 0; t < 5; ++t) {
            result += gCache[gt.x][gt.y + t] * kWeight[t];
        }
        gOutput[outPix] = result;
    }
}
