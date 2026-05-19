// LPV 单次传播 compute —— Kaplanyan 2010 5-face 完整版（B.7 升级）。
// 替换原"front-face only"简化版：每邻居贡献 1 直接面 + 4 侧面，光散布
// 更均匀，扩散更接近 ground-truth radiosity。
//
// 算法：对每个 receiver cell，遍历 6 个邻居（offset O = ±x/±y/±z）。
// 对每个邻居，光从邻居 cell 中心穿过 receiver 的 5 个面进入：
//   - 直接面 (direct)：receiver 与邻居共享的那一面，dir = -O
//   - 4 个侧面 (side)：receiver 上 4 个垂直于 O 轴的面（±perp1, ±perp2），
//     dir = normalize(-O + 0.5·perp_unit·sign)
// 每个面：
//   1. 沿 dir 评估邻居 SH 得到那个方向的 radiance L
//   2. 把 L 当 cosine lobe 沿 dir 投到 receiver SH
//   3. 乘对应 solid angle (Kaplanyan 推导)：直接面 0.4006sr，侧面 0.4234sr
// 6 邻居 × 5 面 = 30 次 SH 操作 / cell / iteration。
//
// 与 front-face-only 版本相比：
//   - 总 solid angle 覆盖：1·0.4006 + 4·0.4234 = 2.094 sr ≈ 2/3 hemisphere，
//     接近完整入射光积分；front-face-only 仅 0.4006 sr，低估 5×。
//   - 光在斜方向也能传，少看到"轴向条纹"瑕疵。
//
// 仍不做：occlusion via Geometry Volume（GV）—— 需要单独 SH 体积存
// scene 几何遮挡，propagate 时按 dir attenuate。是 LPV 防穿墙漏光的
// 关键升级，留给后续。

import sh;

[[vk::binding(0,0)]] Texture3D<float4>   gSrcR;
[[vk::binding(1,0)]] Texture3D<float4>   gSrcG;
[[vk::binding(2,0)]] Texture3D<float4>   gSrcB;
[[vk::binding(3,0)]] RWTexture3D<float4> gDstR;
[[vk::binding(4,0)]] RWTexture3D<float4> gDstG;
[[vk::binding(5,0)]] RWTexture3D<float4> gDstB;
[[vk::binding(6,0)]] Texture3D<float4>   gGv;     // B.8 Geometry Volume

struct PropagatePC {
    uint  gridResolution;
    float occlusionAmplifier;
    float gvOcclusionStrength;   // B.8：GV 遮挡强度，0=禁用 GV，1=full
    uint  _p1;
};
[[vk::push_constant]] PropagatePC gPC;

// 6 个邻居偏移：receiver → neighbor 方向。
static const int3 kOffsets[6] = {
    int3(+1, 0, 0), int3(-1, 0, 0),
    int3( 0,+1, 0), int3( 0,-1, 0),
    int3( 0, 0,+1), int3( 0, 0,-1),
};

// Kaplanyan solid angle ratios（除以 π 让累加保持能量数量级合理）。
static const float kSolidAngleDirect = 0.4006696846 / 3.14159265;
static const float kSolidAngleSide   = 0.4234413544 / 3.14159265;

// 累加一个 face 的贡献。dir 是 neighbor → face center 的方向（已 normalize）。
void accumulateFace(float3 dir, float weight,
                    float4 nR, float4 nG, float4 nB,
                    inout float4 sumR, inout float4 sumG, inout float4 sumB) {
    ShCoeffs shR; shR.c = nR;
    ShCoeffs shG; shG.c = nG;
    ShCoeffs shB; shB.c = nB;
    float lR = evalSh(shR, dir);
    float lG = evalSh(shG, dir);
    float lB = evalSh(shB, dir);
    ShCoeffs lobe = cosineLobeSh(dir);
    float w = weight * gPC.occlusionAmplifier;
    sumR += lobe.c * (lR * w);
    sumG += lobe.c * (lG * w);
    sumB += lobe.c * (lB * w);
}

[shader("compute")]
[numthreads(4, 4, 4)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (any(dt >= uint3(gPC.gridResolution, gPC.gridResolution, gPC.gridResolution))) return;

    float4 sumR = float4(0);
    float4 sumG = float4(0);
    float4 sumB = float4(0);

    [unroll]
    for (int i = 0; i < 6; ++i) {
        int3 nIdx = int3(dt) + kOffsets[i];
        if (any(nIdx < 0) || any(nIdx >= int(gPC.gridResolution))) continue;

        float4 nR = gSrcR.Load(int4(nIdx, 0));
        float4 nG = gSrcG.Load(int4(nIdx, 0));
        float4 nB = gSrcB.Load(int4(nIdx, 0));

        // O = receiver → neighbor，所以 -O = neighbor → receiver（直接方向）
        float3 axis = float3(kOffsets[i]);

        // B.8 GV：在 neighbor cell sample geometry SH，沿光传播方向求遮挡。
        // 光从 neighbor 流向 receiver，方向 = -axis。GV 中存了 cosineLobeSh(-N)，
        // 当 -axis 与 -N 同向（即光打入 surface 正面）→ evalSh 高 → 遮挡强。
        float gvOcc = 0.0;
        if (gPC.gvOcclusionStrength > 0.0) {
            float4 gvN = gGv.Load(int4(nIdx, 0));
            ShCoeffs gvSh; gvSh.c = gvN;
            // evalSh 已 max(0, ...)；放大 + clamp 控制
            float blockage = evalSh(gvSh, -axis);
            gvOcc = saturate(blockage * gPC.gvOcclusionStrength);
        }
        float gvAttenuate = 1.0 - gvOcc;

        // 1. 直接面：dir 直接是 -O
        float3 dirDirect = -axis;
        accumulateFace(dirDirect, kSolidAngleDirect * gvAttenuate, nR, nG, nB, sumR, sumG, sumB);

        // 2. 4 个侧面：在 O 轴的两个垂直方向（perp1, perp2）上分别 ±0.5。
        //    O 是 (±1,0,0) / (0,±1,0) / (0,0,±1)，从 axis 的非零分量推 perp。
        float3 perp1, perp2;
        if (abs(axis.x) > 0.5) {
            perp1 = float3(0, 1, 0);
            perp2 = float3(0, 0, 1);
        } else if (abs(axis.y) > 0.5) {
            perp1 = float3(1, 0, 0);
            perp2 = float3(0, 0, 1);
        } else {
            perp1 = float3(1, 0, 0);
            perp2 = float3(0, 1, 0);
        }

        // 4 侧面方向：-axis ± 0.5·perp（normalize）
        float3 sideDirs[4] = {
            normalize(-axis + 0.5 * perp1),
            normalize(-axis - 0.5 * perp1),
            normalize(-axis + 0.5 * perp2),
            normalize(-axis - 0.5 * perp2),
        };
        [unroll]
        for (int s = 0; s < 4; ++s) {
            accumulateFace(sideDirs[s], kSolidAngleSide * gvAttenuate,
                           nR, nG, nB, sumR, sumG, sumB);
        }
    }

    gDstR[dt] = sumR;
    gDstG[dt] = sumG;
    gDstB[dt] = sumB;
}
