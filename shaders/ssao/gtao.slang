// GTAO (Ground-Truth Ambient Occlusion，Activision/Jimenez 2016) 简化版。
//
// 核心思路（与 SSAO 的差别）：
// SSAO 在半球内采点，逐点遮挡测试 → 各向同性、对几何方向不敏感、有
// "halo"瑕疵。
// GTAO 转 horizon-based：每像素选 N 条屏幕方向（slice），沿正负方向 march
// depth 找最大水平角 h1/h2；用解析积分把 [h1, h2] 区间内的 cosine kernel
// 算出来 = 1 个 slice 的 AO 真值；多 slice 平均 → 接近 ray-traced AO。
// 比 SSAO 多花约 1.5-2× 算力，但消除 halo + 凹角 / 边沿更准。
//
// 简化点（vs 论文完整版）：
// - 4 slice × 4 sample；论文常用 2 slice × 8 sample
// - 用 1 - mean(h_diff) 近似积分；论文用解析公式 cos kernel·N projection
//   到 slice plane（更准但要 atan2 + 三角函数）。简化版视觉差距小但
//   算量低。
// - 没有 thickness heuristic（论文里防"巨型遮挡过暗"），用 falloff 替代
//
// 输入 / 输出与 SSAO 完全相同（gOutAO R8），可直接互换给 lighting.slang
// 用，不需改 LightingPass binding。

[[vk::binding(0, 0)]] Texture2D<float4> gNormalRough;   // xyz=world normal
[[vk::binding(1, 0)]] Texture2D<float>  gDepth;
[[vk::binding(2, 0)]] RWTexture2D<float> gOutAO;        // R8 输出（与 SSAO 共用 rt.ssao）

struct GtaoPC {
    float4x4 proj;
    float4x4 invProj;
    float4x4 view;
    uint2  outSize;
    float2 invOutSize;
    float  radius;        // 屏幕空间最大 march 距离（pixel 单位）
    float  falloff;       // 远 occluder 衰减距离（view-space 单位）
    uint   sliceCount;    // 多少个 slice（默认 4）
    uint   samplesPerSlice;
};
[[vk::push_constant]] GtaoPC gPC;

float3 viewPosFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 v = mul(gPC.invProj, clip);
    return v.xyz / v.w;
}

uint hash2(uint x, uint y) {
    uint h = x * 73856093u ^ y * 19349663u;
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) return;
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) { gOutAO[pix] = 1.0; return; }

    float3 vP = viewPosFromDepth(pix, depth);
    float3 wN = normalize(gNormalRough.Load(int3(pix, 0)).xyz);
    float3 vN = normalize(mul((float3x3)gPC.view, wN));

    // 视方向（view-space 朝相机 = -vP 的 normalize）
    float3 V = normalize(-vP);

    // 每像素哈希旋转 → 邻像素 slice 方向不同，靠 lighting 端 5×5 bilateral 平滑
    float rotJitter = float(hash2(pix.x, pix.y)) * (1.0 / 4294967296.0);

    float aoSum = 0.0;
    for (uint s = 0; s < gPC.sliceCount; ++s) {
        // slice 方向角（屏幕空间），半圆 + 抖动
        float phi = 3.14159265 * (float(s) / float(gPC.sliceCount) + rotJitter);
        float2 sliceDir2D = float2(cos(phi), sin(phi));

        // 沿 +sliceDir 与 -sliceDir 找 horizon（最大水平角的 cos）。
        // h1 / h2 越接近 1 = 没遮挡（horizon 在天顶）。越接近 -1 = 完全
        // 被遮挡（horizon 在脚下）。初始 -1 让任何 sample 都能更新。
        float cosH1 = -1.0;
        float cosH2 = -1.0;

        for (uint i = 1; i <= gPC.samplesPerSlice; ++i) {
            float t = float(i) / float(gPC.samplesPerSlice);
            float pixelOffset = t * gPC.radius;

            // +dir
            int2 p1 = int2(pix) + int2(sliceDir2D * pixelOffset);
            if (all(p1 >= int2(0)) && all(p1 < int2(gPC.outSize))) {
                float d1 = gDepth.Load(int3(p1, 0));
                if (d1 < 1.0) {
                    float3 vS = viewPosFromDepth(uint2(p1), d1);
                    float3 D = vS - vP;
                    float lenD = length(D);
                    if (lenD > 1e-3) {
                        float cosT = dot(D / lenD, V);
                        // falloff：远 occluder 减权
                        float fall = saturate(1.0 - lenD / gPC.falloff);
                        cosT = lerp(-1.0, cosT, fall);
                        cosH1 = max(cosH1, cosT);
                    }
                }
            }

            // -dir
            int2 p2 = int2(pix) - int2(sliceDir2D * pixelOffset);
            if (all(p2 >= int2(0)) && all(p2 < int2(gPC.outSize))) {
                float d2 = gDepth.Load(int3(p2, 0));
                if (d2 < 1.0) {
                    float3 vS = viewPosFromDepth(uint2(p2), d2);
                    float3 D = vS - vP;
                    float lenD = length(D);
                    if (lenD > 1e-3) {
                        float cosT = dot(D / lenD, V);
                        float fall = saturate(1.0 - lenD / gPC.falloff);
                        cosT = lerp(-1.0, cosT, fall);
                        cosH2 = max(cosH2, cosT);
                    }
                }
            }
        }

        // 把 cosH1/cosH2 转成角度，求 horizon angles。
        float h1 = acos(clamp(cosH1, -1.0, 1.0));
        float h2 = acos(clamp(cosH2, -1.0, 1.0));

        // slice plane 内 N 的投影角（view-space N 投到 (V, sliceDir) 平面）。
        // 简化：取 N·V 当 z 分量，N·sliceDir3D 当切向分量；其中
        // sliceDir3D 是 sliceDir2D 在 view-space 平面上的方向（这里用
        // 屏幕空间近似 → 在 view-space 与 V 垂直的方向，即 (sliceDir2D, 0)）。
        float3 sliceDir3D = normalize(float3(sliceDir2D, 0.0));
        float nDotS = dot(vN, sliceDir3D);
        float nDotV = dot(vN, V);
        float nAngle = atan2(nDotS, nDotV);

        // 把 horizon 角相对 N 的 elevation 角夹一下，再积分 cos kernel。
        // 论文公式：ao_slice = (cos(2·h1 - n) + 2·h1·sin(n) - cos(n))/4
        //                     + (cos(2·h2 - n) + 2·h2·sin(n) - cos(n))/4
        // h1 取 +dir 的 acos(cosH1)，h2 取 -dir 的 -acos(cosH2)（+/- 对称
        // 处理后符号反过来）。
        float h1eff = -h1;       // +dir：相对水平向上方向取负
        float h2eff =  h2;       // -dir：取正
        // 把 h_eff 夹到 N 的 ±π/2 范围内
        h1eff = nAngle + max(h1eff - nAngle, -1.5707963);
        h2eff = nAngle + min(h2eff - nAngle,  1.5707963);

        float aoSlice = 0.25 * (
            (-cos(2.0 * h1eff - nAngle) + cos(nAngle) + 2.0 * h1eff * sin(nAngle)) +
            (-cos(2.0 * h2eff - nAngle) + cos(nAngle) + 2.0 * h2eff * sin(nAngle))
        );

        // length(N projected to slice plane) 当 slice 权重 —— N 跟 slice
        // 方向越正交贡献越小（论文 Equation）。
        float weight = sqrt(nDotS * nDotS + nDotV * nDotV);
        aoSum += aoSlice * weight;
    }

    // 平均到 [0,1]：每 slice 满 occlusion = π，sliceCount 个 slice 总
    // 满 occlusion = π·sliceCount，归一化乘 1/π。
    float ao = aoSum / float(gPC.sliceCount) / 3.14159265;
    ao = saturate(ao);
    gOutAO[pix] = ao;
}
