// Deferred lighting compute —— M4 渲染管线的核心 compute 阶段。
//
// 输入：GBuffer（albedo+metallic / normal+roughness / emissive+AO）+ depth
//       + 三个屏幕空间间接光 buffer（SSAO / SSR / SSGI）+ IBL set（set=1）。
// 输出：hdrColor。
// 算法：per-pixel 重建 worldPos / N → 算 direct sun（PBR BRDF）+ indirect
//       （IBL diffuse / specular，SSGI / SSR 分别 lerp 进 diffuse / specular）
//       + emissive。
// sky 像素（depth==1）输出 (0,0,0,1)，留给 SkyboxPass 在下一阶段从 envCube
// 用 view ray 反查填进去。
//
// counts.z 编码间接光开关：0 = None（hemispheric ambient fallback），
//                          1 = IBL（含 SSGI / RSM 的 lerp 叠加；它们各自
//                              通过 counts.w / 输入纹理的 alpha gate 自己
//                              是否参与）。
// counts.w 编码 RSM 启用：0 = 不混 RSM，1 = 把 RSM 一次反弹按 rsmGI.a
//                         lerp 进 diffuse。当 RSM 关时 rsmGI 已被清成
//                         全 0，但仍走 SHADER_READ_ONLY 让描述符合法。
// vxgiCounts.y 编码 VXGI 启用：1 时 receiver 像素从 worldPos+N·offset
//                              出发沿 6 个 cone 方向 march voxel mip
//                              chain，opacity-weighted 累积 radiance。

import shared_types;
import pbr;
import sh;              // M6 LPV：一阶 SH cosineLobeSh / evalSh
import octahedral;      // M11 DDGI：dir ↔ oct UV
import ibl_eval;        // set=1 IBL bindings + evalIBLDiffuse / Specular / applyIBL*BRDF
import gi.ndgi.ndgi_infer; // NDGI: neural irradiance cache

// set=0：GBuffer + depth + 屏幕空间 buffer + 输出 hdrColor。
[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] Texture2D<float4> gAlbedoMetal;
[[vk::binding(2, 0)]] Texture2D<float4> gNormalRough;
[[vk::binding(3, 0)]] Texture2D<float4> gEmissiveAO;
[[vk::binding(4, 0)]] Texture2D<float>  gDepth;
[[vk::binding(5, 0)]] RWTexture2D<float4> gOutHdr;
[[vk::binding(6, 0)]] Texture2D<float>  gSsao;     // R8, 1=完全无遮蔽
[[vk::binding(7, 0)]] Texture2D<float4> gSsr;      // RGBA16F, rgb=反射色, a=fade
[[vk::binding(8, 0)]] Texture2D<float4> gSsgi;     // RGBA16F, rgb=半球平均 radiance, a=hit 比例
[[vk::binding(9, 0)]] Texture2D<float4> gRsmGI;    // RGBA16F, rgb=RSM gather radiance, a=覆盖置信度
[[vk::binding(10, 0)]] Texture3D<float4> gLpvR;    // M6 LPV：R 通道 SH 系数
[[vk::binding(11, 0)]] Texture3D<float4> gLpvG;
[[vk::binding(12, 0)]] Texture3D<float4> gLpvB;
[[vk::binding(13, 0)]] SamplerState     gLpvSampler;   // trilinear clamp，给 LPV 3D 采样用
[[vk::binding(14, 0)]] Texture3D<float4> gVoxelGrid;   // M7 VXGI：voxel 网格 + mipchain（用 gLpvSampler）
[[vk::binding(15, 0)]] Texture3D<float4> gPrtTransfer; // M8 PRT：visibility transfer SH4
[[vk::binding(16, 0)]] Texture2D<float4> gDdgiIrr;     // M11 DDGI：irradiance octahedral atlas
[[vk::binding(17, 0)]] Texture2D<float2> gDdgiDist;    // M11 DDGI：distance/distance² atlas
[[vk::binding(18, 0)]] StructuredBuffer<uint> gDdgiProbeStates;  // B.5：1=active, 0=inactive (probe inside wall)
[[vk::binding(19, 0)]] Texture3D<float4> gVxgiAniso;             // B.6：rgb = X/Y/Z 方向 Beer-composite alpha
[[vk::binding(20, 0)]] Texture3D<float4> gPrtTransferB;          // B.9 PRT SH9：coefs 4..7 (Y2-2/Y2-1/Y20/Y21)
[[vk::binding(21, 0)]] Texture3D<float4> gPrtTransferC;          // B.9 PRT SH9：coef 8 (.x = Y22)
[[vk::binding(22, 0)]] Texture3D<float4> gPrtTransferD;          // B.10 PRT SH16：coefs 9..12
[[vk::binding(23, 0)]] Texture3D<float4> gPrtTransferE;          // B.10 PRT SH16：coefs 13..15 (rgb)
[[vk::binding(24, 0)]] Texture2D<float4> gRestir;                // C.4 ReSTIR DI：rgb=直接光累加，a=valid
[[vk::binding(25, 0)]] Texture2D<float4> gRtGI;                  // M9 RT GI：rgb=入射 radiance，a=hit?1:0
[[vk::binding(26, 0)]] Texture2D<float4> gLumenGI;              // L.5 Lumen-lite gather 输出
// NDGI: MLP 权重 bindings 27-32 由 import ndgi_infer 提供
[[vk::binding(33, 0)]] Texture2D<float> gShadowMask;   // R8_UNORM: 1=lit, 0=shadow

struct LightingPC {
    uint2  outSize;
    float2 invOutSize;
};
[[vk::push_constant]] LightingPC gPC;

// 用 depth 反推世界坐标：pix → ndc.xy → 加上 depth → invViewProj → world。
// 这是 deferred shading 的标准操作（避免在 GBuffer 里多存一张 worldPos）。
// 依赖 invViewProj 在 FrameUBO 里（每帧 CPU 端 glm::inverse(viewProj) 算）。
float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 w = mul(gFrame.invViewProj, clip);
    return w.xyz / w.w;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) return;
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) {
        // No geometry; SkyboxPass will fill this pixel.
        gOutHdr[pix] = float4(0, 0, 0, 1);
        return;
    }

    float4 a = gAlbedoMetal.Load(int3(pix, 0));
    float4 n = gNormalRough.Load(int3(pix, 0));
    float4 e = gEmissiveAO.Load(int3(pix, 0));

    float3 albedo    = a.rgb;
    float  metallic  = a.a;
    float3 N         = normalize(n.xyz);
    float  roughness = n.w;
    float3 emissive  = e.rgb;
    float  aoMat     = e.a;

    // 5×5 depth-bilateral box 过滤，应用到三张噪点屏幕空间 buffer
    // （SSAO / SSR / SSGI）。一次扫描三组累加器共享同一邻居权重，
    // 因此仅多 ~75 次 texel load。depth 权重拒绝跨 silhouette 的邻居
    // （避免 AO/GI 越过几何边沿漏光）；spatial 权重是小高斯，让中心
    // 主导边沿 taper。25 加权样本平均 → 噪声标准差降到约 1/5。
    //
    // 不影响其它 shader 资源（SSAO 仍是 16 sample / SSGI 8 sample），
    // 只在 lighting 里"统一处理"。SSR 在低粗糙度上会被过滤糊化，用户
    // 可以调小 SSR rough threshold 让镜面材质保持锐利。
    float  aoSum   = 0.0;
    float4 ssrSum  = float4(0);
    float4 ssgiSum = float4(0);
    float4 rsmSum  = float4(0);
    float  wSum    = 0.0;
    const int kBlurR = 2;   // 5x5
    for (int dy = -kBlurR; dy <= kBlurR; ++dy) {
        for (int dx = -kBlurR; dx <= kBlurR; ++dx) {
            int2 sp = clamp(int2(pix) + int2(dx, dy),
                            int2(0), int2(gPC.outSize) - 1);
            float sd = gDepth.Load(int3(sp, 0));
            // sky pixels (depth=1) carry no useful screen-space data
            if (sd >= 1.0) continue;

            float depthDiff = abs(sd - depth);
            float wDepth = 1.0 / (1.0 + depthDiff * 500.0);
            float r2 = float(dx*dx + dy*dy);
            float wSpace = exp(-r2 / 4.0);
            float w = wDepth * wSpace;

            aoSum   += gSsao.Load(int3(sp, 0)) * w;
            ssrSum  += gSsr.Load(int3(sp, 0))  * w;
            ssgiSum += gSsgi.Load(int3(sp, 0)) * w;
            rsmSum  += gRsmGI.Load(int3(sp, 0)) * w;
            wSum += w;
        }
    }
    float invW = (wSum > 0.0) ? 1.0 / wSum : 0.0;
    float  aoSS  = aoSum   * invW;
    float4 ssrF  = ssrSum  * invW;
    float4 ssgiF = ssgiSum * invW;
    float4 rsmF  = rsmSum  * invW;
    // Combine material AO from the GBuffer with the screen-space AO; the
    // screen-space term darkens crevices the material map can't know about.
    float  ao    = min(aoMat, aoSS);

    float3 worldPos = worldFromDepth(pix, depth);
    float3 V = normalize(gFrame.cameraPos.xyz - worldPos);

    float3 L = normalize(-gFrame.sunDir.xyz);
    float3 sun = directBRDF(N, V, L, albedo, metallic, roughness) *
                 gFrame.sunColor_intensity.rgb * gFrame.sunColor_intensity.w;

    // 阴影遮罩（R8_UNORM: 1=lit, 0=shadow）
    float shadow = gShadowMask.Load(int3(pix, 0)).x;
    sun *= shadow;

    // 间接光分两条路径：
    //   counts.z = 1（IBL / SSGI）：走 IBL 拆分式 + SSR/SSGI 混合，是 M4 主路径。
    //   counts.z = 0（None）：     hemispheric ambient fallback，给纯金属 / 纯
    //                            镜面也至少有一点方向感（Y 方向 sky/ground 渐变）。
    float3 indirect;
    if (gFrame.counts.z != 0) {
        // L.5 Lumen-lite：gather pass 已输出完整 diffuse indirect（含 BRDF），
        // 跳过 SSGI/VXGI/DDGI/RSM 等混合链。
        if (gFrame.lumenCounts.x != 0) {
            float4 lumenF = gLumenGI.Load(int3(pix, 0));
            float3 diffuse = lumenF.rgb;
            float3 iblSpec = evalIBLSpecular(N, V, albedo, metallic, roughness,
                                              (float)gFrame.counts.y);
            float3 specular = iblSpec;
            if (ssrF.a > 0.0) {
                float3 ssrSpec = applyIBLSpecularBRDF(ssrF.rgb, N, V, albedo, metallic, roughness);
                specular = lerp(specular, ssrSpec, ssrF.a);
            }
            indirect = diffuse + specular;
            indirect *= aoMat;
            gOutHdr[pix] = float4(sun + indirect + emissive, 1);
            return;
        }

        // Diffuse 项 = lerp(IBL 漫反射, SSGI 漫反射, hit 比例)。SSGI 命中越多
        // 越偏向 SSGI 真值；命中少（rays 跑出屏幕） → 回落 IBL diffuse。
        // ssgiF 是 5×5 bilateral 过滤后的 SSGI 输出。
        float3 iblDiff  = evalIBLDiffuse(N, V, albedo, metallic, roughness);
        float3 ssgiDiff = applyIBLDiffuseBRDF(ssgiF.rgb, N, V, albedo, metallic, roughness);
        float3 diffuse  = lerp(iblDiff, ssgiDiff, ssgiF.a);

        // RSM 一次反弹（M5）：rsmF.rgb 是来自 sun 的 incoming radiance
        // gather（未上 BRDF），需要 × albedo·(1-metallic) 才是 reflected
        // diffuse；用 rsmF.a (sun frustum 覆盖置信度) 当 lerp 权重，覆盖
        // 不到的像素 fallback 到 IBL diffuse。counts.w 是 RSM 启用闸门：
        // 关闭时 rsmF.a 已为 0（rsmGI 被 App clear），但保留闸门让 None
        // 模式 / IBL 模式时即便上一帧残留也不会污染。
        if (gFrame.counts.w != 0) {
            float3 rsmDiff = rsmF.rgb * albedo * (1.0 - metallic);
            diffuse = lerp(diffuse, rsmDiff, rsmF.a);
        }

        // M7 VXGI：从 worldPos 出发沿 6 个 cone 方向 march voxel mipchain，
        // opacity-weighted 累积 radiance；自然带遮挡（cone 撞到 alpha=1
        // 实心 cell 时透射变 0，光被挡住）。比 LPV 慢但视觉上更接近真值。
        // 6 cone 方向是 N 半球的离散 cluster：N 主轴 + 5 个倾斜方向。
        // 每条 cone 沿 dir march，sample mip level = log2(coneDiameter/voxelSize)
        // 让 cone 越远 → 采样越粗的 mip。
        if (gFrame.vxgiCounts.y != 0) {
            float3 vMin    = gFrame.vxgiGridMinCell.xyz;
            float  vCell   = gFrame.vxgiGridMinCell.w;
            float  vGridDim = float(gFrame.vxgiCounts.x) * vCell;
            int    vMipMax  = gFrame.vxgiCounts.z;

            // 6 cone 方向（N 半球的 cosine-weighted cluster）：1 主 + 5 偏。
            const int kCones = 6;
            float3 coneDirsT[kCones] = {
                float3(0, 0, 1),
                float3(0.866, 0, 0.5),
                float3(-0.866, 0, 0.5),
                float3(0, 0.866, 0.5),
                float3(0, -0.866, 0.5),
                float3(0.5, 0.5, 0.707),
            };
            float coneWeights[kCones] = { 0.25, 0.15, 0.15, 0.15, 0.15, 0.15 };

            // 法线切空间。
            float3 up = abs(N.y) < 0.999 ? float3(0,1,0) : float3(1,0,0);
            float3 T = normalize(cross(up, N));
            float3 B = cross(N, T);

            float coneAperture = 0.5;   // tan(half-angle)，约 27°
            float3 vxgiAccum = float3(0);
            float vxgiCov = 0.0;
            for (int ci = 0; ci < kCones; ++ci) {
                float3 dT = normalize(coneDirsT[ci]);
                float3 dW = T * dT.x + B * dT.y + N * dT.z;

                // origin 沿 N 推 1 cell 防自交。
                float3 ori = worldPos + N * (vCell * 1.5);

                float3 acc = float3(0);
                float trans = 1.0;   // transmittance, Beer's law
                float dist = vCell;  // 起步从 1 cell 开始
                const int kSteps = 24;
                for (int s = 0; s < kSteps; ++s) {
                    float3 samplePos = ori + dW * dist;
                    float3 uv = (samplePos - vMin) / vGridDim;
                    if (any(uv < 0.0) || any(uv > 1.0)) break;

                    // mip level = log2(cone diameter / voxelSize)
                    float diameter = max(vCell, 2.0 * coneAperture * dist);
                    float mip = clamp(log2(diameter / vCell), 0.0, float(vMipMax - 1));

                    float4 c = gVoxelGrid.SampleLevel(gLpvSampler, uv, mip);
                    // B.6 各向异性 alpha：mip ≥ 1 时用方向加权 max(|dir|·anisoRGB)
                    // 替代各向同性 c.a，避免穿墙泄漏。mip 0 没 aniso 数据
                    // （aniso pass 只 build mip 1+），仍用 c.a。
                    float a = c.a;
                    if (mip >= 1.0) {
                        float4 ani = gVxgiAniso.SampleLevel(gLpvSampler, uv, mip);
                        float3 wdir = abs(dW);
                        float aDir = max(wdir.x * ani.r, max(wdir.y * ani.g, wdir.z * ani.b));
                        a = max(a, aDir);
                    }
                    // Front-to-back composit。
                    acc += trans * c.rgb * a;
                    trans *= max(0.0, 1.0 - a);
                    if (trans < 0.01) break;

                    dist += diameter * 0.5;   // 步长 = 半个 cone 直径
                }
                vxgiAccum += acc * coneWeights[ci];
                vxgiCov += (1.0 - trans) * coneWeights[ci];
            }

            float3 vxgiDiff = vxgiAccum * albedo * (1.0 - metallic);
            diffuse = lerp(diffuse, vxgiDiff, saturate(vxgiCov));
        }

        // NDGI (Neural Dynamic GI)：使用 MLP 推理替代 DDGI 的探针三线性插值
        if (gFrame.ddgiCounts.w == 2) {
            float3 ndgiIrr = ndgiSampleIrradiance(worldPos, N);
            diffuse = ndgiIrr * albedo * (1.0 - metallic) / 3.14159265;
        }

        // M11 DDGI：找 worldPos 周围 8 个 probe，每个：
        //   1. trilinear 权重 (frac/1-frac per axis)
        //   2. backface cull：probe 朝 pixel 的方向 vs N，背面权重 0
        //   3. Chebyshev visibility：sample distance atlas, mean/mean² → vis
        //   4. sample irradiance atlas at N direction
        // 累加 weighted irradiance，归一化。Lerp 进 diffuse。
        if (gFrame.ddgiCounts.w != 0) {
            uint3 pcount = uint3(gFrame.ddgiCounts.xyz);
            float3 ddgiOrig = gFrame.ddgiOrigin.xyz;
            float3 ddgiSp = gFrame.ddgiSpacing.xyz;
            float octaIrrF  = float(gFrame.ddgiOctaSizes.x);
            float octaDistF = float(gFrame.ddgiOctaSizes.y);

            // probe 浮点坐标 + clamp 到 grid 范围。
            float3 fp = (worldPos - ddgiOrig) / ddgiSp;
            fp = clamp(fp, float3(0), float3(pcount) - 1.001);
            int3 ip0 = int3(floor(fp));
            float3 frac = fp - float3(ip0);

            float3 ddgiAcc = float3(0);
            float ddgiW = 0.0;

            float atlasIrrW = float(pcount.x) * octaIrrF;
            float atlasIrrH = float(pcount.y) * float(pcount.z) * octaIrrF;
            float atlasDistW = float(pcount.x) * octaDistF;
            float atlasDistH = float(pcount.y) * float(pcount.z) * octaDistF;

            float2 nOctUv = dirToOctUv(N);
            for (int corner = 0; corner < 8; ++corner) {
                int3 off = int3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
                int3 ip = ip0 + off;
                if (any(ip < 0) || any(ip >= int3(pcount))) continue;

                // B.5：跳过 inactive probe（卡墙体内部，data 是垃圾）。
                uint probeIdx = (ip.z * pcount.y + ip.y) * pcount.x + ip.x;
                if (gDdgiProbeStates[probeIdx] == 0u) continue;

                float3 trilin = lerp(1.0 - frac, frac, float3(off));
                float trilinW = trilin.x * trilin.y * trilin.z;
                if (trilinW < 1e-4) continue;

                float3 probePos = ddgiOrig + float3(ip) * ddgiSp;
                float3 toPixel = worldPos - probePos;
                float distToPixel = length(toPixel);
                float3 dir = (distToPixel > 1e-5) ? (toPixel / distToPixel) : N;

                // Backface 权重：probe 看 pixel 的方向，pixel 朝 probe (即 -dir) 与 N 同向才有贡献。
                float bfRaw = (dot(N, -dir) + 1.0) * 0.5;
                float bfW = bfRaw * bfRaw + 0.2;   // 软化避免硬切 0

                // Chebyshev VSM：在 distance atlas 上 dirToPixel 方向取 (mean, mean²)
                float2 distOctUv = dirToOctUv(dir);
                float2 distAtlasUv = float2(
                    (float(ip.x) * octaDistF + distOctUv.x * octaDistF) / atlasDistW,
                    ((float(ip.z) * float(pcount.y) + float(ip.y)) * octaDistF + distOctUv.y * octaDistF) / atlasDistH);
                float2 distSample = gDdgiDist.SampleLevel(gLpvSampler, distAtlasUv, 0);
                float meanD = distSample.x;
                float varD  = max(0.0, distSample.y - meanD * meanD);
                float chev = 1.0;
                if (distToPixel > meanD) {
                    float diff = distToPixel - meanD;
                    chev = varD / (varD + diff * diff);
                    chev = max(0.0, chev * chev * chev);
                }

                // Irradiance atlas 在 N 方向取 cosine-convolved radiance
                float2 irrAtlasUv = float2(
                    (float(ip.x) * octaIrrF + nOctUv.x * octaIrrF) / atlasIrrW,
                    ((float(ip.z) * float(pcount.y) + float(ip.y)) * octaIrrF + nOctUv.y * octaIrrF) / atlasIrrH);
                float3 irr = gDdgiIrr.SampleLevel(gLpvSampler, irrAtlasUv, 0).rgb;

                float w = trilinW * bfW * chev;
                ddgiAcc += irr * w;
                ddgiW += w;
            }

            if (ddgiW > 1e-4) {
                float3 ddgiIrradiance = ddgiAcc / ddgiW;
                float3 ddgiDiff = ddgiIrradiance * albedo * (1.0 - metallic) / 3.14159265;
                // DDGI 模式独占替换 IBL diffuse（语义同 PRT）
                diffuse = ddgiDiff;
            }
        }

        // M8 PRT：trilinear 采 visibility transfer SH，与 sun lightSH 做
        // element-wise 乘 + Ramamoorthi/Hanrahan A_l 卷积系数，最后用 Y_k(N)
        // 重建。prtCounts.z 编码 SH order：0=SH4，1=SH9，2=SH16。
        // 注：A_l for clamped-cosine kernel：A_0=π, A_1=2π/3, A_2=π/4, A_3=0,
        // A_4=-π/24... 所以 SH16（含 l=3）对 diffuse 与 SH9 数学上等价。
        // 实现 SH16 path 主要为 toggle 验证 + 未来 specular PRT 基础设施。
        if (gFrame.prtCounts.y != 0) {
            float3 pMin = gFrame.prtGridMinCell.xyz;
            float pCell = gFrame.prtGridMinCell.w;
            float pGridDim = float(gFrame.prtCounts.x) * pCell;
            float3 pUv = (worldPos - pMin) / pGridDim;
            if (all(pUv >= 0.0) && all(pUv <= 1.0)) {
                float4 transferA = gPrtTransfer.SampleLevel(gLpvSampler, pUv, 0);
                // A_l 卷积系数（clamped cosine kernel SH）：l=0 π，l=1 2π/3，l=2 π/4
                const float4 AclipL01 = float4(3.14159265, 2.094395, 2.094395, 2.094395);

                // SH4 path：仅 l=0,1（4 系数）
                float4 effR = gFrame.prtLightSH_R * transferA * AclipL01;
                float4 effG = gFrame.prtLightSH_G * transferA * AclipL01;
                float4 effB = gFrame.prtLightSH_B * transferA * AclipL01;
                float4 basisN = float4(0.282094792,
                                       0.488602512 * N.y,
                                       0.488602512 * N.z,
                                       0.488602512 * N.x);
                float3 irr = float3(dot(effR, basisN),
                                    dot(effG, basisN),
                                    dot(effB, basisN));

                // SH9：加 l=2 共 5 项（prtCounts.z >= 1）
                if (gFrame.prtCounts.z >= 1) {
                    float4 transferB = gPrtTransferB.SampleLevel(gLpvSampler, pUv, 0);
                    float  transferC = gPrtTransferC.SampleLevel(gLpvSampler, pUv, 0).x;
                    const float Aclip2 = 0.785398163;   // π/4
                    float Y2n2 = 1.092548431 * N.x * N.y;
                    float Y2n1 = 1.092548431 * N.y * N.z;
                    float Y20  = 0.315391565 * (3.0 * N.z * N.z - 1.0);
                    float Y21  = 1.092548431 * N.z * N.x;
                    float Y22  = 0.546274215 * (N.x * N.x - N.y * N.y);
                    float4 basisN2 = float4(Y2n2, Y2n1, Y20, Y21);
                    float4 effR2 = gFrame.prtLightSH9_R0 * transferB * Aclip2;
                    float  effR2_8 = gFrame.prtLightSH9_R1.x * transferC * Aclip2;
                    irr.r += dot(effR2, basisN2) + effR2_8 * Y22;
                    float4 effG2 = gFrame.prtLightSH9_G0 * transferB * Aclip2;
                    float  effG2_8 = gFrame.prtLightSH9_G1.x * transferC * Aclip2;
                    irr.g += dot(effG2, basisN2) + effG2_8 * Y22;
                    float4 effB2 = gFrame.prtLightSH9_B0 * transferB * Aclip2;
                    float  effB2_8 = gFrame.prtLightSH9_B1.x * transferC * Aclip2;
                    irr.b += dot(effB2, basisN2) + effB2_8 * Y22;
                }

                // B.10 SH16：加 l=3 共 7 项（prtCounts.z >= 2）。A_3 = 0 →
                // 数学上对 diffuse 不贡献；这里仍走完整代码路径让 toggle
                // 可见、且把 SH16 light/transfer 计算就位为后续 specular PRT
                // 用。Aclip3 = 0 让结果与 SH9 完全相同。
                if (gFrame.prtCounts.z >= 2) {
                    float4 transferD = gPrtTransferD.SampleLevel(gLpvSampler, pUv, 0);
                    float3 transferE = gPrtTransferE.SampleLevel(gLpvSampler, pUv, 0).rgb;
                    const float Aclip3 = 0.0;   // R-H 2001 给出 A_3 = 0
                    float Y3n3 = 0.590043589 * N.y * (3.0 * N.x * N.x - N.y * N.y);
                    float Y3n2 = 2.890611442 * N.x * N.y * N.z;
                    float Y3n1 = 0.457045799 * N.y * (5.0 * N.z * N.z - 1.0);
                    float Y30  = 0.373176333 * N.z * (5.0 * N.z * N.z - 3.0);
                    float Y31  = 0.457045799 * N.x * (5.0 * N.z * N.z - 1.0);
                    float Y32  = 1.445305721 * N.z * (N.x * N.x - N.y * N.y);
                    float Y33  = 0.590043589 * N.x * (N.x * N.x - 3.0 * N.y * N.y);
                    float4 basisN3a = float4(Y3n3, Y3n2, Y3n1, Y30);
                    float3 basisN3b = float3(Y31, Y32, Y33);
                    float4 effR3a = gFrame.prtLightSH16_R0 * transferD * Aclip3;
                    float3 effR3b = gFrame.prtLightSH16_R1.xyz * transferE * Aclip3;
                    irr.r += dot(effR3a, basisN3a) + dot(effR3b, basisN3b);
                    float4 effG3a = gFrame.prtLightSH16_G0 * transferD * Aclip3;
                    float3 effG3b = gFrame.prtLightSH16_G1.xyz * transferE * Aclip3;
                    irr.g += dot(effG3a, basisN3a) + dot(effG3b, basisN3b);
                    float4 effB3a = gFrame.prtLightSH16_B0 * transferD * Aclip3;
                    float3 effB3b = gFrame.prtLightSH16_B1.xyz * transferE * Aclip3;
                    irr.b += dot(effB3a, basisN3a) + dot(effB3b, basisN3b);
                }

                irr = max(float3(0), irr);
                float3 prtDiff = irr * albedo * (1.0 - metallic) / 3.14159265;
                diffuse = prtDiff;
            }
        }

        // M6 LPV：worldPos → grid UV → trilinear 采 SH → eval(N)，结果是
        // 入射 radiance（已 cosine kernel 加权过），乘 albedo·(1-metallic)
        // 即 reflected diffuse。grid 之外 fallback 到 IBL；边界用 smoothstep
        // fade 防硬切（5% padding）。
        if (gFrame.lpvCounts.y != 0) {
            float3 gMin    = gFrame.lpvGridMinCell.xyz;
            float  cell    = gFrame.lpvGridMinCell.w;
            float  gridDim = float(gFrame.lpvCounts.x) * cell;
            float3 lpvUv   = (worldPos - gMin) / gridDim;

            // grid 内部 fade = 1，边缘 5% 平滑过渡到 0。
            float3 fadeMin = smoothstep(0.0, 0.05, lpvUv);
            float3 fadeMax = smoothstep(0.0, 0.05, 1.0 - lpvUv);
            float coverage = min(min(fadeMin.x, fadeMin.y),
                             min(fadeMin.z, min(fadeMax.x, min(fadeMax.y, fadeMax.z))));
            coverage = max(0.0, coverage);

            if (coverage > 0.0 && all(lpvUv >= 0.0) && all(lpvUv <= 1.0)) {
                ShCoeffs shR; shR.c = gLpvR.SampleLevel(gLpvSampler, lpvUv, 0);
                ShCoeffs shG; shG.c = gLpvG.SampleLevel(gLpvSampler, lpvUv, 0);
                ShCoeffs shB; shB.c = gLpvB.SampleLevel(gLpvSampler, lpvUv, 0);
                float3 lpvRadiance = float3(evalSh(shR, N),
                                            evalSh(shG, N),
                                            evalSh(shB, N));
                float3 lpvDiff = lpvRadiance * albedo * (1.0 - metallic);
                diffuse = lerp(diffuse, lpvDiff, coverage);
            }
        }

        // M9 RT GI：当 ray 命中几何时，替换 IBL diffuse 为 RT 追踪的一次反弹
        // 直接光。rtGI.rgb 已含 BRDF（baseColor/π * NdotL * sun），无需再乘
        // albedo。rtGI.a=1 时完全替换 diffuse；=0 时保持原 IBL diffuse。
        {
            float4 rtGi = gRtGI.Load(int3(pix, 0));
            if (rtGi.a > 0.0) {
                diffuse = lerp(diffuse, rtGi.rgb, rtGi.a);
            }
        }

        // Specular 项 = lerp(IBL 镜面 prefilter, SSR 命中色, SSR 置信度)。
        // 同上，SSR 在屏幕外 / 高粗糙度时 fade=0，全用 IBL specular。
        float3 iblSpec  = evalIBLSpecular(N, V, albedo, metallic, roughness,
                                          (float)gFrame.counts.y);
        float3 ssrSpec  = applyIBLSpecularBRDF(ssrF.rgb, N, V, albedo, metallic, roughness);
        float3 specular = lerp(iblSpec, ssrSpec, ssrF.a);

        indirect = diffuse + specular;
    } else {
        // None 模式 fallback：没有 env probe，没有 SS GI，只给一份非常粗略
        // 的方向感 ambient。
        // - F0 当 "镜面" 项（金属用 albedo 当反射率；非金属 0.04）。
        // - albedo*(1-metallic) 当 "漫反射" 项（金属抑制掉 Lambertian）。
        // - hemispheric env：Y 朝上的面更亮（"天空"），Y 朝下的面较暗（"地面 0.4"）。
        // 这条路径不严格 PBR，但保证任何材质在 None 下也能看出 3D 结构。
        float3 F0 = lerp(float3(0.04), albedo, metallic);
        float3 diffuse  = albedo * (1.0 - metallic);
        float3 specular = F0;
        float skyT = saturate(N.y * 0.5 + 0.5);
        float3 sky    = gFrame.ambient.rgb;
        float3 ground = gFrame.ambient.rgb * 0.4;
        float3 envProxy = lerp(ground, sky, skyT);
        indirect = envProxy * (diffuse + specular);
    }
    indirect *= ao;   // material AO ∩ SSAO，对 indirect 项整体衰减

    // C.4 ReSTIR DI：把 reservoir 重采样后的多 point light 直接光叠加到
    // 最终结果。关闭时 gRestir 已被 clear-to-0，本项贡献 0。
    float3 restirRgb = gRestir.Load(int3(pix, 0)).rgb;

    gOutHdr[pix] = float4(sun + indirect + emissive + restirRgb, 1.0);
}
