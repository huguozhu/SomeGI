// RSM 几何 pass —— 从 sun 视角 ortho 投影渲染场景，MRT 输出
// (worldPos, worldNormal, flux=albedo·sunColor·sunIntensity·NdotL)。
// 后续 rsm_sample compute 在屏幕空间像素 disk gather 这些 VPL，累加
// 出一次反弹漫反射间接光。
//
// 4 张 RT（外加 depth attachment）：
//   RT0 rsmPosition (RGBA16F): xyz = worldPos
//   RT1 rsmNormal   (RGBA16F): xyz = worldNormal（归一化），与 GBuffer 同约定
//   RT2 rsmFlux     (RGBA16F): rgb = flux（受光面强度，背光面为 0）
//   depth (D32_SFLOAT): 标准深度，给后续 indirect shadow 等留位（M5 不用）
//
// VS 与 gbuffer.slang 几乎一样：用 sunViewProj 而不是 camera viewProj。
// PS 不做 shadow / 法线贴图（RSM 一阶近似不在乎那些细节）—— 只取 albedo
// + 几何 normal 算 flux，越简单越好（RSM 论文的"虚拟点光源"语义）。

import shared_types;

[[vk::binding(0, 0)]] ConstantBuffer<RsmFrameUniforms> gRsm;
[[vk::binding(1, 0)]] StructuredBuffer<MaterialGpu> gMaterials;
[[vk::binding(2, 0)]] SamplerState gLinear;
[[vk::binding(3, 0)]] Texture2D gTextures[];

[[vk::binding(10, 0)]] StructuredBuffer<DrawData> gDrawData;

struct VsIn {
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float4 tangent : TANGENT;
    float2 uv0     : TEXCOORD0;
};

struct VsOut {
    float4 svPos    : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float2 uv       : TEXCOORD2;
    nointerpolation int matIndex : TEXCOORD3;
};

[shader("vertex")]
VsOut vs_main(VsIn i, uint drawID : SV_InstanceID) {
    DrawData dd = gDrawData[drawID];
    VsOut o;
    float4 wp = mul(dd.modelMatrix, float4(i.pos, 1.0));
    o.worldPos = wp.xyz;
    // 用 sun 的 viewProj，所以 svPos 落在 sun light space 的 clip 坐标里。
    o.svPos = mul(gRsm.sunViewProj, wp);
    float3x3 n3 = (float3x3)dd.modelMatrix;
    o.normal  = normalize(mul(n3, i.normal));
    o.uv = i.uv0;
    o.matIndex = int(dd.materialIndex);
    return o;
}

float4 sampleTex(int idx, float2 uv) {
    if (idx < 0) return float4(1, 1, 1, 1);
    return gTextures[idx].Sample(gLinear, uv);
}

struct RsmOut {
    float4 rt0 : SV_Target0;   // worldPos
    float4 rt1 : SV_Target1;   // worldNormal
    float4 rt2 : SV_Target2;   // flux
};

[shader("pixel")]
RsmOut ps_main(VsOut i) {
    MaterialGpu m = gMaterials[i.matIndex];

    // 取 albedo（baseColor × baseColorTex）；alpha cutoff 与 GBuffer 保持
    // 一致 —— alpha-mask 透明部分不该参与 RSM。
    float4 baseTex = sampleTex(m.baseColorTex, i.uv);
    float4 base = m.baseColorFactor * baseTex;
    if (m.alphaMode == 1u && base.a < m.alphaCutoff) discard;

    float3 N = normalize(i.normal);
    // sunDir 约定：与主 FrameUBO 一致，从 surface 到 sun（normalize(-sunDir)
    // 是入射光方向）。直接用 sunDir 做 NdotL 也行，看约定。
    // 这里 gRsm.sunDir 跟主 FrameUBO.sunDir 同源，朝 sun 视角的"光来源
    // 方向"是 -sunDir。
    float3 L = normalize(-gRsm.sunDir.xyz);
    float NdotL = max(0.0, dot(N, L));

    // VPL flux = 入射 sun radiance × albedo × cosine。后续 sample 阶段
    // 把它当 "VPL 出射的 incoming light"，按 RSM 论文公式累加。
    float3 flux = base.rgb * gRsm.sunColor_intensity.rgb * gRsm.sunColor_intensity.w * NdotL;

    RsmOut o;
    o.rt0 = float4(i.worldPos, 1.0);
    o.rt1 = float4(N, 0.0);
    o.rt2 = float4(flux, 1.0);
    return o;
}
