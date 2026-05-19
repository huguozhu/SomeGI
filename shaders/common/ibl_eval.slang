// IBL（Image-Based Lighting）评估辅助 —— forward_ibl / lighting compute 都
// 通过 import 这个文件得到 evalIBLDiffuse / evalIBLSpecular / applyIBL*BRDF。
//
// 资源在 set=1，由 IBLTechnique 提供：
//   binding 0: 漫反射辐照度立方贴图（cosine 半球预积分）
//   binding 1: 镜面 prefilter 立方贴图（按 mip 存不同粗糙度的卷积）
//   binding 2: split-sum 二维 BRDF 表（NoV × roughness → A,B 系数）
//   binding 3: 共享线性 sampler
//   binding 4: 用户在 ImGui 调的 IBL intensity
//
// split-sum 近似（Karis 2013）：把 ∫ Li(L) * BRDF(V,L) cos(θ) dL 分成两项独立
// 卷积 —— prefilter（仅与方向、roughness 有关，烘成立方贴图）× BRDF 积分
// （与 NoV、roughness 有关，烘成二维 LUT）。运行时只采两次纹理就近似出
// 整个半球积分，是实时 IBL 的标准做法。

[[vk::binding(0, 1)]] TextureCube<float4> gIblDiffuse;     // diffuseCube
[[vk::binding(1, 1)]] TextureCube<float4> gIblSpecular;    // specularCube
[[vk::binding(2, 1)]] Texture2D<float2>   gIblBrdfLut;     // brdfLut（A,B）
[[vk::binding(3, 1)]] SamplerState         gIblSampler;    // 线性 + clamp

struct IblParams {
    float intensity;       // ImGui slider，[0,4]，运行时调整 IBL 强度
    float _pad0, _pad1, _pad2;   // std140 vec4 对齐 padding
};
[[vk::binding(4, 1)]] ConstantBuffer<IblParams> gIblParams;

// "粗糙度感知"的 Schlick-Fresnel —— Lazarov 2013 的近似，roughness 越大
// 反射率随入射角的变化越小（粗糙金属几乎各向同性反射）。
float3 fresnelSchlickRoughness(float cosT, float3 F0, float roughness) {
    float a = 1.0 - roughness;
    return F0 + (max(float3(a), F0) - F0) * pow(saturate(1.0 - cosT), 5.0);
}

// 仅漫反射的 IBL 评估。返回 kd * irradiance * albedo * intensity。
// kd = (1 - F) * (1 - metallic)：金属没有 Lambertian 漫反射、Fresnel 反射
// 走掉的那部分能量也不计入 diffuse。irradiance 已经是 cosine-weighted
// 半球积分（在烘焙阶段做完），运行时按 N 直接采。
//
// 注意：applyIBLSpecularBRDF / applyIBLDiffuseBRDF（SSR/SSGI 替代路径）
// 故意不再乘 intensity —— 它们的 incoming radiance 来自 hdrPrev，那是
// 上一帧 lighting 的输出，已经包含了当时的 intensity；通过帧间 feedback
// 自然跟随 slider 变化。
float3 evalIBLDiffuse(float3 N, float3 V, float3 albedo, float metallic, float roughness)
{
    float NoV = saturate(dot(N, V));
    float3 F0 = lerp(float3(0.04), albedo, metallic);
    float3 F  = fresnelSchlickRoughness(NoV, F0, roughness);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    float3 irradiance = gIblDiffuse.SampleLevel(gIblSampler, N, 0).rgb;
    return kd * irradiance * albedo * gIblParams.intensity;
}

// 仅镜面的 IBL 评估（split-sum 第一项 × 第二项）。
// - 第一项 prefiltered：用反射方向 R 在 specularCube 上按 mip 采（mip = roughness *
//   (mips-1)，roughness 0 用 mip 0 镜面级，roughness 1 用最后 mip 几乎漫的）。
// - 第二项 ab：在 brdfLut 上按 (NoV, roughness) 采，得 (A, B) 两个 BRDF 积分系数。
// 最终用 F * A + B 的方式合成（A 抓 Fresnel 主导项，B 是非 Fresnel 残项）。
// 调用方也可以走 applyIBLSpecularBRDF：把 prefiltered 替换成 SSR 命中色 → 同等
// "受 BRDF 加权的入射光" 形式，这样 lerp(iblSpec, ssrSpec, ssr.a) 数值上
// 能直接相加。
float3 evalIBLSpecular(float3 N, float3 V, float3 albedo, float metallic, float roughness,
                      float specularMipCount)
{
    float NoV = saturate(dot(N, V));
    float3 R  = reflect(-V, N);
    float3 F0 = lerp(float3(0.04), albedo, metallic);
    float3 F  = fresnelSchlickRoughness(NoV, F0, roughness);
    float mip = roughness * (specularMipCount - 1.0);
    float3 prefiltered = gIblSpecular.SampleLevel(gIblSampler, R, mip).rgb;
    float2 ab = gIblBrdfLut.SampleLevel(gIblSampler, float2(NoV, roughness), 0).xy;
    return prefiltered * (F * ab.x + ab.y) * gIblParams.intensity;
}

// 便利复合：diffuse + specular。给不需要 splice SSR / SSGI 的调用方用。
// M4 的 lighting compute 已经分别用拆分版本了，这个主要给 M3 的 forward_ibl
// shader 留兼容。
float3 evalIBL(float3 N, float3 V, float3 albedo, float metallic, float roughness,
               float specularMipCount)
{
    return evalIBLDiffuse(N, V, albedo, metallic, roughness)
         + evalIBLSpecular(N, V, albedo, metallic, roughness, specularMipCount);
}

// 把 split-sum 的 BRDF 第二项作用到一段"替代入射光"上（典型用法：把 SSR
// 命中色当成 prefiltered 的等价物），让 SSR 结果能与 evalIBLSpecular 的
// 输出在同一量纲上 lerp。注意此处不乘 intensity，因为入射光自身已经
// 是带 intensity 的（来自 hdrPrev）。
float3 applyIBLSpecularBRDF(float3 incoming,
                            float3 N, float3 V, float3 albedo, float metallic, float roughness)
{
    float NoV = saturate(dot(N, V));
    float3 F0 = lerp(float3(0.04), albedo, metallic);
    float3 F  = fresnelSchlickRoughness(NoV, F0, roughness);
    float2 ab = gIblBrdfLut.SampleLevel(gIblSampler, float2(NoV, roughness), 0).xy;
    return incoming * (F * ab.x + ab.y);
}

// 把 Lambertian 漫反射 BRDF（kd * albedo）作用到一段"替代辐照度"上
// （SSGI 的半球平均 radiance），让 SSGI 结果能与 evalIBLDiffuse 的输出
// 在同一量纲上 lerp。同样不乘 intensity（feedback 路径已带）。
float3 applyIBLDiffuseBRDF(float3 incoming,
                           float3 N, float3 V, float3 albedo, float metallic, float roughness)
{
    float NoV = saturate(dot(N, V));
    float3 F0 = lerp(float3(0.04), albedo, metallic);
    float3 F  = fresnelSchlickRoughness(NoV, F0, roughness);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    return kd * incoming * albedo;
}
