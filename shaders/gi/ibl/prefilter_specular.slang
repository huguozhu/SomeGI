// prefilter_specular：split-sum 第一项。GGX importance sampling 把
// envCube 在某个粗糙度下做镜面卷积，写到 specularCube 的某一 mip。
//
// 烘焙调度：每个 mip 独立 dispatch 一次（共 6 次），同一个 compute
// pipeline 复用，只是 push constant 的 (outSize, roughness) 不同：
//      mip k 的 outSize = max(specSize >> k, 1)
//      mip k 的 roughness = k / (mipCount - 1)，∈ [0, 1]
// mip 0 = 完美镜面（roughness 0），mip N-1 = 漫反射极限（roughness 1）。
// 运行时 evalIBLSpecular 按 roughness * (mips-1) 选 mip，三线性插值。
//
// 采样：每像素 1024 Hammersley 样本走 GGX 重要性采样（N=V 假设）；
// LearnOpenGL 的 IBL 教程里的 mip-bias 技巧（pdf vs solid angle）抑制
// 亮度噪点 —— 数学上是 split-sum 的近似，在视觉上够用。
//
// firefly clamp：min(env, 50) 同样防超亮像素污染。这是 M3 阶段为修
// "Sponza 启用 IBL 后白色噪点"专门加的。

[[vk::binding(0, 0)]] TextureCube<float4> gEnv;
[[vk::binding(1, 0)]] SamplerState         gLinear;
[[vk::binding(2, 0)]] RWTexture2DArray<float4> gOutFaces;  // 当前 mip 的 array view（6 面）

struct PushConsts {
    uint  outSize;       // 当前 mip 的边长
    uint  envCubeSize;   // env cube mip 0 边长（mip-bias 计算用）
    float roughness;     // 当前 mip 对应的粗糙度，∈ [0, 1]
    uint  _p2;
};

[[vk::push_constant]] PushConsts gPC;

static const float PI = 3.14159265359;

float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
float2 hammersley(uint i, uint N) {
    return float2(float(i) / float(N), radicalInverse_VdC(i));
}

float3 importanceSampleGGX(float2 Xi, float3 N, float a) {
    float phi = 2.0 * PI * Xi.x;
    float cosT = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinT = sqrt(1.0 - cosT*cosT);
    float3 H_t = float3(cos(phi)*sinT, sin(phi)*sinT, cosT);

    float3 up    = abs(N.y) < 0.999 ? float3(0,1,0) : float3(1,0,0);
    float3 tan_  = normalize(cross(up, N));
    float3 bitan = cross(N, tan_);
    return normalize(tan_ * H_t.x + bitan * H_t.y + N * H_t.z);
}

float distributionGGX(float NoH, float a) {
    float a2 = a*a;
    float d = NoH*NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-7);
}

float3 dirForFaceUV(uint face, float2 uv) {
    float2 st = uv * 2.0 - 1.0;
    float3 d;
    switch (face) {
        case 0: d = float3( 1.0, -st.y, -st.x); break;
        case 1: d = float3(-1.0, -st.y,  st.x); break;
        case 2: d = float3( st.x,  1.0,  st.y); break;
        case 3: d = float3( st.x, -1.0, -st.y); break;
        case 4: d = float3( st.x, -st.y,  1.0); break;
        default:d = float3(-st.x, -st.y, -1.0); break;
    }
    return normalize(d);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.outSize || dt.y >= gPC.outSize || dt.z >= 6) return;
    float2 uv = (float2(dt.xy) + 0.5) / float(gPC.outSize);
    float3 N = dirForFaceUV(dt.z, uv);
    float3 V = N;

    const uint SAMPLE_COUNT = 1024u;
    float a = max(gPC.roughness * gPC.roughness, 1e-3);
    float3 prefiltered = float3(0);
    float totalWeight = 0.0;

    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        float2 Xi = hammersley(i, SAMPLE_COUNT);
        float3 H = importanceSampleGGX(Xi, N, a);
        float3 L = normalize(2.0 * dot(V, H) * H - V);
        float NoL = saturate(dot(N, L));
        if (NoL > 0.0) {
            float NoH = saturate(dot(N, H));
            float D = distributionGGX(NoH, a);
            float pdf = (D * NoH / (4.0 * NoH)) + 1e-4;
            float saTexel = 4.0 * PI / (6.0 * gPC.envCubeSize * gPC.envCubeSize);
            float saSample = 1.0 / (float(SAMPLE_COUNT) * pdf + 1e-4);
            float mipLevel = (gPC.roughness == 0.0) ? 0.0 : 0.5 * log2(saSample / saTexel);

            // Clamp HDR sample to suppress specular fireflies. Sharp sun in
            // the env (luminance >> 1) otherwise dominates a tight GGX cone
            // and produces sparkly bright pixels in the prefiltered cube.
            float3 c = min(gEnv.SampleLevel(gLinear, L, mipLevel).rgb, float3(50.0));
            prefiltered += c * NoL;
            totalWeight += NoL;
        }
    }
    prefiltered = prefiltered / max(totalWeight, 1e-4);
    gOutFaces[uint3(dt.xy, dt.z)] = float4(prefiltered, 1.0);
}
