// brdf_lut：split-sum 近似的"第二项"二维 LUT。
// 输入坐标 (u, v) = (NoV, roughness)；每个 texel 通过 GGX 重要性采样积分
// 出两个系数 (A, B)，运行时 evalIBLSpecular 用：
//      F * A + B
// 把材质相关的 Fresnel F 与"光线项"分离。这张表与材质 / 光源 / 场景
// 无关，理论上可序列化保存，目前每次启动重烘（~1 ms 量级，可忽略）。
// 输出格式 R16G16_SFLOAT：A、B 两通道。

[[vk::binding(0, 0)]] RWTexture2D<float2> gLut;

struct PushConsts {
    uint size;        // LUT 边长（256）
    uint _p0, _p1, _p2;
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

float gSchlickIBL(float NoV, float a) {
    float k = (a*a) / 2.0;
    return NoV / (NoV * (1 - k) + k);
}
float gSmithIBL(float NoV, float NoL, float a) {
    return gSchlickIBL(NoV, a) * gSchlickIBL(NoL, a);
}

float2 integrateBRDF(float NoV, float roughness) {
    float3 V = float3(sqrt(1.0 - NoV*NoV), 0, NoV);
    float A = 0.0, B = 0.0;
    float3 N = float3(0, 0, 1);
    float a = max(roughness*roughness, 1e-3);
    const uint SAMPLES = 1024u;
    for (uint i = 0u; i < SAMPLES; ++i) {
        float2 Xi = hammersley(i, SAMPLES);
        float3 H = importanceSampleGGX(Xi, N, a);
        float3 L = normalize(2.0 * dot(V, H) * H - V);
        float NoL = saturate(L.z);
        float NoH = saturate(H.z);
        float VoH = saturate(dot(V, H));
        if (NoL > 0.0) {
            float G = gSmithIBL(NoV, NoL, a);
            float G_Vis = (G * VoH) / (NoH * NoV);
            float Fc = pow(1.0 - VoH, 5.0);
            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    return float2(A, B) / float(SAMPLES);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.size || dt.y >= gPC.size) return;
    float NoV = (float(dt.x) + 0.5) / float(gPC.size);
    float roughness = (float(dt.y) + 0.5) / float(gPC.size);
    float2 ab = integrateBRDF(NoV, roughness);
    gLut[dt.xy] = ab;
}
