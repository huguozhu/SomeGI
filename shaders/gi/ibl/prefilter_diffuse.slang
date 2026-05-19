// prefilter_diffuse：对 envCube 做漫反射卷积（cosine-weighted 半球积分）。
// 烘焙阶段 dispatch 一次，输出 32×32 × 6 面的 diffuseCube（仅 1 mip）。
//
// 算法：per output texel 算法线 N，在 N 周围半球内按 (phi, theta) 网格做
// Riemann 求和：
//      irradiance(N) = ∫ L(ω) cos(θ) sin(θ) dω
// 步长 (dPhi, dTheta) = (0.025, 0.025) 约每像素 8000 samples，烘 32²×6
// 总样本约 5e7，启动时一次性烘焙耗时几十 ms 级别。
// 运行时 evalIBLDiffuse 只 SampleLevel 0 一次直接拿到 irradiance。
//
// firefly clamp：min(env, 50) 防止 HDR 中超亮的太阳像素污染半球积分。

[[vk::binding(0, 0)]] TextureCube<float4>    gEnv;
[[vk::binding(1, 0)]] SamplerState            gLinear;
[[vk::binding(2, 0)]] RWTexture2DArray<float4> gDiffuseFaces;

struct PushConsts {
    uint outSize;
    uint _p0, _p1, _p2;
};

[[vk::push_constant]] PushConsts gPC;

static const float PI = 3.14159265359;

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

    float3 up    = abs(N.y) < 0.999 ? float3(0,1,0) : float3(1,0,0);
    float3 right = normalize(cross(up, N));
    up           = normalize(cross(N, right));

    float3 sum = float3(0);
    float weight = 0.0;
    const float dPhi = 0.025;
    const float dTheta = 0.025;
    for (float phi = 0.0; phi < 2.0 * PI; phi += dPhi) {
        for (float theta = 0.0; theta < 0.5 * PI; theta += dTheta) {
            float3 L_t = float3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
            float3 L = L_t.x * right + L_t.y * up + L_t.z * N;
            // Mirror the specular prefilter clamp so a single bright pixel
            // (sun) doesn't blow out the diffuse irradiance integral.
            float3 c = min(gEnv.SampleLevel(gLinear, L, 0).rgb, float3(50.0));
            sum += c * cos(theta) * sin(theta);
            weight += 1.0;
        }
    }
    float3 avg = PI * sum / weight;
    gDiffuseFaces[uint3(dt.xy, dt.z)] = float4(avg, 1.0);
}
