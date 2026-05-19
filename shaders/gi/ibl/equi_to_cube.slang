// equi_to_cube：把 HDR equirectangular 贴图（球面经纬投影）转成 cubemap
// 的 mip 0。每个 cube 面 + UV 算出 3D 方向 → 反查 equirect 的 (phi, theta)
// → 用线性 sampler 采色写出。
//
// 调度：dispatch 维度 (cubeSize/8, cubeSize/8, 6)，z 轴 = 6 个面。
// gCubeFaces 是 RWTexture2DArray<float4> view，layerCount = 6，覆盖 cube
// 的 6 layer。

[[vk::binding(0, 0)]] Texture2D<float4>  gEquirect;
[[vk::binding(1, 0)]] SamplerState        gLinear;
[[vk::binding(2, 0)]] RWTexture2DArray<float4> gCubeFaces;  // viewType=2D_ARRAY, 6 layers

struct PushConsts {
    uint cubeSize;   // 一面边长
    uint _p0, _p1, _p2;
};

[[vk::push_constant]] PushConsts gPC;

static const float PI = 3.14159265359;

// 给定 cube 面索引 + UV ∈ [0,1]，返回从立方体中心指向该 texel 的 3D 方向。
// 6 个面遵循 Vulkan cube 标准朝向：+X / -X / +Y / -Y / +Z / -Z。
float3 dirForFaceUV(uint face, float2 uv) {
    float2 st = uv * 2.0 - 1.0;   // [0,1] → [-1, 1]
    float3 d;
    switch (face) {
        case 0: d = float3( 1.0, -st.y, -st.x); break;  // +X
        case 1: d = float3(-1.0, -st.y,  st.x); break;  // -X
        case 2: d = float3( st.x,  1.0,  st.y); break;  // +Y
        case 3: d = float3( st.x, -1.0, -st.y); break;  // -Y
        case 4: d = float3( st.x, -st.y,  1.0); break;  // +Z
        default:d = float3(-st.x, -st.y, -1.0); break;  // -Z
    }
    return normalize(d);
}

// 把 3D 方向反查到 equirect 的 UV：用 (phi, theta) 球坐标线性映射。
// phi = atan2(z, x) 决定 u（横向，"经度"）；
// theta = asin(y) 决定 v（纵向，"纬度"，y=+1 顶部、y=-1 底部）。
float2 dirToEquirect(float3 d) {
    float phi = atan2(d.z, d.x);                 // [-PI, PI]
    float theta = asin(clamp(d.y, -1.0, 1.0));   // [-PI/2, PI/2]
    float u = phi / (2.0 * PI) + 0.5;
    float v = 0.5 - theta / PI;
    return float2(u, v);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.cubeSize || dt.y >= gPC.cubeSize || dt.z >= 6) return;
    // texel 中心采样（+0.5）避免 1-pixel 偏移。
    float2 uv = (float2(dt.xy) + 0.5) / float(gPC.cubeSize);
    float3 dir = dirForFaceUV(dt.z, uv);
    float2 eq = dirToEquirect(dir);
    float3 c = gEquirect.SampleLevel(gLinear, eq, 0).rgb;
    gCubeFaces[uint3(dt.xy, dt.z)] = float4(c, 1.0);
}
