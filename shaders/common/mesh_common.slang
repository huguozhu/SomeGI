// Mesh Shader 公共模块 —— Task Shader culling 逻辑和共享类型。
// import-only，不编译为独立 spv。

import shared_types;

// Task Shader → Mesh Shader 的 payload
struct TaskPayload {
    uint drawIndex;     // gDrawData 中的索引，Mesh Shader 据此读取几何数据
};

// Culling UBO —— 与 frustum_cull.slang 中 CullUniforms 对齐
struct CullUniforms {
    float4x4 viewProj;       // 用于 AABB 投影
    float4   frustum[6];     // 世界空间视锥体平面
    float2   screenSize;     // 渲染目标尺寸
    uint     drawCount;
    uint     hizMaxMip;      // Hi-Z mip 层级数（0=禁用遮挡剔除）
};

// Hi-Z 采样：按 mip 层级选贴图
float sampleHiZMip(uint mip, int2 coord, Texture2D<float> m1, Texture2D<float> m2,
                   Texture2D<float> m3, Texture2D<float> m4) {
    if (mip == 1) return m1[coord];
    if (mip == 2) return m2[coord];
    if (mip == 3) return m3[coord];
    return m4[coord];
}

// 测试 AABB 是否在视锥体内
bool aabbInFrustum(float3 mn, float3 mx, float4 f[6]) {
    for (int i = 0; i < 6; ++i) {
        float3 n = f[i].xyz; float d = f[i].w;
        float3 p = float3(n.x > 0 ? mn.x : mx.x, n.y > 0 ? mn.y : mx.y, n.z > 0 ? mn.z : mx.z);
        if (dot(n, p) + d < 0) return false;
    }
    return true;
}

// Hi-Z 遮挡测试：投影 AABB 到屏幕空间，与深度金字塔比较
bool isOccluded(float3 aabbMin, float3 aabbMax, float4x4 vp, float2 screenSize,
                uint maxMip, Texture2D<float> hiz1, Texture2D<float> hiz2,
                Texture2D<float> hiz3, Texture2D<float> hiz4) {
    if (maxMip == 0) return false;

    float4 corners[8];
    corners[0] = mul(vp, float4(aabbMin.x, aabbMin.y, aabbMin.z, 1.0));
    corners[1] = mul(vp, float4(aabbMax.x, aabbMin.y, aabbMin.z, 1.0));
    corners[2] = mul(vp, float4(aabbMin.x, aabbMax.y, aabbMin.z, 1.0));
    corners[3] = mul(vp, float4(aabbMax.x, aabbMax.y, aabbMin.z, 1.0));
    corners[4] = mul(vp, float4(aabbMin.x, aabbMin.y, aabbMax.z, 1.0));
    corners[5] = mul(vp, float4(aabbMax.x, aabbMin.y, aabbMax.z, 1.0));
    corners[6] = mul(vp, float4(aabbMin.x, aabbMax.y, aabbMax.z, 1.0));
    corners[7] = mul(vp, float4(aabbMax.x, aabbMax.y, aabbMax.z, 1.0));

    float2 ssMin = float2(1e30, 1e30);
    float2 ssMax = float2(-1e30, -1e30);
    float  minDepth = 1e30;
    bool   behindNear = true;

    for (int i = 0; i < 8; ++i) {
        float4 c = corners[i];
        if (c.w <= 0.0) return false;
        float2 ndc = c.xy / c.w;
        float2 ss = (ndc * 0.5 + 0.5) * screenSize;
        ssMin = min(ssMin, ss);
        ssMax = max(ssMax, ss);
        float depth = c.z / c.w;
        minDepth = min(minDepth, depth);
        if (depth <= 1.0) behindNear = false;
    }
    if (behindNear) return false;

    int2 rectMin = int2(max(0, int(ssMin.x)), max(0, int(ssMin.y)));
    int2 rectMax = int2(min(int(screenSize.x)-1, int(ssMax.x)), min(int(screenSize.y)-1, int(ssMax.y)));
    if (rectMin.x > rectMax.x || rectMin.y > rectMax.y) return false;

    int2 rectSize = rectMax - rectMin + 1;
    int maxDim = max(rectSize.x, rectSize.y);
    uint mip = uint(max(1, int(log2(float(maxDim))) + 1));
    mip = min(mip, maxMip);

    int2 sampleCoords[4] = {rectMin, int2(rectMax.x, rectMin.y), int2(rectMin.x, rectMax.y), rectMax};
    float hizMaxVal = 0.0;
    for (int j = 0; j < 4; ++j)
        hizMaxVal = max(hizMaxVal, sampleHiZMip(mip, sampleCoords[j], hiz1, hiz2, hiz3, hiz4));

    return minDepth > hizMaxVal;
}
