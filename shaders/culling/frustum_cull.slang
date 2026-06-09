import shared_types;

struct CullUniforms {
    float4x4 viewProj;       // for AABB screen projection
    float4   frustum[6];     // world-space frustum planes
    float2   screenSize;     // render target dimensions
    uint     drawCount;
    uint     hizMaxMip;      // max Hi-Z mip level (= 4)
};

[[vk::binding(0,0)]] StructuredBuffer<DrawData> gIn;
[[vk::binding(1,0)]] ConstantBuffer<CullUniforms> gCull;
[[vk::binding(2,0)]] RWStructuredBuffer<IndirectDrawCommand> gOut;
[[vk::binding(3,0)]] RWStructuredBuffer<uint> gCount;
// Hi-Z depth pyramid (4 mip levels, max-reduction, R32_SFLOAT)
[[vk::binding(4,0)]] Texture2D<float> gHiZMip1;
[[vk::binding(5,0)]] Texture2D<float> gHiZMip2;
[[vk::binding(6,0)]] Texture2D<float> gHiZMip3;
[[vk::binding(7,0)]] Texture2D<float> gHiZMip4;

float sampleHiZ(uint mip, int2 coord) {
    if (mip == 1) return gHiZMip1[coord];
    if (mip == 2) return gHiZMip2[coord];
    if (mip == 3) return gHiZMip3[coord];
    return gHiZMip4[coord];
}

// Project point to NDC, returns clip space position
float4 projectPoint(float3 wp, float4x4 vp) {
    return mul(vp, float4(wp, 1.0));
}

// Test if AABB is occluded by Hi-Z
bool isOccluded(float3 aabbMin, float3 aabbMax, float4x4 vp, float2 screenSize, uint maxMip) {
    // Project 8 corners to clip space
    float4 corners[8];
    corners[0] = projectPoint(float3(aabbMin.x, aabbMin.y, aabbMin.z), vp);
    corners[1] = projectPoint(float3(aabbMax.x, aabbMin.y, aabbMin.z), vp);
    corners[2] = projectPoint(float3(aabbMin.x, aabbMax.y, aabbMin.z), vp);
    corners[3] = projectPoint(float3(aabbMax.x, aabbMax.y, aabbMin.z), vp);
    corners[4] = projectPoint(float3(aabbMin.x, aabbMin.y, aabbMax.z), vp);
    corners[5] = projectPoint(float3(aabbMax.x, aabbMin.y, aabbMax.z), vp);
    corners[6] = projectPoint(float3(aabbMin.x, aabbMax.y, aabbMax.z), vp);
    corners[7] = projectPoint(float3(aabbMax.x, aabbMax.y, aabbMax.z), vp);

    // Compute screen-space bounding rect + find nearest depth
    float2 ssMin = float2(1e30, 1e30);
    float2 ssMax = float2(-1e30, -1e30);
    float  minDepth = 1e30;
    bool   behindNear = true;

    for (int i = 0; i < 8; ++i) {
        float4 c = corners[i];
        if (c.w <= 0.0) return false; // behind camera → not occluded

        float2 ndc = c.xy / c.w;
        float2 ss = (ndc * 0.5 + 0.5) * screenSize;
        ssMin = min(ssMin, ss);
        ssMax = max(ssMax, ss);

        float depth = c.z / c.w; // Vulkan NDC depth [0,1]
        minDepth = min(minDepth, depth);
        if (depth <= 1.0) behindNear = false; // any corner inside frustum
    }

    if (behindNear) return false; // fully behind near plane

    // Clamp to screen
    int2 rectMin = int2(max(0, int(ssMin.x)), max(0, int(ssMin.y)));
    int2 rectMax = int2(min(int(screenSize.x)-1, int(ssMax.x)), min(int(screenSize.y)-1, int(ssMax.y)));

    if (rectMin.x > rectMax.x || rectMin.y > rectMax.y) return false;

    // Choose mip: rect size determines which level
    int2 rectSize = rectMax - rectMin + 1;
    int maxDim = max(rectSize.x, rectSize.y);
    uint mip = uint(max(1, int(log2(float(maxDim))) + 1));
    mip = min(mip, maxMip);

    // Sample Hi-Z at the four corners of the rect
    int2 sampleCoords[4] = {rectMin, int2(rectMax.x, rectMin.y), int2(rectMin.x, rectMax.y), rectMax};
    float hizMax = 0.0;
    for (int j = 0; j < 4; ++j) {
        hizMax = max(hizMax, sampleHiZ(mip, sampleCoords[j]));
    }

    // If AABB's nearest depth > Hi-Z farthest depth, the AABB is completely behind
    return minDepth > hizMax;
}

bool aabbInFrustum(float3 mn, float3 mx, float4 f[6]) {
    for (int i = 0; i < 6; ++i) {
        float3 n = f[i].xyz; float d = f[i].w;
        float3 p = float3(n.x > 0 ? mn.x : mx.x, n.y > 0 ? mn.y : mx.y, n.z > 0 ? mn.z : mx.z);
        if (dot(n, p) + d < 0) return false;
    }
    return true;
}

[numthreads(256, 1, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= gCull.drawCount) return;
    DrawData dd = gIn[tid.x];
    if (!aabbInFrustum(dd.aabbMin, dd.aabbMax, gCull.frustum)) return;
    if (isOccluded(dd.aabbMin, dd.aabbMax, gCull.viewProj, gCull.screenSize, gCull.hizMaxMip)) return;
    uint idx; InterlockedAdd(gCount[0], 1, idx);
    IndirectDrawCommand cmd;
    cmd.indexCount = dd.indexCount; cmd.instanceCount = 1;
    cmd.firstIndex = dd.firstIndex; cmd.vertexOffset = dd.vertexOffset;
    cmd.firstInstance = tid.x;
    gOut[idx] = cmd;
}
