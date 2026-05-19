// 屏幕空间光线步进的共享 helper —— SSR / SSGI 都用同一组例程。
//
// 思路：在世界空间步进光线，每一步投影到屏幕得到 UV 和 NDC z；查 gDepth
// 在该 UV 的实际场景深度；当光线投影 z 刚好"穿过"场景表面（在 thickness
// 阈值内）时算命中。
//
// Vulkan NDC z ∈ [0, 1]（GLM_FORCE_DEPTH_ZERO_TO_ONE 已开），离相机近 z 小、
// 离相机远 z 大；判定 sceneZ < ndcZ 就是 "ray 已经穿到了表面背后"。
//
// 已知瑕疵（fixed-step linear march 的常见问题）：
//   - 远距离时同一步 ndc z 跨度大 → 容易越过 thickness 漏命中（"banding"）。
//   - 薄物体（栏杆、纱网）经常被一步跨过去。
//   - 解法：Hi-Z 加速、二分细化、自适应 step。M4 没做，留给后续。

struct ScreenRayHit {
    bool   hit;          // 是否命中场景几何
    float2 uv;           // 命中处的屏幕 UV ∈ [0,1]
    float  depth;        // 命中处的场景 NDC z（来自 gDepth.Load）
    uint   stepsTaken;   // 实际跨出的步数（命中后 +1，否则 = maxSteps）
};

// 把世界空间的点投到屏幕 UV + NDC z。
// 返回 false 表示点在相机后方、视口外、或 NDC z 不在 [0, 1] —— 调用方
// 应放弃此 sample。
bool projectWorldToScreen(float4x4 viewProj, float3 worldPos,
                          out float2 uv, out float ndcZ) {
    uv = float2(0); ndcZ = 0.0;
    float4 c = mul(viewProj, float4(worldPos, 1.0));
    if (c.w <= 0.0) return false;                   // 点在相机后方
    float3 ndc = c.xyz / c.w;
    if (any(abs(ndc.xy) > 1.0)) return false;       // 屏幕外
    if (ndc.z < 0.0 || ndc.z > 1.0) return false;   // Vulkan [0,1] 之外
    uv   = ndc.xy * 0.5 + 0.5;
    ndcZ = ndc.z;
    return true;
}

// 沿 dir 从 origin 做线性 march。每步投到屏幕、查 gDepth、做深度比较。
// stepLength 单位为世界单位；maxSteps 限定开销；thickness 是 NDC z 上的
// 命中容差（越小越严格、越易漏；越大越宽容、越易假命中）。
ScreenRayHit marchScreenWorld(
    float3 origin, float3 dir,
    float stepLength, uint maxSteps, float thickness,
    float4x4 viewProj, Texture2D<float> gDepth, uint2 outSize)
{
    ScreenRayHit r;
    r.hit = false; r.uv = float2(0); r.depth = 0; r.stepsTaken = 0;

    // 初始 offset 一个 stepLength：避免命中自身（origin 已经在表面附近）。
    float t = stepLength;
    [loop]
    for (uint i = 0; i < maxSteps; ++i) {
        float3 p = origin + dir * t;
        float2 uv; float ndcZ;
        if (!projectWorldToScreen(viewProj, p, uv, ndcZ)) {
            // ray 飞出屏幕 / 后方 → miss，提前返回。
            r.stepsTaken = i;
            return r;
        }
        uint2 pix = uint2(uv * float2(outSize));
        float sceneZ = gDepth.Load(int3(pix, 0));
        // 命中条件：ray 的投影 z 已经 *越过* 场景表面（sceneZ < ndcZ 表示
        // 场景比 ray 离相机更近 —— ray 这一步穿过了表面），且越过量没超
        // thickness（避免把 ray 永远在表面后方也算命中）。
        if (sceneZ < ndcZ && (ndcZ - sceneZ) < thickness) {
            r.hit = true; r.uv = uv; r.depth = sceneZ; r.stepsTaken = i + 1;
            return r;
        }
        t += stepLength;
    }
    r.stepsTaken = maxSteps;
    return r;
}
