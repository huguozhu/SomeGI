// 屏幕空间环境光遮蔽（cosine-weighted 半球采样）—— M4.1。
//
// 算法：
// 1. 从 gDepth 反推当前像素的 view-space 位置 vP。
// 2. 从 gNormalRough 取世界法线，view 矩阵转到 view-space 法线 vN。
// 3. 在 vN 周围的 cosine 半球内采 N 个方向（tangent 基底 → view），
//    每个样本沿方向走 [0.25, 1.0]×radius 的随机距离得 vSamp。
// 4. 把 vSamp 投回屏幕，从 gDepth 读那点的真实场景深度 sceneDepth。
//    sceneDepth → vRef。如果 vRef 比 vSamp 离相机更近（vRef.z > vSamp.z）
//    且不在远到 rangeCheck 限制外 → 视为被遮挡。
// 5. AO = 1 - 平均遮挡。R8 输出，1 = 完全无遮蔽。
//
// 噪点来源：每像素仅 16 sample + per-pixel 哈希抖动 → 邻居采样方向差距大。
// 解：lighting.slang 端有 5×5 depth-bilateral 整体过滤。
//
// View-space 约定：相机看向 -Z，所以 z 为"负"代表离相机近、"更负"代表更远。
// 这里 NoV / dot 都是常规右手系 view-space。

[[vk::binding(0, 0)]] Texture2D<float4> gNormalRough;   // xyz=world normal
[[vk::binding(1, 0)]] Texture2D<float>  gDepth;
[[vk::binding(2, 0)]] RWTexture2D<float> gOutAO;        // R8 输出

struct SsaoPC {
    float4x4 proj;        // view → clip（Vulkan Y-flipped）
    float4x4 invProj;     // clip → view（深度反推 view-space 位置用）
    float4x4 view;        // world → view（把 world 法线带到 view-space）
    uint2  outSize;
    float2 invOutSize;
    float  radius;        // 世界单位的采样半径；App 按场景 AABB 自适应
    float  bias;          // 深度比较的小 epsilon，避免自遮挡
    uint   sampleCount;   // 每像素采样数；ImGui 限制 [4, 64]
    uint   _pad;
};
[[vk::push_constant]] SsaoPC gPC;

float3 viewPosFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 clip = float4(ndc, depth, 1.0);
    float4 v = mul(gPC.invProj, clip);
    return v.xyz / v.w;
}

uint hash3(uint x, uint y, uint z) {
    uint h = x * 73856093u ^ y * 19349663u ^ z * 83492791u;
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}
float rand(uint2 pix, uint i) {
    return float(hash3(pix.x, pix.y, i)) * (1.0 / 4294967296.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) return;
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) { gOutAO[pix] = 1.0; return; }

    float3 vP = viewPosFromDepth(pix, depth);
    float3 wN = normalize(gNormalRough.Load(int3(pix, 0)).xyz);
    float3 vN = normalize(mul((float3x3)gPC.view, wN));

    // Tangent basis around the view-space normal.
    float3 up = abs(vN.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 T  = normalize(cross(up, vN));
    float3 B  = cross(vN, T);

    float occ = 0.0;
    uint  used = 0;
    for (uint i = 0; i < gPC.sampleCount; ++i) {
        // Cosine-weighted hemisphere sample in tangent space, then transform
        // into view space.
        float2 xi = float2(rand(pix, i * 2), rand(pix, i * 2 + 1));
        float r1 = sqrt(xi.x);
        float r2 = 6.2831853 * xi.y;
        float3 hsTan = float3(r1 * cos(r2), r1 * sin(r2), sqrt(1.0 - xi.x));
        float3 hsView = T * hsTan.x + B * hsTan.y + vN * hsTan.z;

        // Vary the step length so we don't sample at the exact same radius
        // for every sample — gives more spatial variation.
        float r = lerp(0.25, 1.0, rand(pix, i + 0x9e377u)) * gPC.radius;
        float3 vSamp = vP + hsView * r;

        // Project sample into screen.
        float4 c = mul(gPC.proj, float4(vSamp, 1.0));
        if (c.w <= 0.0) continue;
        float3 ndc = c.xyz / c.w;
        if (any(abs(ndc.xy) > 1.0)) continue;

        float2 sUv = ndc.xy * 0.5 + 0.5;
        uint2  sPix = uint2(sUv * float2(gPC.outSize));
        float  sceneDepth = gDepth.Load(int3(sPix, 0));
        if (sceneDepth >= 1.0) { used++; continue; }
        float3 vRef = viewPosFromDepth(sPix, sceneDepth);

        // Range check fades out distant occluders so a far wall behind the
        // sample doesn't darken the foreground.
        float depthDiff = vP.z - vRef.z;
        float rangeCheck = smoothstep(0.0, 1.0, gPC.radius / max(abs(depthDiff), 1e-3));

        // Z is *negative* into the screen in view space; "more negative" = farther.
        // vRef.z > vSamp.z means scene point is *closer* than the hemisphere sample
        // → the sample is occluded.
        if (vRef.z > vSamp.z + gPC.bias) {
            occ += rangeCheck;
        }
        used++;
    }
    occ /= max(float(used), 1.0);
    gOutAO[pix] = saturate(1.0 - occ);
}
