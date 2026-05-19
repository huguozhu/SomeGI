// DDGI 每帧 ray update —— 每 probe 投 64 rays，march voxel grid 取命中
// radiance + distance，写到 ray buffer 给后续 blend 阶段消费。
//
// dispatch (probeCount * raysPerProbe / 64, 1, 1)；每 thread = 一个 ray。
//
// ray buffer 布局（每 ray 32 字节 = 8 floats）：
//   [0..2] direction (world-space, normalized)
//   [3]    padding
//   [4..6] hit radiance (rgb)
//   [7]    hit distance（命中 = 实际距离；miss = 大正数标记）
//
// ray 方向用 spherical Fibonacci + per-frame 旋转 jitter（DDGI 论文）：
//   - probe 内 64 个方向均匀分布
//   - 每帧整体随机旋转 → 一定时间内覆盖所有方向避免方向偏置

[[vk::binding(0, 0)]] Texture3D<float4>     gVoxelGrid;
[[vk::binding(1, 0)]] SamplerState          gLinearClamp;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gRayBuf;   // 2 floats×4 = 8 floats/ray

struct UpdatePC {
    float3 ddgiOrigin;
    float  _pad0;
    float3 ddgiSpacing;
    float  _pad1;
    uint   probesX, probesY, probesZ;
    uint   raysPerProbe;
    float  randomRotation;     // 每帧 [0, 2π) 旋转 jitter
    float  voxelGridDim;
    float  vxgiCellSize;
    uint   vxgiResolution;
    float3 vxgiGridMin;
    float  _pad2;
};
[[vk::push_constant]] UpdatePC gPC;

// Spherical Fibonacci sequence: 给定 (i, N) 返回单位球面均匀分布方向。
float3 sphericalFibonacci(uint i, uint N) {
    const float PHI = 1.6180339887498949;   // 黄金比
    float phi = 6.2831853 * frac(float(i) * (PHI - 1.0));
    float cosTheta = 1.0 - (2.0 * float(i) + 1.0) / float(N);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

// 每帧绕 Y 轴随机旋转，让多帧累积覆盖更均匀。
float3 rotateY(float3 v, float angle) {
    float c = cos(angle), s = sin(angle);
    return float3(c*v.x + s*v.z, v.y, -s*v.x + c*v.z);
}

// march voxel grid：返回 (hitRadiance, hitDistance)。miss 时 dist 设大值。
float4 traceVoxel(float3 origin, float3 d) {
    float stepLen = gPC.vxgiCellSize * 1.5;
    int maxSteps = 32;
    float trans = 1.0;
    float3 acc = float3(0);
    for (int s = 1; s <= maxSteps; ++s) {
        float dist = stepLen * float(s);
        float3 pos = origin + d * dist;
        float3 uv = (pos - gPC.vxgiGridMin)
                  / (float(gPC.vxgiResolution) * gPC.vxgiCellSize);
        if (any(uv < 0.0) || any(uv > 1.0)) {
            // miss：返回 sky 近似（black or constant ambient）+ 大距离
            return float4(0, 0, 0, 1e6);
        }
        float4 c = gVoxelGrid.SampleLevel(gLinearClamp, uv, 0);
        if (c.a > 0.1) {
            // hit
            return float4(c.rgb, dist);
        }
        trans *= max(0.0, 1.0 - c.a);
    }
    return float4(0, 0, 0, 1e6);   // 步数用尽视为 miss
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint totalRays = gPC.probesX * gPC.probesY * gPC.probesZ * gPC.raysPerProbe;
    if (dt.x >= totalRays) return;

    uint probeIdx = dt.x / gPC.raysPerProbe;
    uint rayIdx   = dt.x % gPC.raysPerProbe;

    // probe 索引 → 3D 坐标
    uint pz = probeIdx / (gPC.probesX * gPC.probesY);
    uint py = (probeIdx / gPC.probesX) % gPC.probesY;
    uint px = probeIdx % gPC.probesX;
    float3 probePos = gPC.ddgiOrigin
                    + float3(float(px), float(py), float(pz)) * gPC.ddgiSpacing;

    // ray 方向
    float3 d = sphericalFibonacci(rayIdx, gPC.raysPerProbe);
    d = rotateY(d, gPC.randomRotation);

    float4 hit = traceVoxel(probePos, d);

    // 写 ray buffer。每 ray 占 2 个 float4：
    //   slot 0 = (dir.xyz, pad)
    //   slot 1 = (hitRgb, hitDist)
    uint base = dt.x * 2;
    gRayBuf[base + 0] = float4(d, 0.0);
    gRayBuf[base + 1] = hit;
}
