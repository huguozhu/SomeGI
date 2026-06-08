// NDGI 权重 Xavier 初始化 —— 一次性执行
// dispatch (1, 1, 1)

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gW1;  // 24 float4
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> gB1;  // 4 float4
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gW2;  // 64 float4
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> gB2;  // 4 float4
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> gW3;  // 12 float4
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> gB3;  // 1 float4

struct InitPC {
    uint seed;
    float scale;
    uint _pad0, _pad1;
};
[[vk::push_constant]] InitPC gPC;

uint pcgHash(uint s) { s = s * 747796405u + 2891336453u; s = ((s>>((s>>28)+4))^s)*277803737u; return (s>>22)^s; }
float pcgFloat(inout uint s) { s = pcgHash(s); return float(s & 0xFFFFFF) / 16777216.0; }

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint tid = dt.x;
    uint rng = gPC.seed + tid * 127;

    float s1 = sqrt(6.0 / (6.0 + 16.0)) * gPC.scale;
    float s2 = sqrt(6.0 / (16.0 + 16.0)) * gPC.scale;
    float s3 = sqrt(6.0 / (16.0 + 3.0)) * gPC.scale;

    if (tid < 24) {
        gW1[tid] = float4(
            (pcgFloat(rng)-0.5)*2.0*s1, (pcgFloat(rng)-0.5)*2.0*s1,
            (pcgFloat(rng)-0.5)*2.0*s1, (pcgFloat(rng)-0.5)*2.0*s1);
    }
    if (tid < 64) {
        gW2[tid] = float4(
            (pcgFloat(rng)-0.5)*2.0*s2, (pcgFloat(rng)-0.5)*2.0*s2,
            (pcgFloat(rng)-0.5)*2.0*s2, (pcgFloat(rng)-0.5)*2.0*s2);
    }
    if (tid < 12) {
        gW3[tid] = float4(
            (pcgFloat(rng)-0.5)*2.0*s3, (pcgFloat(rng)-0.5)*2.0*s3,
            (pcgFloat(rng)-0.5)*2.0*s3, (pcgFloat(rng)-0.5)*2.0*s3);
    }
    if (tid < 4) {
        gB1[tid] = float4(0); gB2[tid] = float4(0);
    }
    if (tid == 0) gB3[0] = float4(0);
}
