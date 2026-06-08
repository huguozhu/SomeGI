// NDGI MLP 推理 —— 在 lighting shader 中查询神经辐照度缓存
// 作为 import 模块使用，不需要单独 dispatch
//
// 架构: 6(pos+normal) → 16 → 16 → 3(RGB irradiance), leaky ReLU

// 权重 buffer（与 lighting.slang bindings 27-32 对齐）
[[vk::binding(27, 0)]] StructuredBuffer<float4> gNdgiW1;  // 24 float4
[[vk::binding(28, 0)]] StructuredBuffer<float4> gNdgiB1;  // 4 float4
[[vk::binding(29, 0)]] StructuredBuffer<float4> gNdgiW2;  // 64 float4
[[vk::binding(30, 0)]] StructuredBuffer<float4> gNdgiB2;  // 4 float4
[[vk::binding(31, 0)]] StructuredBuffer<float4> gNdgiW3;  // 12 float4
[[vk::binding(32, 0)]] StructuredBuffer<float4> gNdgiB3;  // 1 float4

float3 ndgiSampleIrradiance(float3 worldPos, float3 N) {
    // Layer 1: 6→16（输入 pos, N 与权重矩阵相乘 + bias）
    float4 w1r0 = gNdgiW1[0],  w1r1 = gNdgiW1[1],  w1r2 = gNdgiW1[2],  w1r3 = gNdgiW1[3];
    float4 w1r4 = gNdgiW1[4],  w1r5 = gNdgiW1[5],  w1r6 = gNdgiW1[6],  w1r7 = gNdgiW1[7];
    float4 w1r8 = gNdgiW1[8],  w1r9 = gNdgiW1[9],  w1r10=gNdgiW1[10], w1r11=gNdgiW1[11];
    float4 w1r12=gNdgiW1[12], w1r13=gNdgiW1[13], w1r14=gNdgiW1[14], w1r15=gNdgiW1[15];
    float4 w1r16=gNdgiW1[16], w1r17=gNdgiW1[17], w1r18=gNdgiW1[18], w1r19=gNdgiW1[19];
    float4 w1r20=gNdgiW1[20], w1r21=gNdgiW1[21], w1r22=gNdgiW1[22], w1r23=gNdgiW1[23];

    float4 h1[4];
    h1[0] = gNdgiB1[0] + worldPos.x*w1r0 + worldPos.y*w1r4 + worldPos.z*w1r8
                       + N.x*w1r12 + N.y*w1r16 + N.z*w1r20;
    h1[1] = gNdgiB1[1] + worldPos.x*w1r1 + worldPos.y*w1r5 + worldPos.z*w1r9
                       + N.x*w1r13 + N.y*w1r17 + N.z*w1r21;
    h1[2] = gNdgiB1[2] + worldPos.x*w1r2 + worldPos.y*w1r6 + worldPos.z*w1r10
                       + N.x*w1r14 + N.y*w1r18 + N.z*w1r22;
    h1[3] = gNdgiB1[3] + worldPos.x*w1r3 + worldPos.y*w1r7 + worldPos.z*w1r11
                       + N.x*w1r15 + N.y*w1r19 + N.z*w1r23;

    // leaky ReLU
    h1[0] = max(h1[0], 0.0) + 0.01 * min(h1[0], 0.0);
    h1[1] = max(h1[1], 0.0) + 0.01 * min(h1[1], 0.0);
    h1[2] = max(h1[2], 0.0) + 0.01 * min(h1[2], 0.0);
    h1[3] = max(h1[3], 0.0) + 0.01 * min(h1[3], 0.0);

    // Layer 2: 16→16（展平 h1 为 16 标量，乘 W2 + bias）
    float h1f[16] = {
        h1[0].x, h1[0].y, h1[0].z, h1[0].w,
        h1[1].x, h1[1].y, h1[1].z, h1[1].w,
        h1[2].x, h1[2].y, h1[2].z, h1[2].w,
        h1[3].x, h1[3].y, h1[3].z, h1[3].w
    };

    float4 h2[4] = {gNdgiB2[0], gNdgiB2[1], gNdgiB2[2], gNdgiB2[3]};
    for (int j = 0; j < 16; j++) {
        float v = h1f[j];
        int base = j * 4;
        h2[0] += v * gNdgiW2[base + 0];
        h2[1] += v * gNdgiW2[base + 1];
        h2[2] += v * gNdgiW2[base + 2];
        h2[3] += v * gNdgiW2[base + 3];
    }

    h2[0] = max(h2[0], 0.0) + 0.01 * min(h2[0], 0.0);
    h2[1] = max(h2[1], 0.0) + 0.01 * min(h2[1], 0.0);
    h2[2] = max(h2[2], 0.0) + 0.01 * min(h2[2], 0.0);
    h2[3] = max(h2[3], 0.0) + 0.01 * min(h2[3], 0.0);

    // Layer 3: 16→3
    float h2f[16] = {
        h2[0].x, h2[0].y, h2[0].z, h2[0].w,
        h2[1].x, h2[1].y, h2[1].z, h2[1].w,
        h2[2].x, h2[2].y, h2[2].z, h2[2].w,
        h2[3].x, h2[3].y, h2[3].z, h2[3].w
    };
    float r = gNdgiB3[0].x;
    float g = gNdgiB3[0].y;
    float b = gNdgiB3[0].z;
    for (int j = 0; j < 16; j++) {
        float v = h2f[j];
        // W3: 16×3 matrix, stored as 12 float4 (48 floats), row-major packed
        // Row j has 3 floats at linear index j*3, j*3+1, j*3+2
        int base = j * 3;
        int s0 = base / 4, c0 = base % 4;
        int s1 = (base+1) / 4, c1 = (base+1) % 4;
        int s2 = (base+2) / 4, c2 = (base+2) % 4;
        float w0 = (c0==0?gNdgiW3[s0].x:c0==1?gNdgiW3[s0].y:c0==2?gNdgiW3[s0].z:gNdgiW3[s0].w);
        float w1 = (c1==0?gNdgiW3[s1].x:c1==1?gNdgiW3[s1].y:c1==2?gNdgiW3[s1].z:gNdgiW3[s1].w);
        float w2 = (c2==0?gNdgiW3[s2].x:c2==1?gNdgiW3[s2].y:c2==2?gNdgiW3[s2].z:gNdgiW3[s2].w);
        r += v * w0;
        g += v * w1;
        b += v * w2;
    }

    return max(float3(r, g, b), 0.0); // ReLU output (irradiance ≥ 0)
}

// Dummy entry point so this file compiles standalone as a SPIR-V module.
// When imported via `import ndgi_infer;`, only ndgiSampleIrradiance is used.
[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) { }

