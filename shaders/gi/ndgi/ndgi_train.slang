// NDGI MLP 在线训练 —— mini-batch SGD + EMA 平滑 + 反向传播
//
// dispatch (1, 1, 1): 单线程执行（权重更新需原子性），
// 内循环遍历 mini-batch 样本，累积梯度后更新权重
//
// 架构: 6(pos+normal) → 16 → 16 → 3(RGB), leaky ReLU(α=0.01)
// 损失: MSE(pred, target)
// 优化器: SGD + EMA 平滑

// 权重 buffer（RW 读写 —— 训练更新）
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gW1;  // 24 float4 (96 floats)
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> gB1;  // 4 float4 (16 floats)
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gW2;  // 64 float4 (256 floats)
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> gB2;  // 4 float4 (16 floats)
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> gW3;  // 12 float4 (48 floats)
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> gB3;  // 1 float4 (3 floats in .xyz)

// 样本 buffer（probe trace 写入，只读）
[[vk::binding(6, 0)]] StructuredBuffer<float> gSamples; // 9 floats/sample
[[vk::binding(7, 0)]] StructuredBuffer<uint> gSampleCount;

struct TrainPC {
    float learningRate;    // 默认 0.01
    float emaAlpha;        // EMA 平滑系数，默认 0.95
    uint  batchSize;       // mini-batch 大小，默认 256
    uint  numIterations;   // 每帧训练迭代数，默认 4
    uint  totalSamples;    // 从 gSampleCount[0] 读取
    uint  _pad0, _pad1, _pad2;
};
[[vk::push_constant]] TrainPC gPC;

// === 权重 unpack / pack helpers ===
// W1: 6→16, stored as 24 float4. W1[i] = 4 output neurons' weights for ALL 6 inputs
// Actually: W1 layout is row-major for 16 output neurons × 6 inputs.
// W1[0].xyzw = weights from input[0] to neurons 0,1,2,3
// W1[1].xyzw = weights from input[1] to neurons 0,1,2,3  (each neuron per component)
// Wait, let me use a simple approach that matches the inference code.

// For the inference in ndgi_infer.slang:
// h1[0] = B1[0] + pos.x*W1[0] + pos.y*W1[4] + pos.z*W1[8] + N.x*W1[12] + N.y*W1[16] + N.z*W1[20]
// So W1[0] contains weights from (pos.x) to neurons (0,1,2,3)
// W1[4] contains weights from (pos.y) to neurons (0,1,2,3)
// etc.
// W1[j*4 + 0..3] are the 4 float4 weights from input j to all 16 neurons.

// For backprop, we need W1 as [16][6] or [6][16]. Let me use a SIMPLER approach:
// Unpack float4 arrays into float[96] for W1, etc., do the compute, pack back.
// This is slower but correct and maintainable.

// === 前向推理（与 ndgi_infer.slang 一致的实现）===
float3 ndgiForward(float3 pos, float3 N,
                   out float h1[16], out float h2[16],
                   float w1[96], float b1[16], float w2[256], float b2[16],
                   float w3[48], float b3[3]) {
    // Layer 1: 6→16, leaky ReLU
    for (int j = 0; j < 16; j++) {
        float sum = b1[j];
        sum += pos.x * w1[0*16 + j];  // input 0 → neuron j
        sum += pos.y * w1[1*16 + j];  // input 1 → neuron j
        sum += pos.z * w1[2*16 + j];  // input 2 → neuron j
        sum += N.x   * w1[3*16 + j];  // input 3 → neuron j
        sum += N.y   * w1[4*16 + j];  // input 4 → neuron j
        sum += N.z   * w1[5*16 + j];  // input 5 → neuron j
        float act = max(sum, 0.0) + 0.01 * min(sum, 0.0); // leaky ReLU
        h1[j] = act;
    }

    // Layer 2: 16→16, leaky ReLU
    for (int j = 0; j < 16; j++) {
        float sum = b2[j];
        for (int k = 0; k < 16; k++) sum += h1[k] * w2[k*16 + j];
        float act = max(sum, 0.0) + 0.01 * min(sum, 0.0);
        h2[j] = act;
    }

    // Layer 3: 16→3, ReLU (irradiance ≥ 0)
    float r = b3[0], g = b3[1], b = b3[2];
    for (int k = 0; k < 16; k++) {
        r += h2[k] * w3[k*3 + 0];
        g += h2[k] * w3[k*3 + 1];
        b += h2[k] * w3[k*3 + 2];
    }
    return max(float3(r, g, b), 0.0);
}

// === Unpack weights from float4 arrays to float arrays ===
void unpackWeights(out float w1[96], out float b1[16],
                   out float w2[256], out float b2[16],
                   out float w3[48], out float b3[3]) {
    for (int i = 0; i < 96; i++) { int s=i/4, c=i%4; w1[i] = (c==0?gW1[s].x:c==1?gW1[s].y:c==2?gW1[s].z:gW1[s].w); }
    for (int i = 0; i < 16; i++) { int s=i/4, c=i%4; b1[i] = (c==0?gB1[s].x:c==1?gB1[s].y:c==2?gB1[s].z:gB1[s].w); }
    for (int i = 0; i < 256; i++) { int s=i/4, c=i%4; w2[i] = (c==0?gW2[s].x:c==1?gW2[s].y:c==2?gW2[s].z:gW2[s].w); }
    for (int i = 0; i < 16; i++) { int s=i/4, c=i%4; b2[i] = (c==0?gB2[s].x:c==1?gB2[s].y:c==2?gB2[s].z:gB2[s].w); }
    for (int i = 0; i < 48; i++) { int s=i/4, c=i%4; w3[i] = (c==0?gW3[s].x:c==1?gW3[s].y:c==2?gW3[s].z:gW3[s].w); }
    b3[0]=gB3[0].x; b3[1]=gB3[0].y; b3[2]=gB3[0].z;
}

// === Pack weights back to float4 arrays ===
void packWeights(float w1[96], float b1[16], float w2[256], float b2[16],
                 float w3[48], float b3[3]) {
    for (int i = 0; i < 24; i++) gW1[i] = float4(w1[i*4], w1[i*4+1], w1[i*4+2], w1[i*4+3]);
    for (int i = 0; i < 4; i++) gB1[i] = float4(b1[i*4], b1[i*4+1], b1[i*4+2], b1[i*4+3]);
    for (int i = 0; i < 64; i++) gW2[i] = float4(w2[i*4], w2[i*4+1], w2[i*4+2], w2[i*4+3]);
    for (int i = 0; i < 4; i++) gB2[i] = float4(b2[i*4], b2[i*4+1], b2[i*4+2], b2[i*4+3]);
    for (int i = 0; i < 12; i++) gW3[i] = float4(w3[i*4], w3[i*4+1], w3[i*4+2], w3[i*4+3]);
    gB3[0] = float4(b3[0], b3[1], b3[2], 0.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main() {
    uint totalSamples = gSampleCount[0];
    if (totalSamples == 0) return;

    uint batchSize = min(gPC.batchSize, totalSamples);

    // Unpack weights to flat arrays
    float w1[96], b1[16], w2[256], b2[16], w3[48], b3[3];
    unpackWeights(w1, b1, w2, b2, w3, b3);

    float lr = gPC.learningRate;

    for (uint iter = 0; iter < gPC.numIterations; iter++) {
        // Gradient accumulators (initialized to 0 each iteration)
        float gw1[96], gb1[16], gw2[256], gb2[16], gw3[48], gb3[3];
        for (int i = 0; i < 96; i++) gw1[i] = 0; for (int i = 0; i < 16; i++) gb1[i] = 0;
        for (int i = 0; i < 256; i++) gw2[i] = 0; for (int i = 0; i < 16; i++) gb2[i] = 0;
        for (int i = 0; i < 48; i++) gw3[i] = 0; for (int i = 0; i < 3; i++) gb3[i] = 0;
        float totalLoss = 0;

        for (uint bi = 0; bi < batchSize; bi++) {
            // Random-ish sample selection
            uint idx = (bi * 127 + iter * 31) % totalSamples;
            uint base = idx * 9;
            float3 pos = float3(gSamples[base+0], gSamples[base+1], gSamples[base+2]);
            float3 N   = float3(gSamples[base+3], gSamples[base+4], gSamples[base+5]);
            float3 target = float3(gSamples[base+6], gSamples[base+7], gSamples[base+8]);

            // === Forward pass ===
            float h1[16], h2[16];
            float3 pred = ndgiForward(pos, N, h1, h2, w1, b1, w2, b2, w3, b3);

            // === Loss gradient (dL/dpred = pred - target for MSE) ===
            float3 dL = pred - target;
            totalLoss += dot(dL, dL);

            // === Backprop Layer 3 (16→3, ReLU) ===
            float dh2[16];
            for (int j = 0; j < 16; j++) dh2[j] = 0;
            // dL/dw3: for output k, dL/dw[j*3+k] = dL[k] * h2[j] * (pred[k] > 0 ? 1 : 0)
            float outGrad[3];
            outGrad[0] = dL.x * (pred.x > 0 ? 1.0 : 0.0);
            outGrad[1] = dL.y * (pred.y > 0 ? 1.0 : 0.0);
            outGrad[2] = dL.z * (pred.z > 0 ? 1.0 : 0.0);
            for (int j = 0; j < 16; j++) {
                gw3[j*3+0] += outGrad[0] * h2[j];
                gw3[j*3+1] += outGrad[1] * h2[j];
                gw3[j*3+2] += outGrad[2] * h2[j];
                dh2[j] += outGrad[0] * w3[j*3+0] + outGrad[1] * w3[j*3+1] + outGrad[2] * w3[j*3+2];
            }
            gb3[0] += outGrad[0]; gb3[1] += outGrad[1]; gb3[2] += outGrad[2];

            // === Backprop Layer 2 (16→16, leaky ReLU) ===
            float dh1[16];
            for (int j = 0; j < 16; j++) dh1[j] = 0;
            for (int j = 0; j < 16; j++) {
                // leaky ReLU gradient: 1.0 if > 0, else 0.01
                float actGrad = (h2[j] > 0.0) ? 1.0 : 0.01;
                float delta = dh2[j] * actGrad;
                for (int k = 0; k < 16; k++) {
                    gw2[k*16+j] += delta * h1[k];
                    dh1[k] += delta * w2[k*16+j];
                }
                gb2[j] += delta;
            }

            // === Backprop Layer 1 (6→16, leaky ReLU) ===
            float inputs[6] = {pos.x, pos.y, pos.z, N.x, N.y, N.z};
            for (int j = 0; j < 16; j++) {
                float actGrad = (h1[j] > 0.0) ? 1.0 : 0.01;
                float delta = dh1[j] * actGrad;
                for (int k = 0; k < 6; k++) {
                    gw1[k*16+j] += delta * inputs[k];
                }
                gb1[j] += delta;
            }
        }

        // === Average gradients & SGD update ===
        float invB = 1.0 / float(batchSize);
        float w1n[96], b1n[16], w2n[256], b2n[16], w3n[48]; float b3n[3];
        for (int i = 0; i < 96; i++) w1n[i] = w1[i] - lr * gw1[i] * invB;
        for (int i = 0; i < 16; i++) b1n[i] = b1[i] - lr * gb1[i] * invB;
        for (int i = 0; i < 256; i++) w2n[i] = w2[i] - lr * gw2[i] * invB;
        for (int i = 0; i < 16; i++) b2n[i] = b2[i] - lr * gb2[i] * invB;
        for (int i = 0; i < 48; i++) w3n[i] = w3[i] - lr * gw3[i] * invB;
        b3n[0]=b3[0]-lr*gb3[0]*invB; b3n[1]=b3[1]-lr*gb3[1]*invB; b3n[2]=b3[2]-lr*gb3[2]*invB;

        // EMA smoothing: w = alpha * w_old + (1-alpha) * w_new
        float a = gPC.emaAlpha;
        for (int i = 0; i < 96; i++) w1[i] = a * w1[i] + (1.0-a) * w1n[i];
        for (int i = 0; i < 16; i++) b1[i] = a * b1[i] + (1.0-a) * b1n[i];
        for (int i = 0; i < 256; i++) w2[i] = a * w2[i] + (1.0-a) * w2n[i];
        for (int i = 0; i < 16; i++) b2[i] = a * b2[i] + (1.0-a) * b2n[i];
        for (int i = 0; i < 48; i++) w3[i] = a * w3[i] + (1.0-a) * w3n[i];
        b3[0]=a*b3[0]+(1.0-a)*b3n[0]; b3[1]=a*b3[1]+(1.0-a)*b3n[1]; b3[2]=a*b3[2]+(1.0-a)*b3n[2];
    }

    // Pack updated weights back to float4 buffers
    packWeights(w1, b1, w2, b2, w3, b3);
}
