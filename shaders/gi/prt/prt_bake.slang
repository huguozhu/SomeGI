// PRT bake compute —— 烘焙 order-2 SH9（9 系数）visibility transfer。
//
// 数学：v_k(x) = (4π/N) · Σ_i V(x, d_i) · Y_k(d_i)，k=0..8
//
// 9 系数分到 3 张 RGBA16F atlas：
//   prtTransfer  RGBA = (Y00, Y1-1, Y10, Y11)
//   prtTransferB RGBA = (Y2-2, Y2-1, Y20, Y21)
//   prtTransferC R    = Y22; GBA 暂未用
//
// SH4 模式 lighting 仅读 prtTransfer；SH9 模式读全部 3 张。bake 永远
// 输出 SH9（一次性算量小 + 切换不需要 re-bake）。

[[vk::binding(0, 0)]] Texture3D<float4>   gVoxelGrid;
[[vk::binding(1, 0)]] SamplerState        gLinearClamp;
[[vk::binding(2, 0)]] RWTexture3D<float4> gPrtTransfer;   // coefs 0..3   (SH4)
[[vk::binding(3, 0)]] RWTexture3D<float4> gPrtTransferB;  // coefs 4..7   (SH9)
[[vk::binding(4, 0)]] RWTexture3D<float4> gPrtTransferC;  // coef  8      (SH9)
[[vk::binding(5, 0)]] RWTexture3D<float4> gPrtTransferD;  // coefs 9..12  (SH16)
[[vk::binding(6, 0)]] RWTexture3D<float4> gPrtTransferE;  // coefs 13..15 (SH16)

struct BakePC {
    float3 prtGridMin;
    float  prtCellSize;
    uint   prtResolution;
    uint   numSamples;
    float  voxelGridDim;
    uint   _pad0;
    float3 vxgiGridMin;
    float  vxgiCellSize;
    uint   vxgiResolution;
    uint   _pad1, _pad2, _pad3;
};
[[vk::push_constant]] BakePC gPC;

float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float3 hammersleySphere(uint i, uint N) {
    float z = 1.0 - 2.0 * (float(i) + 0.5) / float(N);
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 6.2831853 * radicalInverse_VdC(i);
    return float3(r * cos(phi), r * sin(phi), z);
}

float traceVisibility(float3 origin, float3 d) {
    float stepLen = gPC.vxgiCellSize * 1.5;
    int maxSteps = 24;
    float trans = 1.0;
    for (int s = 1; s <= maxSteps; ++s) {
        float3 pos = origin + d * (stepLen * float(s));
        float3 uv = (pos - gPC.vxgiGridMin)
                  / (float(gPC.vxgiResolution) * gPC.vxgiCellSize);
        if (any(uv < 0.0) || any(uv > 1.0)) break;
        float a = gVoxelGrid.SampleLevel(gLinearClamp, uv, 0).a;
        trans *= max(0.0, 1.0 - a);
        if (trans < 0.05) return 0.0;
    }
    return trans;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    if (any(dt >= uint3(gPC.prtResolution, gPC.prtResolution, gPC.prtResolution))) return;

    float3 cellCenter = gPC.prtGridMin
                      + (float3(dt) + 0.5) * gPC.prtCellSize;

    // 9 个累加器
    float4 sum0 = float4(0);   // coefs 0..3
    float4 sum1 = float4(0);   // coefs 4..7
    float  sum8 = 0.0;          // coef 8
    for (uint i = 0; i < gPC.numSamples; ++i) {
        float3 d = hammersleySphere(i, gPC.numSamples);
        float V = traceVisibility(cellCenter, d);
        if (V <= 0.0) continue;

        float x = d.x, y = d.y, z = d.z;

        // Order-1 (l=0,1)
        float Y0   = 0.282094792;
        float Y1n1 = 0.488602512 * y;
        float Y10  = 0.488602512 * z;
        float Y11  = 0.488602512 * x;
        // Order-2 (l=2)
        float Y2n2 = 1.092548431 * x * y;
        float Y2n1 = 1.092548431 * y * z;
        float Y20  = 0.315391565 * (3.0 * z * z - 1.0);
        float Y21  = 1.092548431 * z * x;
        float Y22  = 0.546274215 * (x * x - y * y);

        sum0 += V * float4(Y0,   Y1n1, Y10, Y11);
        sum1 += V * float4(Y2n2, Y2n1, Y20, Y21);
        sum8 += V * Y22;
    }

    // 4π / N 归一化
    float norm = 12.566370614 / float(gPC.numSamples);
    sum0 *= norm;
    sum1 *= norm;
    sum8 *= norm;

    gPrtTransfer [dt] = sum0;
    gPrtTransferB[dt] = sum1;
    gPrtTransferC[dt] = float4(sum8, 0.0, 0.0, 0.0);
}
