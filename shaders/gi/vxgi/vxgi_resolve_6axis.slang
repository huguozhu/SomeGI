// L.3b 6-axis directional resolve —— 读 isotropic voxelGrid → 写 3 张主轴
// radiance image（+X/+Y/+Z）。每 voxel 朝 6 主轴 cone-trace，各方向结果
// 写到对应 axis image。
//
// axisX.rgb = cone trace in +X direction result
// axisY.rgb = cone trace in +Y direction result
// axisZ.rgb = cone trace in +Z direction result
// axisX.a = cone trace in -X direction (luminance)
// axisY.a = cone trace in -Y direction (luminance)
// axisZ.a = cone trace in -Z direction (luminance)
//
// Dispatch: (res/4, res/4, res/4) groups of 4³

[[vk::binding(0, 0)]] Texture3D<float4>   gVoxelGrid;     // isotropic src
[[vk::binding(1, 0)]] Texture3D<float4>   gVxgiAniso;     // directional alpha
[[vk::binding(2, 0)]] SamplerState        gLinearClamp;
[[vk::binding(3, 0)]] RWTexture3D<float4> gAxisX;         // dst +X rgb, -X lum(a)
[[vk::binding(4, 0)]] RWTexture3D<float4> gAxisY;         // dst +Y rgb, -Y lum(a)
[[vk::binding(5, 0)]] RWTexture3D<float4> gAxisZ;         // dst +Z rgb, -Z lum(a)

struct SixAxisPC {
    uint   gridResolution;
    uint   mipLevels;
    float  cellSize;
    float  strength;
    float3 gridMin;
    float  _pad0;
};
[[vk::push_constant]] SixAxisPC gPC;

static const float kLumWeights[3] = { 0.299, 0.587, 0.114 };

// 6 cone directions and which axis/channel they write to
static const int3 kDirs[6] = {
    int3(1,0,0), int3(-1,0,0),
    int3(0,1,0), int3(0,-1,0),
    int3(0,0,1), int3(0,0,-1),
};
// target: 0=axisX, 1=axisY, 2=axisZ; channel: 0=rgb, 1=a(luminance)
static const int kTargetAxis[6]     = { 0, 0, 1, 1, 2, 2 };
static const int kTargetChan[6]     = { 0, 1, 0, 1, 0, 1 };

float3 traceCone(float3 origin, float3 dir, float aperture) {
    float3 acc = float3(0);
    float trans = 1.0;
    float dist = gPC.cellSize * 1.5;
    const int kSteps = 12;
    for (int s = 0; s < kSteps; ++s) {
        float3 pos = origin + dir * dist;
        float3 uv = (pos - gPC.gridMin) / (float(gPC.gridResolution) * gPC.cellSize);
        if (any(uv < 0.0) || any(uv > 1.0)) break;
        float diameter = max(gPC.cellSize, 2.0 * aperture * dist);
        float mip = clamp(log2(diameter / gPC.cellSize), 0.0, float(gPC.mipLevels - 1));
        float4 c = gVoxelGrid.SampleLevel(gLinearClamp, uv, mip);
        float a = c.a;
        if (mip >= 1.0) {
            float4 ani = gVxgiAniso.SampleLevel(gLinearClamp, uv, mip);
            float3 wdir = abs(dir);
            float aDir = max(wdir.x * ani.r, max(wdir.y * ani.g, wdir.z * ani.b));
            a = max(a, aDir);
        }
        acc += trans * c.rgb * a;
        trans *= max(0.0, 1.0 - a);
        if (trans < 0.01) break;
        dist += diameter * 0.5;
    }
    return acc;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void cs_resolve6Axis(uint3 dt : SV_DispatchThreadID) {
    if (any(dt >= uint3(gPC.gridResolution, gPC.gridResolution, gPC.gridResolution))) return;

    float4 self = gVoxelGrid.Load(int4(int3(dt), 0));
    if (self.a < 0.01) {
        gAxisX[dt] = float4(0); gAxisY[dt] = float4(0); gAxisZ[dt] = float4(0);
        return;
    }

    float3 cellCenter = gPC.gridMin + (float3(dt) + 0.5) * gPC.cellSize;

    float4 result[3] = { float4(0), float4(0), float4(0) };

    [unroll]
    for (int i = 0; i < 6; ++i) {
        float3 dir = float3(kDirs[i]);
        float3 ori = cellCenter + dir * (gPC.cellSize * 0.75);
        float3 gathered = traceCone(ori, dir, 0.5) * gPC.strength;

        // result = self.rgb + gathered (one bounce of indirect)
        float3 coneRgb = self.rgb + gathered;

        int ax = kTargetAxis[i];
        if (kTargetChan[i] == 0) {
            // positive axis: store full RGB
            result[ax].rgb = coneRgb;
        } else {
            // negative axis: store luminance
            result[ax].a = dot(coneRgb, float3(kLumWeights[0], kLumWeights[1], kLumWeights[2]));
        }
    }

    gAxisX[dt] = result[0];
    gAxisY[dt] = result[1];
    gAxisZ[dt] = result[2];
}
