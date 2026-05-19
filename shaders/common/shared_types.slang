struct FrameUniforms {
    float4x4 view;
    float4x4 proj;
    float4x4 viewProj;
    float4x4 invViewProj;      // for deferred world-pos reconstruction from depth
    float4x4 prevViewProj;     // B.4 SSGI 时序：把当前 worldPos 投到上一帧 NDC
    float4 cameraPos;          // xyz + pad
    float4 sunDir;             // xyz + pad
    float4 sunColor_intensity; // xyz=color, w=intensity
    float4 ambient;
    int4 counts;               // x=materialCount, y=iblSpecularMips,
                               // z=iblEnabled (0/1), w=rsmEnabled (0/1)
    // M6 LPV：参数和 grid 的几何信息。
    int4   lpvCounts;          // x=gridResolution, y=lpvEnabled (0/1), z/w=pad
    float4 lpvGridMinCell;     // xyz=gridMin (worldspace), w=cellSize
    // M7 VXGI：voxel 网格几何 + 启用闸门。
    int4   vxgiCounts;         // x=gridResolution, y=vxgiEnabled, z=mipLevels, w=pad
    float4 vxgiGridMinCell;    // xyz=gridMin, w=cellSize
    // M8 PRT：visibility transfer + sun light SH（CPU 每帧投影）。
    int4   prtCounts;          // x=gridResolution, y=prtEnabled, z=useSh9 (B.9), w=pad
    float4 prtGridMinCell;     // xyz=gridMin, w=cellSize
    float4 prtLightSH_R;       // SH4 段：coefs 0..3（也是 SH9 的前 4 项）
    float4 prtLightSH_G;
    float4 prtLightSH_B;
    // B.9 SH9 扩展：l=2 的 5 个系数。R0/G0/B0 各 4 个 (Y2-2/Y2-1/Y20/Y21)，
    // R1/G1/B1 .x 是 Y22。
    float4 prtLightSH9_R0;
    float4 prtLightSH9_R1;
    float4 prtLightSH9_G0;
    float4 prtLightSH9_G1;
    float4 prtLightSH9_B0;
    float4 prtLightSH9_B1;
    // B.10 SH16 扩展：l=3 的 7 个系数。每通道 R0=(Y3-3,Y3-2,Y3-1,Y30)，
    // R1=(Y31,Y32,Y33,pad)。注：A_3=0 → SH16 对 diffuse 数学上与 SH9 等价，
    // 此扩展为运行时 toggle + 后续 specular PRT 预留基础设施。
    float4 prtLightSH16_R0;
    float4 prtLightSH16_R1;
    float4 prtLightSH16_G0;
    float4 prtLightSH16_G1;
    float4 prtLightSH16_B0;
    float4 prtLightSH16_B1;
    // M11 DDGI：probe grid 几何 + 启用闸门。
    int4   ddgiCounts;         // x=probesX, y=probesY, z=probesZ, w=ddgiEnabled
    float4 ddgiOrigin;         // xyz=grid 原点，w=pad
    float4 ddgiSpacing;        // xyz=probe 间距，w=pad
    int4   ddgiOctaSizes;      // x=octaIrr, y=octaDist, z/w=pad
};

struct MaterialGpu {
    float4 baseColorFactor;
    float3 emissiveFactor;
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;
    float alphaCutoff;
    int baseColorTex, mrTex, normalTex, occlusionTex, emissiveTex;
    uint alphaMode;
    uint doubleSided;
    uint _pad0;
};

struct PushConsts {
    float4x4 model;
    int materialIndex;
    int _p0, _p1, _p2;
};

// M5 RSM 专用的 per-frame UBO（小 + 仅 sun 光参数 + sun 视角 ortho viewProj）。
// 由 RsmGeometryPass 自己维护，与主 FrameUniforms 解耦：sun 视角是独立
// 相机，主 camera 跟它无关；分开避免污染 FrameUBO 字段。
struct RsmFrameUniforms {
    float4x4 sunViewProj;      // 用于把 worldPos 投到 sun light space（[0,1] depth）
    float4x4 sunView;          // sun "camera" view（rsm_geometry vs 用 sunViewProj 直接到 clip）
    float4 sunDir;             // xyz = 朝 sun 的方向（与主 FrameUBO 一致，从 surface 到 sun）
    float4 sunColor_intensity; // rgb=color, w=intensity（与主一致）
};
