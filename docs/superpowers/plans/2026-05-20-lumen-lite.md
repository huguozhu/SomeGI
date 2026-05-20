# L 阶段：Lumen-lite 实现计划

> **日期**：2026-05-20
>
> **目标**：在现有 TLAS（SceneRtAS）+ VXGI voxel grid + DDGI probe 基础设施上，
> 搭建 Lumen-lite 混合 GI 管线 —— 屏幕 probe（近）+ world probe（远）fallback，
> specular 单独 HW RT。
>
> **实施顺序**：L.2 → L.5 → L.4 → L.3（来自建议排序：先闭 loop 看到画面，再逐步打磨）。
>
> **GI 下拉**：Lumen-lite 作为第 13 项（index 12），与现有 12 项独立并存。

---

## 前置依赖（已满足）

| 依赖 | 状态 |
|------|------|
| TLAS（SceneRtAS）| M9/M10 已构建，支持 VK_KHR_acceleration_structure + ray_query |
| VXGI voxel grid（VxgiResources）| M7 已实现，128³ RGBA16F + mipchain + 各向异性 alpha |
| DDGI world probes（DdgiResources）| M11 已实现，8×4×8 probes + octahedral atlas |
| lighting.slang GI 切换 | M3-M11 已验证，12 档 GI 切换无 crash |
| Spherical Fibonacci + octahedral | DDGI shader 已有，L.2/L.4 直接 import |
| Screen probe 分辨率计算 | 复用 DDGI 的 probe index → atlas texel 反查逻辑 |

---

## 资源概览（Lumen-lite 新增）

```
LumenResources (lumen_resources.h)
├── m_probeAtlas        RGBA16F, (screenW/16 * 9) × (screenH/16)
│                       每 probe 存 9 个 SH coefficient（SH3 = 4 bands）
│                       layout: [probeGridW * 9, probeGridH] 2D atlas
├── m_rayBuffer         SSBO, probeCount × raysPerProbe × 8 floats
│                       同 DDGI ray buffer 格式: dir(3)+pad, hitRgb(3)+hitDist
├── m_filteredAtlas     RGBA16F, 同上尺寸（spatial filter 输出，ping-pong）
├── m_prevProbeAtlas    RGBA16F, 同上尺寸（temporal 用，D.4 阶段暂不启用）
└── m_probeDispatchBuffer   SSBO, per-probe flags (active/skip) 调试用

Lumen constants（默认值）:
  kProbeGridTile  = 16       // 屏幕空间每 16² 像素一个 probe
  kRaysPerProbe   = 32       // Phase L1 每帧每 probe 射线数（后续升 64）
  kSHBands        = 3        // SH9 = 9 coefficients
  kAtlasChannels  = 9        // SH coefs × 3 channels = 27 floats → 拆成 3 层 RGBA16F
```

屏幕 probe grid 维度随 swapchain resize 变化：
```
probeGridW = ceil(screenW / kProbeGridTile)
probeGridH = ceil(screenH / kProbeGridTile)
probeCount = probeGridW × probeGridH
```

### 显存估算（1920×1080）

| 资源 | 大小 |
|------|------|
| Probe atlas（3 layers × RGBA16F）| 120 × 68 × 9 ch / 3 layers × 8B ≈ 65 KB |
| Ray buffer | 120×68 × 32 rays × 32B ≈ 8.0 MB |
| Filtered atlas（ping）| ≈ 65 KB |
| Prev atlas（temporal, deferred）| ≈ 65 KB |
| **总计** | **~9 MB** |

---

## L.2 Screen Probe Pass（3 ~ 4 天）

### 目标

屏幕 16×16 grid 每 probe 朝 cosine 半球投 32 ray via TLAS → 命中点取
voxelGrid radiance → SH9 投影 → 输出 probe atlas。

### L.2.1 LumenResources 类

**File**: `src/renderer/lumen_resources.{h,cpp}`

```cpp
class LumenResources {
public:
    void create(Device& d, VkExtent2D screenExt);
    void destroy();

    const Image& probeAtlas() const;     // 3-layer RGBA16F
    const Image& filteredAtlas() const;  // spatial filter 输出（L.4）
    const Buffer& rayBuffer() const;

    uint32_t probeGridW() const;
    uint32_t probeGridH() const;
    uint32_t probeCount() const;

private:
    // atlas 是 3 层 2D array（SH 9 系数 × 3 通道 = 27 float → 拆 3 层）：
    Image m_probeAtlas;       // RGBA16F array × 3
    Image m_filteredAtlas;    // RGBA16F array × 3（L.4 输出）
    Image m_prevProbeAtlas;   // RGBA16F array × 3（D.4 temporal 用，暂不分配）
    Buffer m_rayBuffer;
    uint32_t m_probeGridW = 0;
    uint32_t m_probeGridH = 0;
};
```

设计理由：
- 3 层 2D array 而非 27-channel 单层：Vulkan 保证 2D array 的 layer count ≤ 2048
- 与 DDGI atlas 共享 octahedral 工具函数
- Ray buffer 格式与 DDGI 完全一致，shader 端 import 同一结构定义

### L.2.2 lumen_probe.slang

**File**: `shaders/gi/lumen/lumen_probe.slang`

```
import common/frame;
import common/scene;
import common/octahedral;

// Bindings
[[vk::binding(0, 0)]] RaytracingAccelerationStructure gTLAS;
[[vk::binding(1, 0)]] StructuredBuffer<RtInstanceData>  gInstances;
[[vk::binding(2, 0)]] StructuredBuffer<Vertex>          gVertices;
[[vk::binding(3, 0)]] StructuredBuffer<uint>            gIndices;
[[vk::binding(4, 0)]] StructuredBuffer<MaterialGpu>     gMaterials;
[[vk::binding(5, 0)]] Texture2D<float4>                 gAlbedoMetal;
[[vk::binding(6, 0)]] Image2D<float2>                   gNormalRough;
[[vk::binding(7, 0)]] Image2D<float>                    gDepth;
[[vk::binding(8, 0)]] Texture3D<float4>                 gVoxelGrid;  // VXGI mipchain
[[vk::binding(9, 0)]] SamplerState                      gLinearClamp;
[[vk::binding(10,0)]] RWStructuredBuffer<float4>        gRayBuf;
[[vk::binding(11,0)]] RWTexture2DArray<float4>          gProbeAtlas;

struct ProbePC {
    float2 screenSize;
    float2 invScreenSize;
    uint   probeGridW, probeGridH;
    uint   probeTileSize;
    uint   raysPerProbe;
    float  randomSeed;
    // camera
    float4x4 viewProjInv;
    float3   cameraPos;
    // voxel grid
    float3   vxgiGridMin;
    float    vxgiCellSize;
    uint     vxgiResolution;
};

[numthreads(8, 8, 1)]
void cs_generateProbes(uint3 dt : SV_DispatchThreadID) {
    // dt.x = probeIndex, dt.y = rayIndex
    uint probeIdx = dt.x;
    if (probeIdx >= gPC.probeGridW * gPC.probeGridH) return;

    // 1. probe 屏幕中心 worldPos（从 probe grid 坐标反推）
    uint px = probeIdx % gPC.probeGridW;
    uint py = probeIdx / gPC.probeGridW;
    float2 screenUV = (float2(px + 0.5, py + 0.5) * gPC.probeTileSize) * gPC.invScreenSize;
    // 从 depth buffer 取该 tile 中心的深度 → worldPos
    float depth = gDepth[uint2(screenUV * gPC.screenSize)];
    float3 probeWorldPos = screenToWorld(screenUV, depth, gPC.viewProjInv);

    // 2. cosine 半球采样方向（spherical Fibonacci + per-frame rotation）
    float3 rayDir = sphericalFibonacci(dt.y, gPC.raysPerProbe);
    rayDir = rotateY(rayDir, gPC.randomSeed);
    // 对齐到 probe 法线（从 GBuffer 取 tile 中心 N）
    float3 N = gNormalRough[uint2(screenUV * gPC.screenSize)].xyz;
    float3 up = abs(N.y) < 0.999 ? float3(0,1,0) : float3(1,0,0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    rayDir = rayDir.x * T + rayDir.y * B + rayDir.z * N;

    // 3. TLAS ray query
    RayQuery<...> q;
    init(q, gTLAS, probeWorldPos, 0.01, rayDir * 100.0);
    proceed(q);

    float3 hitRadiance = float3(0);
    float hitDist = 1e6;
    if (committed(q)) {
        // 命中 → 查 GBuffer 材质 → 取 voxel grid radiance
        hitPos = worldRayOrigin() + worldRayDirection() * committedRayT(q);
        // 对命中点位置在 voxel grid 中采样 → 间接光 radiance
        float3 voxelUv = (hitPos - gPC.vxgiGridMin) / (gPC.vxgiCellSize * gPC.vxgiResolution);
        hitRadiance = gVoxelGrid.SampleLevel(gLinearClamp, voxelUv, 2).rgb;
        hitDist = committedRayT(q);
    }

    // 4. SH9 投影（at probe index offset in ray buffer）
    // 先写到 ray buffer（每个 ray 8 floats: dir, hitRgb+dist）
    uint raySlot = probeIdx * gPC.raysPerProbe + dt.y;
    gRayBuf[raySlot * 2 + 0] = float4(rayDir, 0);
    gRayBuf[raySlot * 2 + 1] = float4(hitRadiance, hitDist);
}
```

`cs_projectSH`（第二个 entry point 或后续 dispatch）：

```
// 读 ray buffer → 每个 probe 的 N rays → 累积 SH9 coefficients
// dispatch (probeCount, 1, 1)，每 thread 内部循环 N rays
[numthreads(64, 1, 1)]
void cs_projectSH(uint3 dt : SV_DispatchThreadID) {
    uint probeIdx = dt.x;
    if (probeIdx >= gPC.probeGridW * gPC.probeGridH) return;

    // 累积 SH9（9 coeffs × 3 channels = 27 values）
    float shR[9], shG[9], shB[9];
    // zero init...

    for (uint r = 0; r < gPC.raysPerProbe; r++) {
        uint slot = probeIdx * gPC.raysPerProbe + r;
        float3 dir = gRayBuf[slot * 2 + 0].xyz;
        float3 rad = gRayBuf[slot * 2 + 1].xyz;
        float dist = gRayBuf[slot * 2 + 1].w;

        if (dist >= 1e5) continue; // miss

        // cosine weight
        float cosW = max(0.0, dot(dir, N_probe));
        // SH basis eval → 9 scalars
        float shBasis[9];
        evalSH9(dir, shBasis);
        for (int c = 0; c < 9; ++c) {
            float w = shBasis[c] * cosW;
            shR[c] += rad.r * w;
            shG[c] += rad.g * w;
            shB[c] += rad.b * w;
        }
        weightSum += cosW;
    }

    // Normalize → 写 probe atlas
    float invW = weightSum > 0 ? (4.0 * PI / weightSum) : 0;
    // 6 coeffs per layer；layer 0: sh[0..5], layer 1: sh[6..11], etc
    uint atlasX = (probeIdx % gPC.probeGridW) * 9;
    uint atlasY = probeIdx / gPC.probeGridW;

    // 每个 SH coeff 一个 texel column；用 uint2 逐 coeff 写
    for (int c = 0; c < 9; ++c) {
        float3 writeRgb = float3(shR[c], shG[c], shB[c]) * invW;
        uint layer = c / 3;   // 0, 1, 2
        uint ch  = c % 3;     // 0=R, 1=G, 2=B
        // 写浮动 RGBA：每层存 3 个 SH coeff 的 (R, G, B) → 需要 3×3=9 个 float per layer
        // 简化：3 layers × 3 coefs，layer l 存 SH[l*3] / SH[l*3+1] / SH[l*3+2]
        // 方式：atlas texel (atlasX+c, atlasY) 写到 layer 的 .r = SH[l*3+ch].rgb 不对
        // 实际：用 RGBA 的 rgb 存一个 coeff 的 rgb，4 个 coeffs/layer 需要 4 个 texels
        // 更简单：用 uint2(atlasX/4, atlasY) 存 4 coeffs/layer，
        // 用 atlasX%4 选当前 texel 存哪个 coeff
    }
}
```

**SH 投影设计细节**（关键决策）：

SH9 atlas 存储方案 —— 选"per-coeff 单 texel，3 layers × 3 coeffs/layer"：

```
atlas: RGBA16F 2D array[3], size = (probeGridW * 3, probeGridH)

Layer 0: coeff 0(RGB in .rgb), coeff 1(.rgb), coeff 2(.rgb)
         每个 coeff 占 1 个 atlas texel，共 3 texels/probe = probeGridW*3 宽
Layer 1: coeff 3, 4, 5
Layer 2: coeff 6, 7, 8

atlas lookup: (probeX * 3 + coeffIdx, probeY, layer)
```

这是最直白的映射，每个 SH coeff 在 atlas 里占一个独立 RGBA16F texel（.rgb = coeff vector,
.a = 未使用）。非 quad 对齐，但 compute shader storage write 不需要 sampler-compatible。

### L.2.3 LumenProbePass 类

**File**: `src/renderer/lumen_probe_pass.{h,cpp}`

```cpp
class LumenProbePass {
public:
    void init(Device& d);
    void destroy();

    // bind：TLAS + scene SSBO + voxelGrid + GBuffer + probeAtlas + rayBuf
    void bindResources(Device& d, const LumenResources& res,
                       const SceneRtAS& rtAS, const VxgiResources& vxgi,
                       const RenderTargets& rt, VkBuffer frameUbo);

    void record(VkCommandBuffer cmd, uint32_t probeGridW, uint32_t probeGridH,
                uint32_t raysPerProbe, float randomSeed, uint32_t frameIndex);

private:
    Device* m_device = nullptr;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
};
```

dispatch：
- `cs_generateProbes`: (probeCount * raysPerProbe + 63) / 64 × 64 groups（64 线程/group）
  或每个 group 处理若干 probes 的若干 rays
- `cs_projectSH`: (probeCount + 63) / 64 × 64 groups

先做两 dispatch 串联，后面可以融合到一个 shader 加 shared memory（64-thread group 内
shared 累积 SH 再 atomic 回 global）。

---

## L.5 Final Gather Pass（2 ~ 3 天）

### 目标

每像素从 screen probe atlas 采样邻居 4 个 probe → bilinear SH 插值 → 半球 cosine
积分重建 diffuse radiance → 在 lighting.slang 中作为新的 indirect 通道输出。

### L.5.1 lumen_gather.slang

**File**: `shaders/gi/lumen/lumen_gather.slang`

```
import common/frame;
import common/sh;

[[vk::binding(0, 0)]] Texture2DArray<float4> gProbeAtlas;
[[vk::binding(1, 0)]] SamplerState           gPointClamp;
[[vk::binding(2, 0)]] Texture2D<float4>       gAlbedoMetal;
[[vk::binding(3, 0)]] Texture2D<float2>       gNormalRough;
[[vk::binding(4, 0)]] Texture2D<float>        gDepth;
[[vk::binding(5, 0)]] RWTexture2D<float4>     gLumenGI;  // 输出 RGBA16F

struct GatherPC {
    float2 invScreenSize;
    float  probeTileSize;
    uint   probeGridW, probeGridH;
    uint   atlasChannels;
    uint   shBands;
    float4x4 viewProjInv;
};

[numthreads(8, 8, 1)]
void cs_gather(uint3 dt : SV_DispatchThreadID) {
    float2 uv = (float2(dt.xy) + 0.5) * gPC.invScreenSize;

    // 1. 世界坐标
    float depth = gDepth[dt.xy];
    float3 worldPos = screenToWorld(uv, depth, gPC.viewProjInv);
    float3 N = gNormalRough[dt.xy].xyz;

    // 2. bilinear 采相邻 4 个 screen probe
    float2 probeCoord = float2(dt.xy) / gPC.probeTileSize - 0.5;
    int2 baseProbe = int2(floor(probeCoord));
    float2 frac = frac(probeCoord);
    baseProbe = clamp(baseProbe, 0, int2(gPC.probeGridW - 1, gPC.probeGridH - 1));

    float shR[9] = {0}, shG[9] = {0}, shB[9] = {0};
    float totalW = 0;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
            int2 probeIdx = clamp(baseProbe + int2(dx, dy), 0,
                                 int2(gPC.probeGridW - 1, gPC.probeGridH - 1));
            float w = (dx == 0 ? 1 - frac.x : frac.x) *
                      (dy == 0 ? 1 - frac.y : frac.y);

            for (int c = 0; c < 9; ++c) {
                uint layer = c / 3;
                uint offset = c % 3;
                float3 coeff = gProbeAtlas[uint3(probeIdx.x * 9 + offset,
                                                  probeIdx.y, layer)].rgb;
                shR[c] += coeff.r * w;
                shG[c] += coeff.g * w;
                shB[c] += coeff.b * w;
            }
            totalW += w;
        }
    }
    // normalize weight
    for (int c = 0; c < 9; ++c) { shR[c] /= totalW; shG[c] /= totalW; shB[c] /= totalW; }

    // 3. SH9 → 入射 radiance（diffuse cosine convolution）
    // 用 SH 卷积系数：l=0: π, l=1: 2π/3, l=2: π/4, l=3: 0
    float3 irradiance = shConvolveDiffuse(shR, shG, shB, int3(gPC.shBands));

    // 4. 乘 albedo / π → 出射 diffuse
    float3 albedo = gAlbedoMetal[dt.xy].rgb;
    float3 indirectDiffuse = irradiance * albedo * INV_PI;

    gLumenGI[dt.xy] = float4(indirectDiffuse, 1.0);
}
```

### L.5.2 LumenGatherPass 类

**File**: `src/renderer/lumen_gather_pass.{h,cpp}`

```cpp
class LumenGatherPass {
public:
    void init(Device& d);
    void destroy();
    void bindResources(Device& d, const LumenResources& res,
                       const RenderTargets& rt, VkBuffer frameUbo);
    void record(VkCommandBuffer cmd, uint32_t probeGridW, uint32_t probeGridH,
                float probeTileSize);

private:
    Device* m_device = nullptr;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
};
```

### L.5.3 新增 RenderTargets 通道

在 `RenderTargets` 中加一个：

```cpp
Image lumenGI;     // RGBA16F: Lumen-lite 最终间接 diffuse
```

大小同 swapchain。

### L.5.4 lighting.slang 集成

在 lighting.slang 的间接光计算段，当 `frame.flags` 指示 Lumen 模式时：

```slang
// 不再走 IBL/SSGI/VXGI 等单独的 indirect 分支，而是直接 += lumenGI
if (frame.flags & FLAG_LUMEN) {
    diffuse += gLumenGI.Sample(gLinearClamp, uv).rgb;
}
```

### L.5.5 GI 下拉注册

在 `kGis[]` 加第 13 项：

```cpp
{"Lumen-lite", true},  // index 12
```

配置 Lumen 时需要：
1. TLAS 已构建（SceneRtAS）
2. VXGI voxelGrid 已 voxelize+mipmap
3. VXGI 仍需每帧跑（提供 voxelGrid radiance 给 probe pass 采样）
4. DDGI 不冲突（可选远距离 fallback，L.6 才接）

---

## L.4 Probe Filtering（2 ~ 3 天）

### 目标

消除 screen probe 间的空间不连续性（邻 probe 间亮度跳变）和闪烁（帧间噪声）。

Phase L1 只做 **spatial**（邻 probe SH 融合 + normal/depth weight）。Temporal 保留到
L4+（需要 reproject 上帧 probe atlas）。

### L.4.1 lumen_filter.slang

**File**: `shaders/gi/lumen/lumen_filter.slang`

Spatial 部分（cross-bilateral on SH coefficients）：

```
[numthreads(8, 8, 1)]
void cs_spatialFilter(uint3 dt : SV_DispatchThreadID) {
    uint probeIdx = dt.x;  // flattened probe index
    if (probeIdx >= gPC.probeGridW * gPC.probeGridH) return;

    // 取中心 probe 的 worldPos（从 depth + probe grid 坐标反算）
    uint px = probeIdx % gPC.probeGridW;
    uint py = probeIdx / gPC.probeGridW;
    float2 centerUV = (float2(px + 0.5, py + 0.5) * gPC.probeTileSize) * gPC.invScreenSize;
    float3 centerPos = screenToWorld(centerUV, gDepth, gPC.viewProjInv);
    float3 centerN = gNormalRough[uint2(centerUV * gPC.screenSize)].xyz;

    // 遍历 3×3 邻域 probe
    float filteredR[9] = {0}, filteredG[9] = {0}, filteredB[9] = {0};
    float totalW = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 neighbor = int2(px + dx, py + dy);
            if (neighbor.x < 0 || neighbor.x >= gPC.probeGridW ||
                neighbor.y < 0 || neighbor.y >= gPC.probeGridH) continue;
            uint nIdx = neighbor.y * gPC.probeGridW + neighbor.x;

            // 取 neighbor 的 worldPos/N
            float2 nUV = (float2(neighbor) + 0.5) * gPC.probeTileSize * gPC.invScreenSize;
            float3 nPos = screenToWorld(nUV, gDepth, gPC.viewProjInv);
            float3 nN = gNormalRough[uint2(nUV * gPC.screenSize)].xyz;

            // weight: depth similarity × normal similarity × distance
            float depthW = exp(-abs(dot(centerN, centerPos) - dot(nN, nPos)) / gPC.sigmaDepth);
            float normalW = pow(max(0.0, dot(centerN, nN)), gPC.normalPower);  // ~32
            float distW = 1.0 / (1.0 + length(nPos - centerPos));
            float w = depthW * normalW * distW;

            // 读 neighbor SH coefs from probe atlas
            float nR[9], nG[9], nB[9];
            readProbeSH(nIdx, nR, nG, nB);
            for (int c = 0; c < 9; ++c) {
                filteredR[c] += nR[c] * w;
                filteredG[c] += nG[c] * w;
                filteredB[c] += nB[c] * w;
            }
            totalW += w;
        }
    }
    // normalize → write filtered atlas
    for (int c = 0; c < 9; ++c) {
        filteredR[c] /= totalW;
        filteredG[c] /= totalW;
        filteredB[c] /= totalW;
    }
    writeProbeSH(probeIdx, filteredR, filteredG, filteredB, gFilteredAtlas);
}
```

Temporal 部分（deferred）：

与 SSGI temporal（B.4）完全同模式——
`probeWorldPos → reproject 上帧 NDC → 取上帧 probe SH → α-blend`。
Lumen 的 reproject 目标是 probe atlas 层面（不是 per-pixel），可以复用
`m_prevViewProj`。

### L.4.2 LumenFilterPass 类

```cpp
class LumenFilterPass {
public:
    void init(Device& d);
    void destroy();
    void bindResources(Device& d, const LumenResources& res,
                       const RenderTargets& rt, VkBuffer frameUbo);
    void record(VkCommandBuffer cmd, const FilterParams& params);
};
```

spatial: 每帧跑一次
temporal: 先用 1.0 alpha（全用 spatial），保留接口位

### L.4.3 集成点

- L.2 probe pass 写 `m_probeAtlas`
- L.4 filter pass 读 `m_probeAtlas` → 写 `m_filteredAtlas`
- L.5 gather pass 改为读 `m_filteredAtlas`（而非 `m_probeAtlas`）

首版可跳过 L.4：L.5 直接读 raw probe atlas（有 discontinuity 但能看到画面）。
后续加上 L.4 filter 只换 atlas binding，不改 gather 逻辑。

---

## L.3 Voxel Surface Cache（5 ~ 7 天）

### 目标

把 VXGI voxel grid 从 isotropic radiance 升级为 **6-axis directional radiance cache** +
**多轮 iterative relight**，让它充当 Lumen surface card 的简化替代。

### LL.3.1 6-axis voxel storage

**现状**：VXGI voxel mip 0 是一个 RGBA16F（rgb=radiance, a=opacity）—— 各向同性，
每个 voxel 只能存一个颜色。

**目标**：存储 6 个主轴方向（±X, ±Y, ±Z）的 radiance。改用 3 张 RGBA16F 图（每张存
2 个方向，rgba 的高 16 位 / 低 16 位或每张 2 层）。

设计：**3 层 RGBA16F**，每层存 2 个方向：
```
Layer 0: rgba = (dir+X.rgb, dir-X.r)
Layer 1: rgba = (dir+Y.rgb, dir-Y.r)
Layer 2: rgba = (dir+Z.rgb, dir-Z.r)
```

方向采样时：按视线方向 d 的 sign 和 dominant axis 选层 + 分量。

总大小：128³ × 3 layers × 8B = 48 MB（vs 当前单层 16 MB）。

### LL.3.2 多轮 iterative relight

**现状**（C.2）：单轮 relight —— voxelGrid mip 0 → cone trace → relightScratch → copy
回 mip 0。每帧只跑一跳间接反弹。

**目标**：每帧跑 N 轮（建议 N=3 ~ 5），在 relight pass 内 ping-pong 累积多 bounce。

方案：
```
Frame N:
  for bounce = 1..kRelightPasses:
    srcVoxel = (bounce == 1) ? vxgiMip0 : relightPong
    dstVoxel = (bounce == kRelightPasses) ? relightScratch : relightPing
    coneTrace from srcVoxel → accumulate into dstVoxel (6-axis write)
    swap ping/pong
  vkCmdCopyImage dst → voxelGrid mip0（最后 bounce 已累积所有前序）
```

Shader 改动：
- `vxgi_relight.slang` 改成 6-axis write：
  对每个 voxel，cone trace N directions（6 主轴或 random hemisphere），
  结果按 direction 的 dominant axis 投影到对应层/分量
- 6-axis read 在 L.2 probe pass 中：
  probe ray 命中 voxel 后，按 rayDir 选最近主轴（dot product max），
  取该轴对应的 radiance

### LL.3.3 cone trace 适配 6-axis

lit surfel（voxel 辐射源）读 6-axis 存储：
```
float3 sample6Axis(uint3 coord, float3 d) {
    // 按 d 的 dominant axis + sign 选 layer
    float adx = abs(d.x), ady = abs(d.y), adz = abs(d.z);
    if (adx >= ady && adx >= adz) {
        uint layer = (d.x >= 0) ? 0 : 3; // offset inside layer
        return gVoxelSixAxis[uint3(coord.x, coord.y, coord.z)][layer].rgb;
    } else if (ady >= adz) {
        uint layer = (d.y >= 0) ? 1 : 4;
        ...
    } else { ... }
}
```

cone trace 中 scattered voxel（入射光累积 target）同样按 6-axis 写。

### LL.3.4 兼容现有 VXGI pipeline

6-axis 升级是 **VXGI 基础设施的增强**，不影响其他 GI 模式：
- VXGI 模式：现有单层 isotropic 不变（M7 已验证）
- Lumen 模式：走 6-axis voxel + 多轮 relight（L.3 专属）
- 切换 GI 时 `LumenResources` 的 extra 3-layer voxel 只在 Lumen 模式分配

实现策略：在 `VxgiResources` 中 **新增** 6-axis extra（不与现有 `m_image` 混合）：
```cpp
// 仅在 Lumen 模式分配
Image m_sixAxisImage;  // RGBA16F 2D array[6] or 3-layer image with 2 dirs/layer
std::vector<VkImageView> m_sixAxisMipViews;
```

relight pass 在 Lumen 模式走 6-axis 分支（加 `#define LUMEN_SIX_AXIS 1` 宏变体编译）。

---

## 帧序列（Lumen 模式）

```
FrameStart
 ├─ Camera.update
 ├─ GBufferPass (graphics)
 ├─ VXGI: voxelize → injectSun → mipmap (compute, same as M7)
 ├─ VXGI: relight multi-bounce (L.3 升级版，6-axis write, compute)
 ├─ [L.7 deferred] Lumen Reflections (specular ray query)
 ├─ L.2 LumenProbePass: generateRays → projectSH (compute)
 ├─ L.4 LumenFilterPass: spatialFilter (compute)
 ├─ L.5 LumenGatherPass: cs_gather → lumenGI (compute)
 ├─ LightingPass: direct sun + Lumen indirect (compute, reads lumenGI)
 ├─ ForwardPass (transparent)
 ├─ PostPass (tonemap)
 └─ ImGuiPass → Present
```

---

## 验证 checklist

### L.2 验证
- [ ] probe ray buffer 有有效命中数据（ImGui 显示 hit/miss ratio）
- [ ] probe atlas SH coefs 非零（ImGui 显示 probe worldPos + 平均 radiance）
- [ ] validation clean

### L.5 验证
- [ ] LumenGI 画面有可见间接光（与 None 档对比）
- [ ] bilinear probe 插值无明显的块状 16×16 artifact
- [ ] 与 IBL 对比：Lumen 间接光应随场景几何有空间变化（IBL 是全屏均匀）
- [ ] GI 下拉切换不 crash（12 档 + Lumen = 13 档）
- [ ] validation clean

### L.4 验证
- [ ] spatial filter on/off toggle（ImGui checkbox）→ 可见 probe 间 discontinuity 减少
- [ ] 边角（depth/normal 突变）不引入 blur 跨边（weight 够大时）
- [ ] 60 fps+ stable

### L.3 验证
- [ ] 6-axis vs isotropic：Sponza 曲面（柱子）indirect 方向性更明显
- [ ] multi-bounce relight：3 bounce vs 1 bounce 总亮度提升（暗角处尤其）
- [ ] 不启用 Lumen 时 VXGI 单层功能不受影响
- [ ] 显存增加 48 MB（6-axis）在 4060 8 GB 内可接受
- [ ] validation clean

---

## 风险

| 风险 | 缓解 |
|------|------|
| Screen probe 密度不够导致 flicker | 自适应密度（8×8 tile, deferred）；先接受 16² |
| SH9 不够准确（低频丢失）| Lumen 论文也是 SH3（9 coeffs）；需要时扩展到 SH16 |
| 6-axis voxel + multi-bounce 显存超 8 GB | 降 voxel 到 64³（6-axis @64³ = 6 MB）；不降到 32³ |
| TLAS ray query 开销太高（32 rays × ~8000 probes = 256K rays/frame）| probe count for 1080p: 120×68=8160 × 32 = 261K rays —— RTX 4060 应该轻松；后续加 temporally amortize（probe 分帧更新） |
| 6-axis sample 导致 cone trace 方向突变 | cosine-weighted blend 相邻主轴（hemisphere cone 重叠区）|
| probe atlas atlas lookup 太慢（stochastic 片元纹理 fetch）| SH coefs 可放入 shared memory（group 内 64 线程处理多个探针）|

---

## 时间估算

| 任务 | 预估 | 说明 |
|------|------|------|
| L.2.1 LumenResources | 0.5 天 | 3-layer atlas + ray buffer 创建 |
| L.2.2 lumen_probe.slang | 1.5 天 | ray query + SH 投影；最难是 SH 投影实现 |
| L.2.3 LumenProbePass | 1 天 | pipeline 创建 + bind + dispatch |
| L.5.1 lumen_gather.slang | 0.5 天 | SH 解算 + bilinear 插值 |
| L.5.2 LumenGatherPass | 0.5 天 | pipeline class |
| L.5.3 RenderTargets 扩展 | 0.5 天 | lumenGI image + lighting.slang 接入 |
| L.5.4 GI dropdown 注册 | 0.5 天 | kGis[13] + App 编排 |
| L.4.1 lumen_filter.slang | 1 天 | spatial cross-bilateral |
| L.4.2 LumenFilterPass + integrate | 1 天 | pipeline + pipe to L.5 gather |
| L.3.1 6-axis storage | 2 天 | 升级 VxgiResources + voxelize/inject adapt |
| L.3.2 multi-bounce relight | 2.5 天 | iterative cone trace + 6-axis read/write |
| L.3.3 probe pass adapt | 1 天 | L.2 改读 6-axis voxel |
| L.3.4 validation + polish | 1.5 天 | round trip 调试 + L.4 temporal 可选 |
| **合计** | **~16 天** | Phase L1 全部 close |

---

## 自检

- [ ] Lumen-lite 作为第 13 档 GI，与现有 12 档共存，切换无 crash
- [ ] Phase L1: L.2 + L.5 端到端画面 → 有可见几何相关的间接光
- [ ] Phase L1+: L.4 spatial filter → probe discontinuity 减少
- [ ] Phase L1++: L.3 6-axis + multi-bounce → 方向性间接光 + 能量增强
- [ ] Sponza 红色帘附近白柱有色溢；拐角软阴影
- [ ] 1080p Sponza ≥ 60 fps
- [ ] validation clean
