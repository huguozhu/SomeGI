# RHI (Render Hardware Interface) 设计文档

日期： 2026-06-16
分支： rhi
状态： 设计阶段

---

## 1. 目标

将 SomeGI 从 Vulkan 单一后端解耦，支持 Vulkan / D3D12 / Metal 三个图形 API，
使代码一次编写即可跨 Windows（D3D12/Vulkan）、Linux（Vulkan）、macOS/iOS（Metal）运行。

### 核心原则

- **Pipeline State Object (PSO)** 作为一等公民——描述完整的 GPU 管线配置
- **编译期后端选择**——通过 `RHIDevice` 多态或 `#ifdef` 零开销分发
- **与现代 API 对齐**——Vulkan 1.3 / D3D12 / Metal 3 都是 bindless-capable
- **渐进迁移**——RHI 层与现有 Vulkan 代码共存，逐模块替换

---

## 2. 架构总览

```
┌───────────────────────────────────────────────────────────┐
│                    App / Renderer                         │
├───────────────────────────────────────────────────────────┤
│                  FrameGraph (RHI)                         │
├───────────────────────────────────────────────────────────┤
│                      RHI 层                               │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐ │
│  │ Device   │ Buffer   │ Texture  │ Shader   │ Swapchain│ │
│  ├──────────┼──────────┼──────────┼──────────┼──────────┤ │
│  │ CmdBuf   │ PSO      │ DescSet  │ Fence    │ Semaphore│ │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘ │
├───────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Vulkan    │  │    D3D12     │  │    Metal     │      │
│  │  Backend   │  │   Backend    │  │   Backend    │      │
│  └────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────────────────────────────────────────┘
```

---

## 3. 核心抽象

### 3.1 RHIDevice

```cpp
// rhi/device.h
class RHIDevice {
public:
    enum class Backend { Vulkan, D3D12, Metal };

    static std::unique_ptr<RHIDevice> create(Backend backend, const DeviceDesc& desc);

    // 查询
    Backend backend() const;
    DeviceLimits limits() const;

    // 资源创建
    RHIBuffer createBuffer(const BufferDesc& desc);
    RHITexture createTexture(const TextureDesc& desc);
    RHIShader createShader(const ShaderDesc& desc, const void* bytecode, size_t size);
    RHIPipelineState createGraphicsPSO(const GraphicsPSODesc& desc);
    RHIPipelineState createComputePSO(const ComputePSODesc& desc);
    RHIDescriptorSetLayout createDescriptorSetLayout(const DescSetLayoutDesc& desc);
    RHISwapchain createSwapchain(void* nativeWindow, const SwapchainDesc& desc);

    // 命令
    RHICommandPool createCommandPool();
    RHIFence createFence(bool signaled = false);
    RHISemaphore createSemaphore();

    // 提交
    void submit(const SubmitDesc& desc);
    void present(RHISwapchain swapchain, RHISemaphore waitSemaphore);

    // 同步
    void waitIdle();
    void waitForFence(RHIFence fence, uint64_t timeout = UINT64_MAX);

    // 查询
    void* nativeDevice() const;  // VkDevice / ID3D12Device* / id<MTLDevice>
};
```

### 3.2 RHIBuffer

```cpp
// rhi/buffer.h
enum class BufferUsage : uint32_t {
    Vertex       = 1 << 0,
    Index        = 1 << 1,
    Uniform      = 1 << 2,
    Storage      = 1 << 3,
    Indirect     = 1 << 4,
    TransferSrc  = 1 << 5,
    TransferDst  = 1 << 6,
    AccelStruct  = 1 << 7,  // Ray tracing (Vulkan/D3D12)
};

enum class MemoryType {
    DeviceLocal,   // GPU only
    HostVisible,   // CPU write + GPU read (upload)
    HostCached,    // CPU read + GPU write (readback)
};

struct BufferDesc {
    uint64_t size = 0;
    BufferUsage usage = BufferUsage::Storage;
    MemoryType memory = MemoryType::DeviceLocal;
    const char* debugName = nullptr;
};

class RHIBuffer {
public:
    // 映射（仅 HostVisible/HostCached）
    void* map();
    void unmap();

    uint64_t size() const;
    uint64_t deviceAddress() const;  // GPU virtual address
    void* nativeHandle() const;      // VkBuffer / ID3D12Resource* / id<MTLBuffer>
};
```

### 3.3 RHITexture

```cpp
// rhi/texture.h
enum class TextureUsage : uint32_t {
    Sampled        = 1 << 0,
    Storage        = 1 << 1,
    ColorAttachment = 1 << 2,
    DepthStencil   = 1 << 3,
    TransferSrc    = 1 << 4,
    TransferDst    = 1 << 5,
};

enum class TextureFormat {
    R8_UNORM,
    R8G8B8A8_UNORM,
    R16G16B16A16_SFLOAT,
    R32_SFLOAT,
    D32_SFLOAT,
    // ... 映射到 VK_FORMAT_*/DXGI_FORMAT_*/MTLPixelFormat
};

struct TextureDesc {
    TextureFormat format;
    uint32_t width, height, depth = 1;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    TextureUsage usage;
    uint32_t samples = 1;
    bool isCubemap = false;
    const char* debugName = nullptr;
};

class RHITexture {
public:
    RHITextureView createView(const TextureViewDesc& desc);

    TextureFormat format() const;
    uint32_t width() const;
    uint32_t height() const;
    uint32_t mipLevels() const;
    void* nativeHandle() const;  // VkImage / ID3D12Resource* / id<MTLTexture>
};
```

### 3.4 RHIShader

```cpp
// rhi/shader.h
enum class ShaderStage {
    Vertex,
    Fragment,     // Pixel in D3D12
    Compute,
    Mesh,         // VK_EXT_mesh_shader / D3D12 Mesh Shader
    Task,         // VK_EXT_mesh_shader / D3D12 Amplification
    RayGen,       // Ray tracing
    RayMiss,
    RayClosestHit,
    RayAnyHit,
};

enum class ShaderFormat {
    SPIRV,   // Vulkan (native) / D3D12 (via SPIRV-Cross) / Metal (via SPIRV-Cross)
    DXIL,    // D3D12 native
    MSL,     // Metal native
};

struct ShaderDesc {
    ShaderStage stage;
    ShaderFormat format;
    const char* entryPoint = "main";
    const char* debugName = nullptr;
};

class RHIShader {
public:
    ShaderStage stage() const;
};
```

### 3.5 RHIPipelineState（核心）

PSO 是 RHI 的核心概念——描述一次 draw/dispatch 所需的全部 GPU 状态。

```cpp
// rhi/pipeline_state.h

// ── 顶点输入 ──
enum class VertexFormat {
    Float, Float2, Float3, Float4,
    Uint, Uint2, Uint3, Uint4,
    // ...
};

struct VertexAttribute {
    uint32_t location;
    VertexFormat format;
    uint32_t offset;
    uint32_t binding;
};

struct VertexBinding {
    uint32_t binding;
    uint32_t stride;
    bool perInstance = false;
};

struct VertexInputState {
    std::vector<VertexBinding> bindings;
    std::vector<VertexAttribute> attributes;
};

// ── 输入装配 ──
enum class PrimitiveTopology {
    TriangleList,
    TriangleStrip,
    LineList,
    PointList,
    PatchList,
};

// ── 光栅化 ──
enum class FillMode { Solid, Wireframe };

enum class CullMode { None, Front, Back };

struct RasterizationState {
    FillMode fill = FillMode::Solid;
    CullMode cull = CullMode::Back;
    bool frontCCW = true;
    float depthBias = 0.0f;
    float depthBiasSlope = 0.0f;
};

// ── 深度/模板 ──
enum class CompareFunc { Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always };

struct DepthStencilState {
    bool depthTest = true;
    bool depthWrite = true;
    CompareFunc depthCompare = CompareFunc::Less;
    bool stencilTest = false;
    // stencil state ...
};

// ── 混合 ──
enum class BlendFactor {
    Zero, One, SrcColor, InvSrcColor, SrcAlpha, InvSrcAlpha,
    DstAlpha, InvDstAlpha, DstColor, InvDstColor,
};

enum class BlendOp { Add, Subtract, ReverseSubtract, Min, Max };

struct BlendAttachment {
    bool blendEnable = false;
    BlendFactor srcColor = BlendFactor::One;
    BlendFactor dstColor = BlendFactor::Zero;
    BlendOp colorOp = BlendOp::Add;
    BlendFactor srcAlpha = BlendFactor::One;
    BlendFactor dstAlpha = BlendFactor::Zero;
    BlendOp alphaOp = BlendOp::Add;
};

struct BlendState {
    bool alphaToCoverage = false;
    std::vector<BlendAttachment> attachments;
};

// ── 渲染目标格式 ──
struct RenderTargetFormats {
    std::vector<TextureFormat> colorFormats;
    TextureFormat depthFormat = TextureFormat::D32_SFLOAT;
    uint32_t sampleCount = 1;
};

// ── Graphics PSO 描述符 ──
struct GraphicsPSODesc {
    const char* debugName = nullptr;

    // Shader stages
    RHIShader vertexShader;
    RHIShader fragmentShader;
    RHIShader meshShader;    // optional mesh shader
    RHIShader taskShader;    // optional task/amplification shader

    // Fixed-function state
    VertexInputState vertexInput;
    PrimitiveTopology topology = PrimitiveTopology::TriangleList;
    RasterizationState rasterization;
    DepthStencilState depthStencil;
    BlendState blend;
    RenderTargetFormats renderTargets;

    // Pipeline layout
    std::vector<RHIDescriptorSetLayout> descriptorSetLayouts;
    std::vector<PushConstantRange> pushConstants;
};

// ── Compute PSO 描述符 ──
struct ComputePSODesc {
    const char* debugName = nullptr;
    RHIShader computeShader;

    std::vector<RHIDescriptorSetLayout> descriptorSetLayouts;
    std::vector<PushConstantRange> pushConstants;
};

// ── PSO 对象 ──
class RHIPipelineState {
public:
    // 绑定到命令缓冲区后，所有 draw/dispatch 使用此 PSO
    void* nativeHandle() const;  // VkPipeline / ID3D12PipelineState* / id<MTLRenderPipelineState>
};
```

### 3.6 RHICommandBuffer

```cpp
// rhi/command_buffer.h
class RHICommandBuffer {
public:
    void begin();
    void end();
    void reset();

    // ── PSO 绑定 ──
    void bindPipelineState(const RHIPipelineState& pso);
    void bindDescriptorSet(uint32_t slot, const RHIDescriptorSet& set);
    void pushConstants(const void* data, uint32_t size, uint32_t offset = 0);
    void pushConstants(ShaderStage stage, const void* data, uint32_t size, uint32_t offset = 0);

    // ── 顶点 / 索引 ──
    void bindVertexBuffer(const RHIBuffer& buffer, uint64_t offset = 0);
    void bindIndexBuffer(const RHIBuffer& buffer, uint64_t offset = 0, bool uint16 = true);

    // ── Draw ──
    void draw(uint32_t vertexCount, uint32_t firstVertex = 0, uint32_t firstInstance = 0);
    void drawIndexed(uint32_t indexCount, uint32_t firstIndex = 0, int32_t vertexOffset = 0, uint32_t firstInstance = 0);
    void drawIndirect(const RHIBuffer& buffer, uint64_t offset = 0, uint32_t drawCount = 1, uint32_t stride = 0);
    void drawIndexedIndirect(const RHIBuffer& buffer, uint64_t offset = 0, uint32_t drawCount = 1, uint32_t stride = 0);
    void drawMeshTasks(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);
    void drawMeshTasksIndirect(const RHIBuffer& buffer, uint64_t offset = 0, uint32_t drawCount = 1, uint32_t stride = 0);

    // ── Dispatch ──
    void dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
    void dispatchIndirect(const RHIBuffer& buffer, uint64_t offset = 0);
    void dispatchMesh(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);

    // ── 复制 / 清除 ──
    void copyBuffer(const RHIBuffer& src, const RHIBuffer& dst, uint64_t size, uint64_t srcOffset = 0, uint64_t dstOffset = 0);
    void copyTexture(const RHITexture& src, const RHITexture& dst);
    void clearColor(const RHITextureView& view, float r, float g, float b, float a);
    void clearDepth(const RHITextureView& view, float depth);

    // ── 屏障（RHI 自动管理，极少手动调用） ──
    void textureBarrier(const RHITexture& tex, TextureLayout oldLayout, TextureLayout newLayout);
    void bufferBarrier(const RHIBuffer& buf, BufferAccess oldAccess, BufferAccess newAccess);
    void globalBarrier(PipelineStage src, PipelineStage dst);

    // ── 渲染通道（Metal 模型） ──
    void beginRendering(const RenderingInfo& info);
    void endRendering();

    // ── 时间戳 ──
    void writeTimestamp(const RHIQueryPool& pool, uint32_t index);

    // ── 原生句柄（兼容迁移期） ──
    void* nativeHandle() const;  // VkCommandBuffer / ID3D12GraphicsCommandList* / id<MTLCommandBuffer>
};
```

### 3.7 RHIDescriptorSet

```cpp
// rhi/descriptor.h

// ── 描述符类型 ──
enum class DescriptorType {
    SampledImage,
    StorageImage,
    UniformBuffer,
    StorageBuffer,
    Sampler,
    AccelerationStructure,  // TLAS (Ray tracing)
};

// ── Shader 可见性 ──
enum class ShaderVisibility : uint32_t {
    Vertex      = 1 << 0,
    Fragment    = 1 << 1,
    Compute     = 1 << 2,
    Mesh        = 1 << 3,
    Task        = 1 << 4,
    All         = 0xFFFFFFFF,
};

// ── 描述符绑定 ──
struct DescriptorBinding {
    uint32_t binding;
    DescriptorType type;
    uint32_t count = 1;
    ShaderVisibility visibility = ShaderVisibility::All;
};

// ── Descriptor Set Layout ──
struct DescSetLayoutDesc {
    std::vector<DescriptorBinding> bindings;
    const char* debugName = nullptr;
};

class RHIDescriptorSetLayout { /* opaque */ };

// ── Descriptor Set ──
struct DescriptorWrite {
    uint32_t binding;
    DescriptorType type;
    // 根据 type 选择：
    const RHITextureView* textureView = nullptr;
    const RHIBuffer* buffer = nullptr;
    uint64_t bufferOffset = 0;
    uint64_t bufferRange = 0;
};

class RHIDescriptorSet {
public:
    void write(const std::vector<DescriptorWrite>& writes);
};
```

### 3.8 辅助类型

```cpp
// rhi/common.h

struct PushConstantRange {
    ShaderStage stages;
    uint32_t offset;
    uint32_t size;
};

struct DeviceLimits {
    uint32_t maxTextureSize;
    uint32_t maxSampledTextures;
    uint32_t maxStorageTextures;
    uint32_t maxUniformBufferSize;
    uint32_t maxStorageBufferSize;
    uint32_t maxPushConstantsSize;
    uint32_t maxDrawIndirectCount;
    bool meshShaderSupported;
    bool rayTracingSupported;
    uint32_t timestampPeriod;  // ns per tick
};

// ── 纹理视图 ──
struct TextureViewDesc {
    TextureFormat format;
    uint32_t baseMip = 0;
    uint32_t mipCount = 1;
    uint32_t baseLayer = 0;
    uint32_t layerCount = 1;
};

class RHITextureView {
public:
    void* nativeHandle() const;  // VkImageView / D3D12_CPU_DESCRIPTOR_HANDLE / id<MTLTexture>
};

// ── 渲染信息（替代 VkRenderingInfo） ──
struct RenderingAttachment {
    RHITextureView view;
    TextureLayout layout;
    // resolve/load store ops
};

struct RenderingInfo {
    std::vector<RenderingAttachment> colorAttachments;
    RenderingAttachment depthAttachment;
    uint32_t width, height;
};
```

---

## 4. 后端实现策略

### 4.1 Vulkan 后端

| RHI 概念 | Vulkan 实现 |
|---|---|
| RHIDevice | VkDevice + VkInstance + VMA |
| RHIBuffer | VkBuffer + VMA allocation |
| RHITexture | VkImage + VkImageView |
| RHIShader | VkShaderModule (SPIR-V) |
| RHIPipelineState | VkPipeline + VkPipelineLayout |
| RHICommandBuffer | VkCommandBuffer |
| RHIDescriptorSet | VkDescriptorSet |
| BeginRendering | vkCmdBeginRendering (dynamic rendering) |
| TextureBarrier | vkCmdPipelineBarrier2 (VkImageMemoryBarrier2) |

**优势**：与现有代码几乎 1:1 映射，迁移成本极低。

### 4.2 D3D12 后端

| RHI 概念 | D3D12 实现 |
|---|---|
| RHIDevice | ID3D12Device |
| RHIBuffer | ID3D12Resource (committed/placed) |
| RHITexture | ID3D12Resource + descriptor handle |
| RHIShader | ID3DBlob (DXIL) |
| RHIPipelineState | ID3D12PipelineState + ID3D12RootSignature |
| RHICommandBuffer | ID3D12GraphicsCommandList (bundles) |
| RHIDescriptorSet | descriptor heap + root parameter 映射 |
| BeginRendering | OMSetRenderTargets + ClearRenderTargetView |

**挑战**：
- 描述符堆管理（需要 ring-buffer 式的 GPU 可见堆）
- 资源状态追踪（D3D12 比 Vulkan 更严格）
- 命令列表分配和重置
- Shader 格式转换（SPIR-V → DXIL 通过 SPIRV-Cross）

### 4.3 Metal 后端

| RHI 概念 | Metal 实现 |
|---|---|
| RHIDevice | id<MTLDevice> |
| RHIBuffer | id<MTLBuffer> |
| RHITexture | id<MTLTexture> |
| RHIShader | id<MTLFunction> (from MSL library) |
| RHIPipelineState | id<MTLRenderPipelineState> / id<MTLComputePipelineState> |
| RHICommandBuffer | id<MTLCommandBuffer> + id<MTLRenderCommandEncoder> |
| RHIDescriptorSet | MTLArgumentEncoder / argument buffer |
| BeginRendering | MTLRenderPassDescriptor + beginEncoding |

**挑战**：
- 显式 render pass 模型（必须提前指定 attachments）
- Argument buffer 管理
- Shader 格式转换（SPIR-V → MSL 通过 SPIRV-Cross）
- 无 geometry shader / mesh shader（Metal 3 支持 mesh shader）
- 无 ray tracing（直到 Apple GPU 支持）

---

## 5. Shader 编译管线

```
┌─────────────────┐
│   GLSL / HLSL   │  ← 用户编写的着色器
└────────┬────────┘
         │ glslang / DXC
         ▼
    ┌─────────┐
    │  SPIR-V  │  ← 中间表示（IR）
    └────┬────┘
         │ SPIRV-Cross
    ┌────┼────┬──────────┐
    ▼    ▼    ▼          ▼
  SPIR-V DXIL  MSL    Metal IR
 (Vulkan)(D3D12)(Metal) (Metal)
```

- **Vulkan**：SPIR-V 直接使用
- **D3D12**：SPIRV-Cross → HLSL → DXC → DXIL；或直接使用 SPIR-V（通过 `VK_KHR_spirv_1_4` ？不，D3D12 不支持 SPIR-V）
- **Metal**：SPIRV-Cross → MSL → Metal compiler

### 编译工具集成（CMake）

```cmake
# 对每个 .glsl/.hlsl 文件：
# 1. glslangValidator → .spv（Vulkan）
# 2. spirv-cross → .hlsl → dxc → .dxil（D3D12）
# 3. spirv-cross → .msl → metal → .metallib（Metal）
```

---

## 6. 迁移策略

### Phase 1：RHI 核心抽象
- `rhi/device.h`, `rhi/buffer.h`, `rhi/texture.h`, `rhi/shader.h`
- Vulkan backend 实现（复用现有 Device/Buffer/Image/ShaderModule）
- 现有代码仍直接使用 Vulkan 类型，RHI 作为 thin wrapper

### Phase 2：RHI Pipeline State
- `rhi/pipeline_state.h`, `rhi/descriptor.h`
- GraphicsPSODesc / ComputePSODesc 取代手写 VkPipeline 创建
- 逐 pass 迁移（GBuffer 最先，简单 compute pass 其次）

### Phase 3：RHI Command Buffer
- `rhi/command_buffer.h`
- 替换 VkCommandBuffer 直接使用
- 自动 barrier 集成（复用 FrameGraph）

### Phase 4：D3D12 后端
- SPIRV-Cross 工具链集成
- 描述符堆管理器
- 资源状态追踪器
- Window/Swapchain D3D12 实现

### Phase 5：Metal 后端
- MSL shader 编译
- Argument buffer 管理
- Render pass 模型适配
- macOS/iOS Window 实现（GLFW metal / MoltenVK）

---

## 7. 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| Shader IR | SPIR-V | 行业标准，工具链成熟，三端均支持 |
| PSO 模型 | 显式 GraphicsPSODesc + ComputePSODesc | D3D12/Metal 原生，Vulkan 可高效映射 |
| 描述符模型 | Descriptor Set + Layout | Vulkan 原生，D3D12 root signature 可映射，Metal argument buffer 可映射 |
| 渲染通道 | 可选 beginRendering/endRendering | Metal 需要显式 render pass，Vulkan/D3D12 可用 dynamic rendering |
| 后端选择 | 编译期 `#ifdef` + 运行时多态 | 零开销 + 方便切换 |
| 资源状态追踪 | RHI 层自动管理（类似 FrameGraph barrier） | 降低 D3D12 迁移复杂度，统一三端行为 |
| 内存管理 | VMA (Vulkan) / D3D12MA / Metal 内置 allocator | 统一 API，减少冗余 |

---

## 8. 文件布局

```
src/rhi/
├── common.h              — 通用类型（Format, Usage, Limits, etc.）
├── device.h              — RHIDevice 声明
├── buffer.h              — RHIBuffer 声明
├── texture.h             — RHITexture + RHITextureView
├── shader.h              — RHIShader + ShaderFormat
├── pipeline_state.h      — GraphicsPSODesc / ComputePSODesc / RHIPipelineState
├── descriptor.h          — DescriptorSet / DescriptorSetLayout
├── command_buffer.h      — RHICommandBuffer
├── swapchain.h           — RHISwapchain
├── fence.h               — RHIFence / RHISemaphore
├── query_pool.h          — RHIQueryPool
│
├── vulkan/
│   ├── vk_device.cpp     — Vulkan 后端实现
│   ├── vk_buffer.cpp
│   ├── vk_texture.cpp
│   ├── vk_shader.cpp
│   ├── vk_pso.cpp
│   ├── vk_descriptor.cpp
│   ├── vk_command.cpp
│   └── vk_swapchain.cpp
│
├── d3d12/
│   └── ...               — D3D12 后端实现（Phase 4）
│
├── metal/
│   └── ...               — Metal 后端实现（Phase 5）
│
└── CMakeLists.txt
```

---

## 9. 与现有代码的兼容

Phase 1-2 期间，RHI 对象提供 `.nativeHandle()` 方法返回原生句柄。
现有 `Device`/`Buffer`/`Image`/`ShaderModule` 映射为 RHIDevice/RHIBuffer/RHITexture/RHIShader 的 Vulkan 特化。

```cpp
// 迁移过渡期混用模式：
RHIBuffer rhiBuf = device.createBuffer(desc);
VkBuffer vkBuf = (VkBuffer)rhiBuf.nativeHandle();  // 传给现有 VK 代码
```
