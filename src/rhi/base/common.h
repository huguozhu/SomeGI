// rhi/common.h — RHI 通用类型定义
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace somegi {
namespace rhi {

constexpr uint32_t kFramesInFlight = 2;

// ════════════════════════════════════════════════════════════════
// 后端枚举
// ════════════════════════════════════════════════════════════════
enum class Backend { Vulkan, D3D12, Metal };

// ════════════════════════════════════════════════════════════════
// 资源格式
// ════════════════════════════════════════════════════════════════
enum class Format : uint32_t {
    Unknown = 0,
    R8_UNORM,
    R8G8B8A8_UNORM,
    R16G16_SFLOAT,
    R16_SFLOAT,
    R16G16B16A16_SFLOAT,
    R32_UINT,
    R32_SFLOAT,
    R32G32_SFLOAT,
    R32G32B32A32_UINT,
    D32_SFLOAT,
    R8G8B8A8_SRGB,
    R32G32B32A32_SFLOAT,
    B8G8R8A8_UNORM,
    B8G8R8A8_SRGB,
};

// ════════════════════════════════════════════════════════════════
// Buffer → Texture 复制区域描述
// ════════════════════════════════════════════════════════════════
struct BufferTextureCopyRegion {
    uint64_t bufferOffset = 0;       // 源 buffer 偏移（字节）
    uint32_t bufferRowLength = 0;    // 行字节数（0 = 使用 extent.width * 像素字节）
    uint32_t bufferImageHeight = 0;  // 图像高度（0 = 使用 extent.height）
    uint32_t texMipLevel = 0;        // 目标 mip level
    uint32_t texArrayLayer = 0;      // 目标 array layer
    int32_t texOffsetX = 0;          // 目标纹理偏移 X
    int32_t texOffsetY = 0;          // 目标纹理偏移 Y
    int32_t texOffsetZ = 0;          // 目标纹理偏移 Z
    uint32_t extentWidth = 0;        // 复制宽度
    uint32_t extentHeight = 1;       // 复制高度
    uint32_t extentDepth = 1;        // 复制深度
};

// ════════════════════════════════════════════════════════════════
// Texture → Texture Blit 区域描述
// ════════════════════════════════════════════════════════════════
struct TextureBlitRegion {
    uint32_t srcMipLevel = 0;
    int32_t srcOffsetX = 0, srcOffsetY = 0, srcOffsetZ = 0;
    uint32_t srcExtentWidth = 0, srcExtentHeight = 1, srcExtentDepth = 1;
    uint32_t dstMipLevel = 0;
    int32_t dstOffsetX = 0, dstOffsetY = 0, dstOffsetZ = 0;
    uint32_t dstExtentWidth = 0, dstExtentHeight = 1, dstExtentDepth = 1;
    uint32_t layerCount = 1;         // Number of array layers to blit (e.g., 6 for cubemap faces)
    // Filter: Linear (for mipmap generation) or Nearest (for integer formats)
    bool linearFilter = true;
};

// ════════════════════════════════════════════════════════════════
// Buffer
// ════════════════════════════════════════════════════════════════
enum class BufferUsage : uint32_t {
    Vertex          = 1 << 0,
    Index           = 1 << 1,
    Uniform         = 1 << 2,
    Storage         = 1 << 3,
    Indirect        = 1 << 4,
    TransferSrc     = 1 << 5,
    TransferDst     = 1 << 6,
    AccelStruct     = 1 << 7,  // 加速结构存储（VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR）
    ShaderBindingTable = 1 << 8,  // SBT（VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR）
};

enum class MemoryType {
    DeviceLocal,   // GPU only (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    HostVisible,   // CPU write + GPU read
    HostCached,    // CPU read + GPU write (readback)
};

struct BufferDesc {
    uint64_t size = 0;
    BufferUsage usage = BufferUsage::Storage;
    MemoryType memory = MemoryType::DeviceLocal;
    uint64_t alignment = 0;
    const char* debugName = nullptr;
};

// ════════════════════════════════════════════════════════════════
// Texture
// ════════════════════════════════════════════════════════════════
enum class TextureUsage : uint32_t {
    Sampled         = 1 << 0,
    Storage         = 1 << 1,
    ColorAttachment = 1 << 2,
    DepthStencil    = 1 << 3,
    TransferSrc     = 1 << 4,
    TransferDst     = 1 << 5,
};
inline constexpr TextureUsage operator|(TextureUsage a, TextureUsage b) {
    return static_cast<TextureUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

struct TextureDesc {
    Format format = Format::R8G8B8A8_UNORM;
    uint32_t width = 1, height = 1, depth = 1;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    TextureUsage usage = TextureUsage::Sampled;
    uint32_t samples = 1;
    bool isCubemap = false;
    const char* debugName = nullptr;
};

// 纹理视图类型（与 VkImageViewType 对应，默认根据纹理属性自动推断）
enum class TextureViewType : uint32_t {
    Default = 0,  // 自动推断：cubemap→Cube, layerCount>1→2DArray, 否则→2D
    Texture2D,
    Texture2DArray,
    TextureCube,
    TextureCubeArray,
    Texture3D,
};

struct TextureViewDesc {
    Format format = Format::Unknown;       // Unknown = inherit from texture
    TextureViewType viewType = TextureViewType::Default;
    uint32_t baseMip = 0;
    uint32_t mipCount = 1;
    uint32_t baseLayer = 0;
    uint32_t layerCount = 1;
};

// ════════════════════════════════════════════════════════════════
// Shader
// ════════════════════════════════════════════════════════════════
enum class ShaderStage : uint32_t {
    Vertex     = 1 << 0,
    Fragment   = 1 << 1,
    Compute    = 1 << 2,
    Mesh       = 1 << 3,
    Task       = 1 << 4,
    RayGen     = 1 << 5,
    RayMiss    = 1 << 6,
    RayClosestHit = 1 << 7,
    RayAnyHit  = 1 << 8,
};
inline constexpr ShaderStage operator|(ShaderStage a, ShaderStage b) {
    return static_cast<ShaderStage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

enum class ShaderFormat { SPIRV, DXIL, MSL };

struct ShaderDesc {
    ShaderStage stage;
    ShaderFormat format = ShaderFormat::SPIRV;
    const char* entryPoint = "main";
    const char* debugName = nullptr;
};

// ════════════════════════════════════════════════════════════════
// Pipeline State
// ════════════════════════════════════════════════════════════════
enum class VertexFormat { Float, Float2, Float3, Float4, Uint, Uint2, Uint3, Uint4 };

enum class PrimitiveTopology { TriangleList, TriangleStrip, LineList, PointList };
enum class FillMode { Solid, Wireframe };
enum class CullMode { None, Front, Back };
enum class CompareFunc { Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always };

// ════════════════════════════════════════════════════════════════
// Sampler
// ════════════════════════════════════════════════════════════════
enum class Filter { Nearest, Linear };
enum class SamplerAddressMode { ClampToEdge, Repeat, MirroredRepeat, ClampToBorder };
enum class SamplerMipmapMode { Nearest, Linear };

struct SamplerDesc {
    Filter magFilter = Filter::Linear;
    Filter minFilter = Filter::Linear;
    SamplerMipmapMode mipmapMode = SamplerMipmapMode::Linear;
    SamplerAddressMode addressU = SamplerAddressMode::ClampToEdge;
    SamplerAddressMode addressV = SamplerAddressMode::ClampToEdge;
    SamplerAddressMode addressW = SamplerAddressMode::ClampToEdge;
    float maxLod = 0.0f;
    bool compareEnable = false;
    CompareFunc compareOp = CompareFunc::LessEqual;
    const char* debugName = nullptr;
};
enum class BlendFactor { Zero, One, SrcColor, InvSrcColor, SrcAlpha, InvSrcAlpha, DstColor, InvDstColor };
enum class BlendOp { Add, Subtract, ReverseSubtract, Min, Max };

struct VertexAttribute { uint32_t location; VertexFormat format; uint32_t offset; uint32_t binding; };
struct VertexBinding { uint32_t binding; uint32_t stride; bool perInstance = false; };
struct VertexInputState { std::vector<VertexBinding> bindings; std::vector<VertexAttribute> attributes; };

struct RasterizationState {
    FillMode fill = FillMode::Solid;
    CullMode cull = CullMode::Back;
    bool frontCCW = true;
    // Depth bias（阴影贴图消除 shadow acne）
    bool depthBiasEnable = false;
    float depthBiasConstantFactor = 0.0f;
    float depthBiasSlopeFactor = 0.0f;
    float depthBiasClamp = 0.0f;
};
struct DepthStencilState { bool depthTest = true; bool depthWrite = true; CompareFunc depthCompare = CompareFunc::Less; };
struct BlendAttachment { bool blendEnable = false; BlendFactor srcColor = BlendFactor::One; BlendFactor dstColor = BlendFactor::Zero; BlendOp colorOp = BlendOp::Add; };
struct BlendState { std::vector<BlendAttachment> attachments; };

struct PushConstantRange { ShaderStage stages; uint32_t offset; uint32_t size; };

// ════════════════════════════════════════════════════════════════
// 渲染通道（beginRendering attachment 描述）
// ════════════════════════════════════════════════════════════════
enum class AttachmentLoadOp { Clear, Load, DontCare };
enum class AttachmentStoreOp { Store, DontCare };
enum class ResolveMode { Average, Min, Max };

// ════════════════════════════════════════════════════════════════
// 管线阶段（用于 barrier）
// ════════════════════════════════════════════════════════════════
enum class PipelineStage : uint32_t {
    None             = 0,
    TopOfPipe        = 1 << 7,   // 管线顶部
    VertexShader     = 1 << 0,
    FragmentShader   = 1 << 1,   // 包含 ColorAttachmentOutput
    ComputeShader    = 1 << 2,
    MeshShader       = 1 << 3,
    TaskShader       = 1 << 4,
    RayTracingShader = 1 << 5,
    Transfer         = 1 << 6,
    BottomOfPipe     = 1 << 8,   // 管线底部
    AllCommands      = 0xFFFFFFFF,
};
inline constexpr PipelineStage operator|(PipelineStage a, PipelineStage b) {
    return static_cast<PipelineStage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

// ════════════════════════════════════════════════════════════════
// Buffer 访问类型（用于 buffer barrier）
// ════════════════════════════════════════════════════════════════
enum class BufferAccess : uint32_t {
    None              = 0,
    UniformRead       = 1 << 0,  // UBO 读取
    StorageRead       = 1 << 1,  // SSBO 读取
    StorageWrite      = 1 << 2,  // SSBO 写入
    IndexRead         = 1 << 3,  // 索引缓冲读取
    VertexRead        = 1 << 4,  // 顶点缓冲读取
    IndirectRead      = 1 << 5,  // 间接绘制参数读取
    TransferRead      = 1 << 6,  // 传输/复制源
    TransferWrite     = 1 << 7,  // 传输/复制目标
    ColorAttachmentRead  = 1 << 8,  // 颜色附件读取
    ColorAttachmentWrite = 1 << 9,  // 颜色附件写入
    DepthStencilRead  = 1 << 10, // 深度/模板读取
    DepthStencilWrite = 1 << 11, // 深度/模板写入
    MemoryRead        = 1u << 30, // 通用内存读取（用于屏障）
    MemoryWrite       = 1u << 31, // 通用内存写入（用于屏障）
};
inline constexpr BufferAccess operator|(BufferAccess a, BufferAccess b) {
    return static_cast<BufferAccess>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

// ════════════════════════════════════════════════════════════════
// 纹理布局（用于 texture barrier）
// ════════════════════════════════════════════════════════════════
enum class TextureLayout : uint32_t {
    Undefined, General, ShaderReadOnly, ColorAttachment, DepthAttachment,
    TransferSrc, TransferDst, Present,
};

// ════════════════════════════════════════════════════════════════
// 设备限制
// ════════════════════════════════════════════════════════════════
struct DeviceLimits {
    // 基础限制
    uint32_t maxTextureSize = 0;
    uint32_t maxSampledTextures = 0;       // 单 set 最大采样纹理数
    uint32_t maxSampledImages = 0;          // 单 set 最大采样图像数（Vulkan descriptor count）
    uint32_t maxUniformBufferSize = 0;
    uint32_t maxStorageBufferSize = 0;
    uint32_t maxPushConstantsSize = 0;
    float timestampPeriod = 1.0f;
    // 能力标志
    bool meshShaderSupported = false;
    bool taskShaderSupported = false;
    bool rayTracingSupported = false;
    bool accelStructSupported = false;
    bool rayQuerySupported = false;
    // 网格着色器限制
    uint32_t maxMeshOutputVertices = 0;
    uint32_t maxMeshOutputPrimitives = 0;
    uint32_t maxMeshWorkGroupInvocations = 0;
    uint32_t maxMeshWorkGroupSize = 0;  // Mesh Shader 工作组大小（第一维）
    // MSAA 支持的采样数位掩码
    uint32_t supportedSampleCounts = 0;
};

} // namespace rhi
} // namespace somegi
