// rhi/common.h — RHI 通用类型定义
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace somegi {
namespace rhi {

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
    R16G16B16A16_SFLOAT,
    R32_UINT,
    R32_SFLOAT,
    D32_SFLOAT,
    B8G8R8A8_UNORM,
    B8G8R8A8_SRGB,
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

struct TextureViewDesc {
    Format format = Format::Unknown;  // Unknown = inherit
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
enum class BlendFactor { Zero, One, SrcColor, InvSrcColor, SrcAlpha, InvSrcAlpha, DstColor, InvDstColor };
enum class BlendOp { Add, Subtract, ReverseSubtract, Min, Max };

struct VertexAttribute { uint32_t location; VertexFormat format; uint32_t offset; uint32_t binding; };
struct VertexBinding { uint32_t binding; uint32_t stride; bool perInstance = false; };
struct VertexInputState { std::vector<VertexBinding> bindings; std::vector<VertexAttribute> attributes; };

struct RasterizationState { FillMode fill = FillMode::Solid; CullMode cull = CullMode::Back; bool frontCCW = true; };
struct DepthStencilState { bool depthTest = true; bool depthWrite = true; CompareFunc depthCompare = CompareFunc::Less; };
struct BlendAttachment { bool blendEnable = false; BlendFactor srcColor = BlendFactor::One; BlendFactor dstColor = BlendFactor::Zero; BlendOp colorOp = BlendOp::Add; };
struct BlendState { std::vector<BlendAttachment> attachments; };

struct PushConstantRange { ShaderStage stages; uint32_t offset; uint32_t size; };

// ════════════════════════════════════════════════════════════════
// 管线阶段（用于 barrier）
// ════════════════════════════════════════════════════════════════
enum class PipelineStage : uint32_t {
    None             = 0,
    VertexShader     = 1 << 0,
    FragmentShader   = 1 << 1,
    ComputeShader    = 1 << 2,
    MeshShader       = 1 << 3,
    TaskShader       = 1 << 4,
    RayTracingShader = 1 << 5,
    Transfer         = 1 << 6,
    AllCommands      = 0xFFFFFFFF,
};

// ════════════════════════════════════════════════════════════════
// Buffer 访问类型（用于 buffer barrier）
// ════════════════════════════════════════════════════════════════
enum class BufferAccess : uint32_t {
    None            = 0,
    UniformRead     = 1 << 0,  // UBO 读取
    StorageRead     = 1 << 1,  // SSBO 读取
    StorageWrite    = 1 << 2,  // SSBO 写入
    IndexRead       = 1 << 3,  // 索引缓冲读取
    VertexRead      = 1 << 4,  // 顶点缓冲读取
    IndirectRead    = 1 << 5,  // 间接绘制参数读取
    TransferRead    = 1 << 6,  // 传输/复制源
    TransferWrite   = 1 << 7,  // 传输/复制目标
};

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
    uint32_t maxTextureSize = 0;
    uint32_t maxSampledTextures = 0;
    uint32_t maxUniformBufferSize = 0;
    uint32_t maxStorageBufferSize = 0;
    uint32_t maxPushConstantsSize = 0;
    float timestampPeriod = 1.0f;
    bool meshShaderSupported = false;
    bool rayTracingSupported = false;
};

} // namespace rhi
} // namespace somegi
