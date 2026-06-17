// rhi/base/pipeline_state.h — Pipeline State Object 描述符
#pragma once
#include "common.h"
#include <memory>
#include <vector>

namespace somegi {
namespace rhi {

class RHIShader;
class RHIDescriptorSetLayout;

// ════════════════════════════════════════════════════════════════
// Graphics PSO
//
// 支持两种几何管线路径：
//   1. 传统 VS+FS：设置 vertexShader + fragmentShader
//   2. Mesh Shader：设置 meshShader + fragmentShader（+ 可选的 taskShader）
//   vertexShader 与 meshShader 互斥，同时设置时 meshShader 优先。
// ════════════════════════════════════════════════════════════════
struct RenderTargetFormats {
    std::vector<Format> colorFormats;
    Format depthFormat = Format::Unknown;
    uint32_t sampleCount = 1;
};

struct GraphicsPSODesc {
    const char* debugName = nullptr;

    // 着色器（vertexShader 与 meshShader 互斥）
    const RHIShader* vertexShader = nullptr;   // 传统顶点着色器
    const RHIShader* fragmentShader = nullptr; // 像素着色器（两种路径共用）
    const RHIShader* meshShader = nullptr;     // Mesh Shader（替代 VS）
    const RHIShader* taskShader = nullptr;     // Task/Amplification Shader（可选，需配合 meshShader）

    VertexInputState vertexInput;              // Mesh Shader 路径忽略此字段
    PrimitiveTopology topology = PrimitiveTopology::TriangleList;
    RasterizationState rasterization;
    DepthStencilState depthStencil;
    BlendState blend;
    RenderTargetFormats renderTargets;

    std::vector<const RHIDescriptorSetLayout*> descriptorSetLayouts;
    std::vector<PushConstantRange> pushConstants;
};

// ════════════════════════════════════════════════════════════════
// Compute PSO
// ════════════════════════════════════════════════════════════════
struct ComputePSODesc {
    const char* debugName = nullptr;
    const RHIShader* computeShader = nullptr;
    std::vector<const RHIDescriptorSetLayout*> descriptorSetLayouts;
    std::vector<PushConstantRange> pushConstants;
};

// ════════════════════════════════════════════════════════════════
// Ray Tracing PSO
//
// 映射关系：
//   Vulkan:   VkRayTracingPipelineCreateInfoKHR
//   D3D12:    ID3D12StateObject (DXIL libraries + hit groups)
//   Metal:    N/A（Metal 不支持原生 RT 管线，使用 Ray Query 替代）
// ════════════════════════════════════════════════════════════════

// Shader Binding Table 中的组类型
enum class ShaderGroupType : uint32_t {
    RayGen,     // 光线生成（VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR）
    Miss,       // 未命中（VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR）
    Hit,        // 命中组 = ClosestHit + 可选 AnyHit + 可选 Intersection
    Callable,   // 可调用着色器（VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR，预留）
};

// SBT 中的单个着色器组
struct ShaderGroup {
    ShaderGroupType type = ShaderGroupType::RayGen;

    // 通用着色器索引（用于 RayGen / Miss / Callable），指向 shaders 数组
    uint32_t generalShader = UINT32_MAX;

    // 命中组专用（type == Hit），指向 shaders 数组
    uint32_t closestHitShader = UINT32_MAX;
    uint32_t anyHitShader = UINT32_MAX;        // 可选
    uint32_t intersectionShader = UINT32_MAX;  // 可选，用于程序化几何
};

struct RayTracingPSODesc {
    const char* debugName = nullptr;

    // 所有独立的着色器模块（被 groups 通过索引引用）
    // 顺序：[raygen, miss0, miss1, ..., chit0, chit1, ..., ahit0, ...]
    std::vector<const RHIShader*> shaders;

    // SBT 着色器组（顺序决定 SBT 中的组句柄顺序）
    //   groups[0] 必须是 RayGen 组
    //   后续为 Miss 组和 Hit 组，按应用需求排列
    std::vector<ShaderGroup> groups;

    // 递归深度限制（0 = 仅 primary ray，1 = primary + 1 次递归，...）
    uint32_t maxRecursionDepth = 1;

    // 最大 payload / attribute 大小（字节），用于驱动层编译优化
    uint32_t maxPayloadSize = 0;
    uint32_t maxHitAttributeSize = 0;

    std::vector<const RHIDescriptorSetLayout*> descriptorSetLayouts;
    std::vector<PushConstantRange> pushConstants;
};

// ════════════════════════════════════════════════════════════════
// PSO 对象（不透明句柄）
// ════════════════════════════════════════════════════════════════
class RHIPipelineState {
public:
    virtual ~RHIPipelineState() = default;
    virtual void* nativeHandle() const = 0;
};

} // namespace rhi
} // namespace somegi
