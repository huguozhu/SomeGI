// rhi/pipeline_state.h
#pragma once
#include "common.h"
#include <memory>

namespace somegi {
namespace rhi {

class RHIShader;
class RHIDescriptorSetLayout;

// ════════════════════════════════════════════════════════════════
// Graphics PSO
// ════════════════════════════════════════════════════════════════
struct RenderTargetFormats {
    std::vector<Format> colorFormats;
    Format depthFormat = Format::Unknown;
    uint32_t sampleCount = 1;
};

struct GraphicsPSODesc {
    const char* debugName = nullptr;

    const RHIShader* vertexShader = nullptr;
    const RHIShader* fragmentShader = nullptr;
    const RHIShader* meshShader = nullptr;
    const RHIShader* taskShader = nullptr;

    VertexInputState vertexInput;
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
// PSO 对象
// ════════════════════════════════════════════════════════════════
class RHIPipelineState {
public:
    virtual ~RHIPipelineState() = default;
    virtual void* nativeHandle() const = 0;
};

} // namespace rhi
} // namespace somegi
