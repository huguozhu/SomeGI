// rhi/descriptor.h
#pragma once
#include "common.h"
#include <memory>
#include <vector>

namespace somegi {
namespace rhi {

class RHIBuffer;
class RHITextureView;
class RHISampler;
class RHIAccelerationStructure;

// ════════════════════════════════════════════════════════════════
// Descriptor Set Layout
// ════════════════════════════════════════════════════════════════
enum class DescriptorType {
    SampledImage,
    StorageImage,
    UniformBuffer,
    StorageBuffer,
    Sampler,
    AccelerationStructure,  // TLAS / BLAS（光线追踪）
};

struct DescriptorBinding {
    uint32_t binding;
    DescriptorType type;
    uint32_t count = 1;
    ShaderStage visibility = ShaderStage::Compute;
    bool partiallyBound = false;  // VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
    // D3D12 HLSL 寄存器号（~0u 表示使用 binding 值）
    // CBV/SRV/UAV 独立编号，可覆盖 Vulkan binding 实现不同寄存器映射
    uint32_t hlslRegister = ~0u;
};

struct DescSetLayoutDesc {
    std::vector<DescriptorBinding> bindings;
    const char* debugName = nullptr;
    bool updateAfterBind = false;  // VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT
    // 单个 binding 的 UPDATE_AFTER_BIND 标志（binding index → true）
    std::vector<uint32_t> updateAfterBindBindings;
};

class RHIDescriptorSetLayout {
public:
    virtual ~RHIDescriptorSetLayout() = default;
    virtual void* nativeHandle() const = 0;
};

// ════════════════════════════════════════════════════════════════
// Descriptor Set
// ════════════════════════════════════════════════════════════════
struct DescriptorWrite {
    uint32_t binding;
    DescriptorType type;
    const RHITextureView* textureView = nullptr;
    const RHIBuffer* buffer = nullptr;
    uint64_t bufferOffset = 0;
    uint64_t bufferRange = 0;
    const RHISampler* sampler = nullptr;
    const RHIAccelerationStructure* accelerationStructure = nullptr;  // TLAS / BLAS

    // 纹理数组绑定（count > 1 时使用，与 textureView 互斥）
    uint32_t textureArrayCount = 0;
    const RHITextureView* const* textureViewArray = nullptr;
};

class RHIDescriptorSet {
public:
    virtual ~RHIDescriptorSet() = default;
    virtual void write(const std::vector<DescriptorWrite>& writes) = 0;
    virtual void* nativeHandle() const = 0;
};

} // namespace rhi
} // namespace somegi
