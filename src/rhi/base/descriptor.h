// rhi/descriptor.h
#pragma once
#include "common.h"
#include <memory>
#include <vector>

namespace somegi {
namespace rhi {

class RHIBuffer;
class RHITextureView;

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
    const void* sampler = nullptr;  // 原生 VkSampler / ID3D12DescriptorHeap / id<MTLSamplerState>
};

class RHIDescriptorSet {
public:
    virtual ~RHIDescriptorSet() = default;
    virtual void write(const std::vector<DescriptorWrite>& writes) = 0;
    virtual void* nativeHandle() const = 0;
};

} // namespace rhi
} // namespace somegi
