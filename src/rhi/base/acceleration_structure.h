// rhi/base/acceleration_structure.h — RHIAccelerationStructure 抽象
#pragma once

namespace somegi {
namespace rhi {

// 不透明句柄，遵循 RHISampler 的零方法模式。
// Vulkan:  VkAccelerationStructureKHR
// D3D12:   ID3D12Resource (TLAS/BLAS)
// Metal:   id<MTLAccelerationStructure> (GPU family ≥ Apple7)
class RHIAccelerationStructure {
public:
    virtual ~RHIAccelerationStructure() = default;
    virtual void* nativeHandle() const = 0;
};

} // namespace rhi
} // namespace somegi
