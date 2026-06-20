// rhi/d3d12/d3d12_acceleration_structure.h — D3D12 加速结构（光线追踪）
#pragma once
#include "../base/acceleration_structure.h"

namespace somegi {
namespace rhi {

class D3D12RHIAccelerationStructure : public RHIAccelerationStructure {
public:
    void* nativeHandle() const override { return nullptr; }
};

} // namespace rhi
} // namespace somegi
