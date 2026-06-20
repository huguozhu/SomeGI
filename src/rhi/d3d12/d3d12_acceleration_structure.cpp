// rhi/d3d12/d3d12_acceleration_structure.cpp — D3D12 加速结构实现
#include "d3d12_acceleration_structure.h"

namespace somegi {
namespace rhi {

D3D12RHIAccelerationStructure::D3D12RHIAccelerationStructure(ID3D12Resource* resource, bool owns)
    : m_resource(resource), m_owns(owns) {}

D3D12RHIAccelerationStructure::~D3D12RHIAccelerationStructure() {
    if (m_owns && m_resource) {
        m_resource->Release();
    }
}

std::unique_ptr<RHIAccelerationStructure> D3D12RHIAccelerationStructure::createNonOwning(ID3D12Resource* as) {
    return std::unique_ptr<RHIAccelerationStructure>(new D3D12RHIAccelerationStructure(as, false));
}

} // namespace rhi
} // namespace somegi
