// rhi/d3d12/d3d12_sampler.h — D3D12 采样器
#pragma once
#include "../base/sampler.h"
#include <d3d12.h>

namespace somegi {
namespace rhi {

class D3D12RHIDevice;

class D3D12RHISampler : public RHISampler {
public:
    D3D12RHISampler(D3D12RHIDevice& device, const SamplerDesc& desc);
    void* nativeHandle() const override { return (void*)(uintptr_t)m_cpuHandle.ptr; }
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle() const { return m_cpuHandle; }
private:
    D3D12_CPU_DESCRIPTOR_HANDLE m_cpuHandle{};
};

} // namespace rhi
} // namespace somegi
