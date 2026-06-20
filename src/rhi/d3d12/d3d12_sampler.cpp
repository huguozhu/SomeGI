// rhi/d3d12/d3d12_sampler.cpp — D3D12 采样器实现
#include "d3d12_sampler.h"
#include "d3d12_device.h"
#include "d3d12_texture.h" // toD3D12Cmp
#include <cstdio>

namespace somegi {
namespace rhi {

static D3D12_FILTER toD3D12SamplerFilter(Filter mag, Filter min, SamplerMipmapMode mip) {
    if (min == Filter::Linear && mag == Filter::Linear && mip == SamplerMipmapMode::Linear)
        return D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    if (min == Filter::Linear && mag == Filter::Linear)
        return D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    return D3D12_FILTER_MIN_MAG_MIP_LINEAR;
}

static D3D12_TEXTURE_ADDRESS_MODE toD3D12SamplerAddr(SamplerAddressMode m) {
    switch (m) {
        case SamplerAddressMode::Repeat:         return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        case SamplerAddressMode::MirroredRepeat: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
        case SamplerAddressMode::ClampToBorder:  return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
        default: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    }
}

D3D12RHISampler::D3D12RHISampler(D3D12RHIDevice& device, const SamplerDesc& desc) {
    if (!device.cpuSamplerHeap()) return;

    static uint32_t smpIdx = 0;
    m_cpuHandle = device.cpuSamplerHeap()->GetCPUDescriptorHandleForHeapStart();
    m_cpuHandle.ptr += smpIdx++ * device.cpuSamplerIncrement();

    D3D12_SAMPLER_DESC sd{};
    sd.Filter = toD3D12SamplerFilter(desc.magFilter, desc.minFilter, desc.mipmapMode);
    sd.AddressU = toD3D12SamplerAddr(desc.addressU);
    sd.AddressV = toD3D12SamplerAddr(desc.addressV);
    sd.AddressW = toD3D12SamplerAddr(desc.addressW);
    sd.MipLODBias = 0;
    sd.MaxAnisotropy = 1;
    sd.ComparisonFunc = desc.compareEnable
        ? toD3D12Cmp(desc.compareOp) : D3D12_COMPARISON_FUNC_NEVER;
    sd.MinLOD = 0;
    sd.MaxLOD = desc.maxLod > 0 ? desc.maxLod : D3D12_FLOAT32_MAX;

    device.device()->CreateSampler(&sd, m_cpuHandle);
}

} // namespace rhi
} // namespace somegi
