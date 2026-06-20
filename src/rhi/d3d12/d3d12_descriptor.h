// rhi/d3d12/d3d12_descriptor.h — D3D12 描述符集
#pragma once
#include "../base/descriptor.h"
#include <d3d12.h>
#include <vector>

namespace somegi {
namespace rhi {

class D3D12RHIDevice;

class D3D12RHIDescriptorSetLayout : public RHIDescriptorSetLayout {
public:
    D3D12RHIDescriptorSetLayout(const DescSetLayoutDesc& desc);
    void* nativeHandle() const override { return nullptr; }
    const std::vector<DescriptorBinding>& bindings() const { return m_bindings; }
    bool hasSamplerTable() const { return m_samplerParamIdx != ~0u; }
    uint32_t resourceParamIdx() const { return m_resourceParamIdx; }
    uint32_t samplerParamIdx() const { return m_samplerParamIdx; }
    uint32_t samplerCount() const { return m_samplerCount; }
    void setResourceParam(uint32_t idx) { m_resourceParamIdx = idx; }
    void setSamplerParam(uint32_t idx) { m_samplerParamIdx = idx; }
private:
    friend class D3D12RHIPipelineState;
    std::vector<DescriptorBinding> m_bindings;
    uint32_t m_resourceParamIdx = ~0u;
    uint32_t m_samplerParamIdx = ~0u;
    uint32_t m_samplerCount = 0;
};

class D3D12RHIDescriptorSet : public RHIDescriptorSet {
public:
    D3D12RHIDescriptorSet(D3D12RHIDevice& device, D3D12RHIDescriptorSetLayout& layout);
    ~D3D12RHIDescriptorSet() override;
    void write(const std::vector<DescriptorWrite>& writes) override;
    void* nativeHandle() const override { return (void*)(uintptr_t)m_gpuStart.ptr; }
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle() const { return m_gpuStart; }
    D3D12_GPU_DESCRIPTOR_HANDLE samplerGpuHandle() const { return m_samplerGpuStart; }
    uint32_t descriptorCount() const { return m_count; }
private:
    D3D12RHIDevice& m_device;
    D3D12_GPU_DESCRIPTOR_HANDLE m_gpuStart{};
    uint32_t m_count = 0;
    D3D12_GPU_DESCRIPTOR_HANDLE m_samplerGpuStart{};
    uint32_t m_samplerCount = 0;
    std::vector<DescriptorWrite> m_pendingWrites;
};

} // namespace rhi
} // namespace somegi
