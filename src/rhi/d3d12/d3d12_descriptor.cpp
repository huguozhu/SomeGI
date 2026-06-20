// rhi/d3d12/d3d12_descriptor.cpp — D3D12 描述符集实现
#include "d3d12_descriptor.h"
#include "d3d12_device.h"
#include "d3d12_sampler.h"   // 必须在 d3d12_texture.h 之前（见纹理头文件内 namespace 嵌套）
#include "d3d12_texture.h"
#include <cassert>
#include <cstdio>

namespace somegi {
namespace rhi {

D3D12RHIDescriptorSetLayout::D3D12RHIDescriptorSetLayout(const DescSetLayoutDesc& desc)
    : m_bindings(desc.bindings) {
    for (auto& b : desc.bindings) {
        if (b.type == DescriptorType::Sampler)
            m_samplerCount += b.count;
    }
}

D3D12RHIDescriptorSet::D3D12RHIDescriptorSet(D3D12RHIDevice& device,
                                               D3D12RHIDescriptorSetLayout& layout)
    : m_device(device) {
    // 统计资源描述符（CBV/SRV/UAV）数量
    uint32_t resCount = 0;
    for (auto& b : layout.bindings()) {
        if (b.type != DescriptorType::Sampler)
            resCount += b.count;
    }
    m_count = resCount;
    if (m_count > 0) {
        auto alloc = device.allocDescriptors(m_count);
        m_gpuStart = alloc.gpu;
    }

    // 分配采样器描述符堆空间
    uint32_t smpCount = layout.samplerCount();
    if (smpCount > 0) {
        auto alloc = device.allocSamplerDescriptors(smpCount);
        m_samplerGpuStart = alloc.gpu;
        m_samplerCount = smpCount;
    }
}

D3D12RHIDescriptorSet::~D3D12RHIDescriptorSet() = default;

void D3D12RHIDescriptorSet::write(const std::vector<DescriptorWrite>& writes) {
    uint32_t smpIndex = 0;
    for (auto& w : writes) {
        if (w.sampler) {
            assert(smpIndex < m_samplerCount);
            // 采样器写入：从 CPU sampler 堆拷贝到 GPU 可见 sampler 堆
            auto* smp = static_cast<const D3D12RHISampler*>(w.sampler);
            D3D12_CPU_DESCRIPTOR_HANDLE src = smp->cpuHandle();

            ID3D12DescriptorHeap* gpuSmpHeap = m_device.gpuSamplerHeap();
            D3D12_CPU_DESCRIPTOR_HANDLE dst = gpuSmpHeap->GetCPUDescriptorHandleForHeapStart();
            SIZE_T baseOffset = m_samplerGpuStart.ptr
                - gpuSmpHeap->GetGPUDescriptorHandleForHeapStart().ptr;
            dst.ptr += baseOffset
                       + static_cast<SIZE_T>(smpIndex) * m_device.gpuSamplerIncrement();

            m_device.device()->CopyDescriptorsSimple(1, dst, src,
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
            smpIndex++;
        } else if (w.textureView) {
            auto* view = static_cast<const D3D12RHITextureView*>(w.textureView);
            D3D12_CPU_DESCRIPTOR_HANDLE src = view->srvCpuHandle();
            D3D12_CPU_DESCRIPTOR_HANDLE dst = m_device.gpuDescriptorHeap()->GetCPUDescriptorHandleForHeapStart();
            dst.ptr += m_gpuStart.ptr -
                m_device.gpuDescriptorHeap()->GetGPUDescriptorHandleForHeapStart().ptr;
            m_device.device()->CopyDescriptorsSimple(1, dst, src,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
    }
}

} // namespace rhi
} // namespace somegi
