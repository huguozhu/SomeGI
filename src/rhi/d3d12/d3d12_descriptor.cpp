// rhi/d3d12/d3d12_descriptor.cpp — D3D12 描述符集实现
#include "d3d12_descriptor.h"
#include "d3d12_device.h"
#include "d3d12_texture.h"
#include <cstdio>

namespace somegi {
namespace rhi {

D3D12RHIDescriptorSetLayout::D3D12RHIDescriptorSetLayout(const DescSetLayoutDesc& desc)
    : m_bindings(desc.bindings) {}

D3D12RHIDescriptorSet::D3D12RHIDescriptorSet(D3D12RHIDevice& device,
                                               D3D12RHIDescriptorSetLayout& layout)
    : m_device(device) {
    for (auto& b : layout.bindings()) m_count += b.count;
    if (m_count > 0) {
        auto alloc = device.allocDescriptors(m_count);
        m_gpuStart = alloc.gpu;
    }
}

D3D12RHIDescriptorSet::~D3D12RHIDescriptorSet() = default;

void D3D12RHIDescriptorSet::write(const std::vector<DescriptorWrite>& writes) {
    // 向 GPU 可见堆拷贝描述符
    for (auto& w : writes) {
        if (w.textureView) {
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
