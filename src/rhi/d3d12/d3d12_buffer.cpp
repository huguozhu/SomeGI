// rhi/d3d12/d3d12_buffer.cpp — D3D12 缓冲实现
#include "d3d12_buffer.h"
#include "d3d12_device.h"
#include <stdexcept>
#include <cstdio>

namespace somegi {
namespace rhi {

// 缓冲用法 → D3D12_RESOURCE_FLAGS
static D3D12_RESOURCE_FLAGS toD3D12ResourceFlags(BufferUsage usage) {
    D3D12_RESOURCE_FLAGS f = D3D12_RESOURCE_FLAG_NONE;
    if ((uint32_t)usage & (uint32_t)BufferUsage::Storage)
        f |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if ((uint32_t)usage & (uint32_t)BufferUsage::AccelStruct)
        f |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return f;
}

// MemoryType → D3D12_HEAP_TYPE
static D3D12_HEAP_TYPE toD3D12HeapType(MemoryType mt) {
    switch (mt) {
        case MemoryType::DeviceLocal:  return D3D12_HEAP_TYPE_DEFAULT;
        case MemoryType::HostVisible:  return D3D12_HEAP_TYPE_UPLOAD;
        case MemoryType::HostCached:   return D3D12_HEAP_TYPE_READBACK;
        default: return D3D12_HEAP_TYPE_DEFAULT;
    }
}

D3D12RHIBuffer::D3D12RHIBuffer(D3D12RHIDevice& device, const BufferDesc& desc)
    : m_device(device), m_size(desc.size), m_memoryType(desc.memory) {

    auto heapType = toD3D12HeapType(desc.memory);
    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = heapType;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC resDesc{};
    resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resDesc.Alignment = 0;
    resDesc.Width = desc.size;
    resDesc.Height = 1;
    resDesc.DepthOrArraySize = 1;
    resDesc.MipLevels = 1;
    resDesc.Format = DXGI_FORMAT_UNKNOWN;
    resDesc.SampleDesc = {1, 0};
    resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resDesc.Flags = toD3D12ResourceFlags(desc.usage);

    auto initialState = (heapType == D3D12_HEAP_TYPE_UPLOAD)
        ? D3D12_RESOURCE_STATE_GENERIC_READ
        : D3D12_RESOURCE_STATE_COMMON;

    if (FAILED(device.device()->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
            initialState, nullptr, IID_PPV_ARGS(&m_resource)))) {
        throw std::runtime_error("[d3d12] CreateCommittedResource(buffer) failed");
    }

    if (desc.debugName) {
        wchar_t wname[128];
        MultiByteToWideChar(CP_UTF8, 0, desc.debugName, -1, wname, 128);
        m_resource->SetName(wname);
    }

    m_gpuAddr = m_resource->GetGPUVirtualAddress();
    // 注册初始资源状态
    device.trackResourceState(m_resource, initialState);

    if (desc.memory == MemoryType::HostVisible || desc.memory == MemoryType::HostCached) {
        // UPLOAD/READBACK heap：持久映射
        D3D12_RANGE range{0, 0};
        m_resource->Map(0, &range, &m_mapped);
    }
}

D3D12RHIBuffer::~D3D12RHIBuffer() {
    if (m_mapped) {
        m_resource->Unmap(0, nullptr);
    }
    if (m_resource) {
        m_device.removeResourceState(m_resource);
        m_resource->Release();
    }
}

void* D3D12RHIBuffer::map() {
    if (m_mapped) return m_mapped; // 已持久映射
    D3D12_RANGE range{0, static_cast<SIZE_T>(m_size)};
    void* ptr = nullptr;
    m_resource->Map(0, &range, &ptr);
    return ptr;
}

void D3D12RHIBuffer::unmap() {
    if (!m_mapped) {
        m_resource->Unmap(0, nullptr);
    }
}

uint64_t D3D12RHIBuffer::deviceAddress() const {
    return m_gpuAddr;
}

} // namespace rhi
} // namespace somegi
