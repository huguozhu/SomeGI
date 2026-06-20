// rhi/d3d12/d3d12_buffer.h — D3D12 缓冲
#pragma once
#include "../base/buffer.h"
#include <d3d12.h>

namespace somegi {
namespace rhi {

class D3D12RHIDevice;

class D3D12RHIBuffer : public RHIBuffer {
public:
    D3D12RHIBuffer(D3D12RHIDevice& device, const BufferDesc& desc);
    ~D3D12RHIBuffer() override;

    void* map() override;
    void unmap() override;
    uint64_t size() const override { return m_size; }
    uint64_t deviceAddress() const override;
    void* nativeHandle() const override { return (void*)m_resource; }

    ID3D12Resource* resource() { return m_resource; }
    D3D12_GPU_VIRTUAL_ADDRESS gpuAddress() const { return m_gpuAddr; }

    // 非拥有型包装（用于临时包装已有 Vulkan 资源，暂不实现）
    // static std::unique_ptr<RHIBuffer> createNonOwning(D3D12RHIDevice&, ...);

private:
    D3D12RHIDevice& m_device;
    ID3D12Resource* m_resource = nullptr;
    uint64_t m_size = 0;
    D3D12_GPU_VIRTUAL_ADDRESS m_gpuAddr = 0;
    void* m_mapped = nullptr;
    MemoryType m_memoryType = MemoryType::DeviceLocal;
};

} // namespace rhi
} // namespace somegi
