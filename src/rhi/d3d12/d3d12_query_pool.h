// rhi/d3d12/d3d12_query_pool.h — D3D12 查询池
#pragma once
#include "../base/command_buffer.h" // RHIQueryPool
#include <d3d12.h>

namespace somegi {
namespace rhi {

class D3D12RHIDevice;

class D3D12RHIQueryPool : public RHIQueryPool {
public:
    D3D12RHIQueryPool(D3D12RHIDevice& device, uint32_t count);
    ~D3D12RHIQueryPool() override;
    void getResults(uint32_t first, uint32_t count, uint64_t* data) override;
    void* nativeHandle() const override { return (void*)m_heap; }
    ID3D12QueryHeap* heap() const { return m_heap; }
private:
    ID3D12QueryHeap* m_heap = nullptr;
    uint32_t m_count = 0;
};

} // namespace rhi
} // namespace somegi
