// rhi/d3d12/d3d12_query_pool.cpp — D3D12 查询池（时间戳）
#include "d3d12_query_pool.h"
#include "d3d12_device.h"
#include <stdexcept>

namespace somegi {
namespace rhi {

D3D12RHIQueryPool::D3D12RHIQueryPool(D3D12RHIDevice& device, uint32_t count)
    : m_count(count) {
    D3D12_QUERY_HEAP_DESC qd{};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = count;
    if (FAILED(device.device()->CreateQueryHeap(&qd, IID_PPV_ARGS(&m_heap))))
        throw std::runtime_error("[d3d12] CreateQueryHeap failed");
}

D3D12RHIQueryPool::~D3D12RHIQueryPool() {
    if (m_heap) m_heap->Release();
}

void D3D12RHIQueryPool::getResults(uint32_t first, uint32_t count, uint64_t* data) {
    // D3D12 时间戳回读需要 resolve + copy 到 readback buffer
    // Phase 5 完整实现
    (void)first; (void)count; (void)data;
}

} // namespace rhi
} // namespace somegi
