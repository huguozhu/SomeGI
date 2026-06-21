// rhi/d3d12/d3d12_fence.cpp — D3D12 栅栏实现
#include "d3d12_fence.h"
#include "d3d12_device.h"
#include <stdexcept>

namespace somegi {
namespace rhi {

D3D12RHIFence::D3D12RHIFence(D3D12RHIDevice& device, bool signaled) {
    if (FAILED(device.device()->CreateFence(
            signaled ? 1 : 0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) {
        throw std::runtime_error("[d3d12] CreateFence failed");
    }
    m_event = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    m_value = signaled ? 1 : 0;
}

D3D12RHIFence::~D3D12RHIFence() {
    if (m_fence) m_fence->Release();
    if (m_event) CloseHandle(m_event);
}

void D3D12RHIFence::wait(uint64_t timeoutNs) {
    if (m_fence->GetCompletedValue() < m_value) {
        m_fence->SetEventOnCompletion(m_value, m_event);
        DWORD ms = (timeoutNs == UINT64_MAX) ? INFINITE
                   : static_cast<DWORD>(timeoutNs / 1'000'000);
        WaitForSingleObject(m_event, ms);
    }
}

void D3D12RHIFence::reset() {
    m_value++;
    m_fence->Signal(m_value);
}

} // namespace rhi
} // namespace somegi
