// rhi/d3d12/d3d12_fence.h — D3D12 栅栏和信号量
#pragma once
#include "../base/fence.h"
#include <d3d12.h>
#include <windows.h>

namespace somegi {
namespace rhi {

class D3D12RHIDevice;

class D3D12RHIFence : public RHIFence {
public:
    D3D12RHIFence(D3D12RHIDevice& device, bool signaled);
    ~D3D12RHIFence() override;
    void wait(uint64_t timeoutNs) override;
    void reset() override;
    void* nativeHandle() const override { return (void*)m_fence; }
    ID3D12Fence* fence() { return m_fence; }
    uint64_t& value() { return m_value; }
    HANDLE event() { return m_event; }
private:
    ID3D12Fence* m_fence = nullptr;
    HANDLE m_event = nullptr;
    uint64_t m_value = 0;
};

class D3D12RHISemaphore : public RHISemaphore {
public:
    D3D12RHISemaphore() = default;
    ~D3D12RHISemaphore() override = default;
    void* nativeHandle() const override { return nullptr; }
};

} // namespace rhi
} // namespace somegi
