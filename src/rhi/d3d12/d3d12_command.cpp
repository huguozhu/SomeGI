// rhi/d3d12/d3d12_command.cpp — D3D12 命令缓冲/池/栅栏实现
#include "d3d12_command.h"
#include "d3d12_device.h"
#include <stdexcept>
#include <cstdio>

namespace somegi {
namespace rhi {

// ════════════════════════════════════════════════════════════════
// D3D12RHICommandPool
// ════════════════════════════════════════════════════════════════

D3D12RHICommandPool::D3D12RHICommandPool(D3D12RHIDevice& device)
    : m_device(device) {
    if (FAILED(device.device()->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_allocator)))) {
        throw std::runtime_error("[d3d12] CreateCommandAllocator failed");
    }
}

D3D12RHICommandPool::~D3D12RHICommandPool() {
    if (m_allocator) m_allocator->Release();
}

RHICommandBuffer* D3D12RHICommandPool::allocateRaw() {
    return new D3D12RHICommandBuffer(m_device, *this);
}

void D3D12RHICommandPool::reset() {
    m_allocator->Reset();
}

// ════════════════════════════════════════════════════════════════
// D3D12RHICommandBuffer
// ════════════════════════════════════════════════════════════════

D3D12RHICommandBuffer::D3D12RHICommandBuffer(D3D12RHIDevice& device,
                                               D3D12RHICommandPool& pool)
    : m_device(device), m_pool(pool) {
    if (FAILED(device.device()->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT, pool.m_allocator,
            nullptr, IID_PPV_ARGS(&m_cmdList)))) {
        throw std::runtime_error("[d3d12] CreateCommandList failed");
    }
    // 创建后立即关闭（D3D12 命令列表创建时处于 recording 状态）
    m_cmdList->Close();
}

D3D12RHICommandBuffer::~D3D12RHICommandBuffer() {
    if (m_cmdList) m_cmdList->Release();
}

void D3D12RHICommandBuffer::begin() {
    m_pool.m_allocator->Reset();
    m_cmdList->Reset(m_pool.m_allocator, nullptr);
    m_recording = true;
}

void D3D12RHICommandBuffer::end() {
    m_cmdList->Close();
    m_recording = false;
}

void D3D12RHICommandBuffer::reset() {
    if (m_recording) end();
    begin();
}

void D3D12RHICommandBuffer::setViewport(float x, float y, float w, float h,
                                         float minD, float maxD) {
    D3D12_VIEWPORT vp{x, y, w, h, minD, maxD};
    m_cmdList->RSSetViewports(1, &vp);
}

void D3D12RHICommandBuffer::setScissor(int32_t x, int32_t y,
                                        uint32_t w, uint32_t h) {
    D3D12_RECT rc{LONG(x), LONG(y), LONG(x + w), LONG(y + h)};
    m_cmdList->RSSetScissorRects(1, &rc);
}

// ════════════════════════════════════════════════════════════════
// D3D12RHIFence
// ════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════
// D3D12RHISemaphore（桩，D3D12 用 fence 代替 semaphore）
// ════════════════════════════════════════════════════════════════

D3D12RHISemaphore::D3D12RHISemaphore() = default;
D3D12RHISemaphore::~D3D12RHISemaphore() = default;

} // namespace rhi
} // namespace somegi
