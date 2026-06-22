// rhi/d3d12/d3d12_swapchain.cpp — D3D12 交换链实现
#include "d3d12_swapchain.h"
#include "d3d12_device.h"
#include "d3d12_texture.h"
#include <stdexcept>
#include <cstdio>

namespace somegi {
namespace rhi {

D3D12RHISwapchain::D3D12RHISwapchain(D3D12RHIDevice& device, void* nativeWindow,
                                       uint32_t width, uint32_t height)
    : m_device(device), m_hwnd(D3D12RHIDevice::getHwnd(nativeWindow)) {
    createSwapchain(width, height);
    createRTVs();
    std::printf("[d3d12] swapchain created: %ux%u\n", width, height);
}

D3D12RHISwapchain::~D3D12RHISwapchain() {
    releaseResources();
}

void D3D12RHISwapchain::createSwapchain(uint32_t width, uint32_t height) {
    m_width = width;
    m_height = height;

    IDXGIFactory6* factory = m_device.dxgiFactory();

    // ── 描述交换链 ──
    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Width       = width;
    sd.Height      = height;
    sd.Format      = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.Stereo      = FALSE;
    sd.SampleDesc  = {1, 0};
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = kFrameCount;
    sd.Scaling     = DXGI_SCALING_STRETCH;
    sd.SwapEffect  = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.AlphaMode   = DXGI_ALPHA_MODE_UNSPECIFIED;
    sd.Flags       = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

    // ── 创建交换链 ──
    IDXGISwapChain1* sc1 = nullptr;
    if (FAILED(factory->CreateSwapChainForHwnd(
            m_device.commandQueue(), m_hwnd, &sd, nullptr, nullptr, &sc1))) {
        throw std::runtime_error("[d3d12] CreateSwapChainForHwnd failed");
    }
    sc1->QueryInterface(IID_PPV_ARGS(&m_swapchain));
    sc1->Release();

    // 禁用 Alt+Enter 全屏切换
    factory->MakeWindowAssociation(m_hwnd, DXGI_MWA_NO_ALT_ENTER);

    m_format = Format::B8G8R8A8_UNORM;
}

void D3D12RHISwapchain::createRTVs() {
    // ── RTV 描述符堆 ──
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.NumDescriptors = kFrameCount;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(m_device.device()->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_rtvHeap)))) {
        throw std::runtime_error("[d3d12] CreateDescriptorHeap(RTV) failed");
    }
    m_rtvDescriptorSize = m_device.device()->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // ── 创建 RTV ──
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    m_backBuffers.resize(kFrameCount);

    for (uint32_t i = 0; i < kFrameCount; ++i) {
        if (FAILED(m_swapchain->GetBuffer(i, IID_PPV_ARGS(&m_backBuffers[i])))) {
            throw std::runtime_error("[d3d12] GetBuffer failed");
        }
        m_device.device()->CreateRenderTargetView(m_backBuffers[i], nullptr, rtvH);
        rtvH.ptr += m_rtvDescriptorSize;
    }

    // ── 同步对象 ──
    m_syncs.resize(kFrameCount);
    for (auto& s : m_syncs) {
        m_device.device()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s.fence));
        s.fenceEvent = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
        s.fence->Signal(1); // 初始状态为已完成
        s.fenceValue = 1;
    }
}

void D3D12RHISwapchain::releaseResources() {
    for (auto& s : m_syncs) {
        if (s.fence)       { s.fence->Release(); }
        if (s.fenceEvent)  { CloseHandle(s.fenceEvent); }
    }
    for (auto& buf : m_backBuffers) {
        if (buf) { buf->Release(); }
    }
    if (m_rtvHeap)   { m_rtvHeap->Release(); }
    if (m_swapchain) { m_swapchain->Release(); }
}

// ════════════════════════════════════════════════════════════════
// 帧循环
// ════════════════════════════════════════════════════════════════

SwapchainFrame D3D12RHISwapchain::acquireNextFrame() {
    SwapchainFrame f;
    f.frameInFlight = m_frameIndex % kFrameCount;
    auto& sync = m_syncs[f.frameInFlight];

    // 等待上一帧完成
    if (sync.fence->GetCompletedValue() < sync.fenceValue) {
        WaitForSingleObject(sync.fenceEvent, INFINITE);
    }

    f.width  = m_width;
    f.height = m_height;

    // 创建 back buffer 的非拥有型包装（texture + view）
    uint32_t idx = m_frameIndex % kFrameCount;
    f.texture = D3D12RHITexture::createNonOwning(m_device, m_backBuffers[idx], m_format, m_width, m_height, 1);
    auto* bbView = new D3D12RHITextureView();
    bbView->m_isRTV = true;
    bbView->m_rtvHandle = rtvHandle(idx);
    f.view.reset(bbView);

    m_frameIndex++;
    return f;
}

void D3D12RHISwapchain::present(const SwapchainFrame& frame) {
    presentCurrentFrame();
    // 信号本帧 fence（acquire 时等待）
    auto& sync = m_syncs[frame.frameInFlight];
    sync.fenceValue++;
    m_device.commandQueue()->Signal(sync.fence, sync.fenceValue);
    sync.fence->SetEventOnCompletion(sync.fenceValue, sync.fenceEvent);
}

void D3D12RHISwapchain::presentCurrentFrame() const {
    m_swapchain->Present(1, 0);
}

void D3D12RHISwapchain::signalSubmitFence(ID3D12CommandQueue* queue, uint64_t fenceVal) {
    auto& sync = m_syncs[m_frameIndex % kFrameCount];
    sync.fenceValue = fenceVal;
    queue->Signal(sync.fence, fenceVal);
}

void D3D12RHISwapchain::recreate() {
    m_device.waitIdle();
    releaseResources();
    m_backBuffers.clear();
    m_syncs.clear();
    m_swapchain = nullptr;
    m_rtvHeap = nullptr;
    createSwapchain(m_width, m_height);
    createRTVs();
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12RHISwapchain::uavHandle(uint32_t index) const {
    return m_uavHandles[index % kFrameCount];
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12RHISwapchain::rtvHandle(uint32_t index) const {
    D3D12_CPU_DESCRIPTOR_HANDLE h = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += static_cast<SIZE_T>(index) * m_rtvDescriptorSize;
    return h;
}

} // namespace rhi
} // namespace somegi
