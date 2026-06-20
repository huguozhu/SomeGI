// rhi/d3d12/d3d12_swapchain.h — D3D12 交换链
#pragma once
#include "../base/swapchain.h"
#include "../base/texture.h"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <vector>
#include <memory>

namespace somegi {
namespace rhi {

class D3D12RHIDevice;

// D3D12 交换链：封装 IDXGISwapChain4 + RTV 描述符堆 + 每帧同步对象
class D3D12RHISwapchain : public RHISwapchain {
public:
    D3D12RHISwapchain(D3D12RHIDevice& device, void* nativeWindow,
                      uint32_t width, uint32_t height);
    ~D3D12RHISwapchain() override;

    // ── RHISwapchain 接口 ──
    SwapchainFrame acquireNextFrame() override;
    void present(const SwapchainFrame& frame) override;
    void recreate() override;

    Format format() const override { return m_format; }
    uint32_t width() const override  { return m_width; }
    uint32_t height() const override { return m_height; }
    bool hdrAvailable() const override { return false; }
    bool hdrEnabled() const override   { return false; }
    void setHdrEnabled(bool) override  {}

    // ── D3D12 特定 ──
    IDXGISwapChain4* dxgiSwapchain() { return m_swapchain; }
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle(uint32_t index) const;
    ID3D12Resource* backBuffer(uint32_t index) const { return m_backBuffers[index]; }
    uint32_t backBufferCount() const { return (uint32_t)m_backBuffers.size(); }
    // 当前帧 back buffer 的 RTV descriptor handle
    D3D12_CPU_DESCRIPTOR_HANDLE currentRTV(uint32_t frameInFlight) const {
        return rtvHandle(frameInFlight % kFrameCount);
    }

private:
    void createSwapchain(uint32_t width, uint32_t height);
    void createRTVs();
    void releaseResources();

    D3D12RHIDevice& m_device;
    HWND m_hwnd = nullptr;
    IDXGISwapChain4* m_swapchain = nullptr;
    std::vector<ID3D12Resource*> m_backBuffers;

    // RTV 描述符堆
    ID3D12DescriptorHeap* m_rtvHeap = nullptr;
    uint32_t m_rtvDescriptorSize = 0;

    // 同步对象
    struct FrameSync {
        ID3D12Fence*   fence = nullptr;
        HANDLE         fenceEvent = nullptr;
        uint64_t       fenceValue = 0;
    };
    std::vector<FrameSync> m_syncs;

    Format m_format = Format::B8G8R8A8_UNORM;
    uint32_t m_width = 0, m_height = 0;
    uint32_t m_frameIndex = 0;
    static constexpr uint32_t kFrameCount = 2; // 双缓冲
};

} // namespace rhi
} // namespace somegi
