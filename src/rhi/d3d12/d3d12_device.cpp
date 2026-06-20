// rhi/d3d12/d3d12_device.cpp — D3D12 设备实现（骨架）
#include "d3d12_device.h"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <stdexcept>
#include <cstdio>

namespace somegi {
namespace rhi {

// ════════════════════════════════════════════════════════════════
// 构造 / 析构
// ════════════════════════════════════════════════════════════════

D3D12RHIDevice::D3D12RHIDevice() {
    // ── 启用 D3D12 调试层 ──
#if defined(_DEBUG)
    {
        ID3D12Debug* debugCtrl = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugCtrl)))) {
            debugCtrl->EnableDebugLayer();
            debugCtrl->Release();
            std::printf("[d3d12] debug layer enabled\n");
        }
    }
#endif

    // ── 创建 DXGI 工厂 ──
    UINT factoryFlags = 0;
#if defined(_DEBUG)
    factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif
    if (FAILED(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&m_factory)))) {
        throw std::runtime_error("[d3d12] CreateDXGIFactory2 failed");
    }

    // ── 选择适配器 ──
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0;
         m_factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
         ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            adapter->Release();
            adapter = nullptr;
            continue;
        }
        std::printf("[d3d12] adapter: %S (VRAM %llu MB)\n",
                    desc.Description,
                    (unsigned long long)(desc.DedicatedVideoMemory / (1024 * 1024)));
        break;
    }

    if (!adapter) {
        throw std::runtime_error("[d3d12] no hardware adapter found");
    }

    // ── 创建设备 ──
    HRESULT hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&m_device));
    if (FAILED(hr)) {
        hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device));
        if (SUCCEEDED(hr))
            std::printf("[d3d12] using feature level 12.0\n");
    } else {
        std::printf("[d3d12] using feature level 12.1\n");
    }
    adapter->Release();

    if (FAILED(hr)) {
        throw std::runtime_error("[d3d12] D3D12CreateDevice failed");
    }

    // ── 创建命令队列 ──
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    if (FAILED(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queue)))) {
        throw std::runtime_error("[d3d12] CreateCommandQueue failed");
    }

    // ── 填充设备限制 ──
    m_limits.maxTextureSize       = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
    m_limits.maxSampledTextures   = 128;
    m_limits.maxUniformBufferSize = 65536;
    m_limits.maxStorageBufferSize = 256 * 1024 * 1024;
    m_limits.maxPushConstantsSize = 256;
    m_limits.timestampPeriod      = 1.0f;
    m_limits.meshShaderSupported  = false;
    m_limits.rayTracingSupported  = false;

    std::printf("[d3d12] device created successfully\n");
}

D3D12RHIDevice::~D3D12RHIDevice() {
    waitIdle();
    if (m_queue)   { m_queue->Release(); }
    if (m_device)  { m_device->Release(); }
    if (m_factory) { m_factory->Release(); }
}

// ════════════════════════════════════════════════════════════════
// 资源创建（Phase 2+ 实现）
// ════════════════════════════════════════════════════════════════

#define NOT_IMPL(msg) throw std::runtime_error("[d3d12] " msg " not implemented")

std::unique_ptr<RHIBuffer> D3D12RHIDevice::createBuffer(const BufferDesc&) { NOT_IMPL("createBuffer"); }
std::unique_ptr<RHITexture> D3D12RHIDevice::createTexture(const TextureDesc&) { NOT_IMPL("createTexture"); }
std::unique_ptr<RHITextureView> D3D12RHIDevice::createTextureView(const RHITexture&, const TextureViewDesc&) { NOT_IMPL("createTextureView"); }
std::unique_ptr<RHIShader> D3D12RHIDevice::createShader(const ShaderDesc&, const void*, size_t) { NOT_IMPL("createShader"); }
std::unique_ptr<RHISampler> D3D12RHIDevice::createSampler(const SamplerDesc&) { NOT_IMPL("createSampler"); }
std::unique_ptr<RHISwapchain> D3D12RHIDevice::createSwapchain(void*, uint32_t, uint32_t) { NOT_IMPL("createSwapchain"); }
std::unique_ptr<RHIPipelineState> D3D12RHIDevice::createGraphicsPSO(const GraphicsPSODesc&) { NOT_IMPL("createGraphicsPSO"); }
std::unique_ptr<RHIPipelineState> D3D12RHIDevice::createComputePSO(const ComputePSODesc&) { NOT_IMPL("createComputePSO"); }
std::unique_ptr<RHIPipelineState> D3D12RHIDevice::createRayTracingPSO(const RayTracingPSODesc&) { NOT_IMPL("createRayTracingPSO"); }
std::unique_ptr<RHIDescriptorSetLayout> D3D12RHIDevice::createDescriptorSetLayout(const DescSetLayoutDesc&) { NOT_IMPL("createDescriptorSetLayout"); }
std::unique_ptr<RHIDescriptorSet> D3D12RHIDevice::createDescriptorSet(const RHIDescriptorSetLayout&) { NOT_IMPL("createDescriptorSet"); }
std::unique_ptr<RHICommandPool> D3D12RHIDevice::createCommandPool() { NOT_IMPL("createCommandPool"); }
std::unique_ptr<RHIFence> D3D12RHIDevice::createFence(bool) { NOT_IMPL("createFence"); }
std::unique_ptr<RHISemaphore> D3D12RHIDevice::createSemaphore() { NOT_IMPL("createSemaphore"); }
std::unique_ptr<RHIQueryPool> D3D12RHIDevice::createQueryPool(uint32_t) { NOT_IMPL("createQueryPool"); }

// ════════════════════════════════════════════════════════════════
// 提交 / 呈现 / 同步
// ════════════════════════════════════════════════════════════════

void D3D12RHIDevice::submit(const SubmitDesc&) { NOT_IMPL("submit"); }
void D3D12RHIDevice::present(const RHISwapchain&, const RHISemaphore*) { NOT_IMPL("present"); }
void D3D12RHIDevice::waitForFence(const RHIFence&, uint64_t) { NOT_IMPL("waitForFence"); }

void D3D12RHIDevice::waitIdle() {
    if (m_queue && m_device) {
        ID3D12Fence* fence = nullptr;
        m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        m_queue->Signal(fence, 1);
        HANDLE event = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
        fence->SetEventOnCompletion(1, event);
        WaitForSingleObject(event, INFINITE);
        CloseHandle(event);
        fence->Release();
    }
}

// ════════════════════════════════════════════════════════════════
// 辅助
// ════════════════════════════════════════════════════════════════

HWND D3D12RHIDevice::getHwnd(void* nativeWindow) {
    return static_cast<HWND>(nativeWindow);
}

} // namespace rhi
} // namespace somegi
