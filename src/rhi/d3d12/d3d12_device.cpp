// rhi/d3d12/d3d12_device.cpp — D3D12 设备实现（骨架）
#include "d3d12_device.h"
#include "d3d12_shader.h"  // 必须在 D3D12RHIShader 使用之前
#include "d3d12_sampler.h" // 必须在 D3D12RHISampler 使用之前
#include "d3d12_buffer.h"
#include "d3d12_texture.h"
#include "d3d12_sampler.h"
#include "d3d12_shader.h"
#include "d3d12_swapchain.h"
#include "d3d12_command.h"
#include "core/window.h"

// glfwGetWin32Window 声明（避免包含 glfw3native.h，它与 HWND 前向声明冲突）
extern "C" { HWND glfwGetWin32Window(struct GLFWwindow* window); }
#include "d3d12_fence.h"
#include "d3d12_pso.h"
#include "d3d12_descriptor.h"
#include "d3d12_query_pool.h"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cassert>
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
            // 启用 GPU 验证（捕获 PSO/root signature 错误）
            ID3D12Debug3* debug3 = nullptr;
            if (SUCCEEDED(debugCtrl->QueryInterface(IID_PPV_ARGS(&debug3)))) {
                debug3->SetEnableGPUBasedValidation(true);
                debug3->Release();
            }
            debugCtrl->Release();
            std::printf("[d3d12] debug layer + GPU validation enabled\n");
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
    m_limits.maxSampledImages     = 128;
    m_limits.maxUniformBufferSize = 65536;
    m_limits.maxStorageBufferSize = 256 * 1024 * 1024;
    m_limits.maxPushConstantsSize = 256;
    m_limits.timestampPeriod      = 1.0f;
    m_limits.meshShaderSupported  = false;  // D3D12 mesh shader 需要额外能力查询
    m_limits.rayTracingSupported  = false;
    m_limits.maxMeshOutputVertices = 0;
    m_limits.maxMeshOutputPrimitives = 0;
    m_limits.maxMeshWorkGroupInvocations = 0;
    m_limits.supportedSampleCounts = 0xF;  // 1, 2, 4, 8 (假设标准支持)

    std::printf("[d3d12] device created successfully\n");

    // ── 创建 GPU 可见描述符堆 ──
    createDescriptorHeap();

    // ── 创建 CPU 端描述符堆（持久 RTV/DSV/SRV） ──
    {
        D3D12_DESCRIPTOR_HEAP_DESC hdRtv{};
        hdRtv.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        hdRtv.NumDescriptors = 256;
        hdRtv.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        m_device->CreateDescriptorHeap(&hdRtv, IID_PPV_ARGS(&m_cpuRtvHeap));
        m_cpuRtvInc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        D3D12_DESCRIPTOR_HEAP_DESC hdDsv{};
        hdDsv.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        hdDsv.NumDescriptors = 64;
        hdDsv.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        m_device->CreateDescriptorHeap(&hdDsv, IID_PPV_ARGS(&m_cpuDsvHeap));
        m_cpuDsvInc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

        D3D12_DESCRIPTOR_HEAP_DESC hdSrv{};
        hdSrv.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hdSrv.NumDescriptors = 1024;
        hdSrv.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        m_device->CreateDescriptorHeap(&hdSrv, IID_PPV_ARGS(&m_cpuSrvHeap));
        m_cpuSrvInc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        D3D12_DESCRIPTOR_HEAP_DESC hdSmp{};
        hdSmp.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        hdSmp.NumDescriptors = 64;
        hdSmp.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        m_device->CreateDescriptorHeap(&hdSmp, IID_PPV_ARGS(&m_cpuSamplerHeap));
        m_cpuSamplerInc = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    }
}

void D3D12RHIDevice::createDescriptorHeap() {
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = kGpuDescHeapSize;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(m_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_gpuDescHeap)))) {
        throw std::runtime_error("[d3d12] CreateDescriptorHeap(CBV_SRV_UAV) failed");
    }
    m_gpuDescIncrement = m_device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_gpuDescStartCPU = m_gpuDescHeap->GetCPUDescriptorHandleForHeapStart();
    m_gpuDescStartGPU = m_gpuDescHeap->GetGPUDescriptorHandleForHeapStart();
    std::printf("[d3d12] GPU descriptor heap: %u slots\n", kGpuDescHeapSize);

    // ── 创建 GPU 可见采样器描述符堆 ──
    {
        D3D12_DESCRIPTOR_HEAP_DESC hdSmp{};
        hdSmp.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        hdSmp.NumDescriptors = kGpuSamplerHeapSize;
        hdSmp.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(m_device->CreateDescriptorHeap(&hdSmp, IID_PPV_ARGS(&m_gpuSamplerHeap)))) {
            throw std::runtime_error("[d3d12] CreateDescriptorHeap(SAMPLER, SHADER_VISIBLE) failed");
        }
        m_gpuSamplerIncrement = m_device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        m_gpuSamplerStartCPU = m_gpuSamplerHeap->GetCPUDescriptorHandleForHeapStart();
        m_gpuSamplerStartGPU = m_gpuSamplerHeap->GetGPUDescriptorHandleForHeapStart();
        std::printf("[d3d12] GPU sampler heap: %u slots\n", kGpuSamplerHeapSize);
    }
}

D3D12RHIDevice::DescAlloc D3D12RHIDevice::allocDescriptors(uint32_t count) {
    uint32_t offset = m_gpuDescOffset;
    m_gpuDescOffset += count;

    // 确保不超过堆容量（末尾保留区域除外）
    assert(offset + count <= kGpuDescHeapSize - kPersistentDescCount);

    DescAlloc a;
    a.offset = offset;
    a.cpu = m_gpuDescStartCPU;
    a.cpu.ptr += static_cast<SIZE_T>(offset) * m_gpuDescIncrement;
    a.gpu = m_gpuDescStartGPU;
    a.gpu.ptr += static_cast<SIZE_T>(offset) * m_gpuDescIncrement;
    return a;
}

void D3D12RHIDevice::resetDescriptorHeap() {
    m_gpuDescOffset = 0;
}

// 持久描述符分配（堆末尾保留区域，不受帧重置影响）
// 用于 ImGui 字体 SRV 等需要跨帧存活的描述符
D3D12RHIDevice::DescAlloc D3D12RHIDevice::allocPersistentDescriptors(uint32_t count) {
    uint32_t baseSlot = kGpuDescHeapSize - kPersistentDescCount;
    uint32_t offset = baseSlot + m_persistentDescCursor;
    m_persistentDescCursor += count;
    assert(m_persistentDescCursor <= kPersistentDescCount && "持久描述符槽位不足");

    DescAlloc a;
    a.offset = offset;
    a.cpu = m_gpuDescStartCPU;
    a.cpu.ptr += static_cast<SIZE_T>(offset) * m_gpuDescIncrement;
    a.gpu = m_gpuDescStartGPU;
    a.gpu.ptr += static_cast<SIZE_T>(offset) * m_gpuDescIncrement;
    return a;
}

D3D12RHIDevice::DescAlloc D3D12RHIDevice::allocSamplerDescriptors(uint32_t count) {
    uint32_t offset = m_gpuSamplerOffset;
    m_gpuSamplerOffset += count;

    // Ensure we don't exceed heap capacity
    assert(offset + count <= kGpuSamplerHeapSize);

    DescAlloc a;
    a.offset = offset;
    a.cpu = m_gpuSamplerStartCPU;
    a.cpu.ptr += static_cast<SIZE_T>(offset) * m_gpuSamplerIncrement;
    a.gpu = m_gpuSamplerStartGPU;
    a.gpu.ptr += static_cast<SIZE_T>(offset) * m_gpuSamplerIncrement;
    return a;
}

void D3D12RHIDevice::resetSamplerHeap() {
    m_gpuSamplerOffset = 0;
}

void D3D12RHIDevice::trackResourceState(ID3D12Resource* res, D3D12_RESOURCE_STATES state) {
    m_resourceStates[res] = state;
}

D3D12_RESOURCE_STATES D3D12RHIDevice::getResourceState(ID3D12Resource* res) const {
    auto it = m_resourceStates.find(res);
    return (it != m_resourceStates.end()) ? it->second : D3D12_RESOURCE_STATE_COMMON;
}

void D3D12RHIDevice::removeResourceState(ID3D12Resource* res) {
    m_resourceStates.erase(res);
}

D3D12RHIDevice::~D3D12RHIDevice() {
    waitIdle();
    if (m_gpuDescHeap) { m_gpuDescHeap->Release(); }
    if (m_cpuRtvHeap)  { m_cpuRtvHeap->Release(); }
    if (m_cpuDsvHeap)  { m_cpuDsvHeap->Release(); }
    if (m_cpuSrvHeap)     { m_cpuSrvHeap->Release(); }
    if (m_cpuSamplerHeap) { m_cpuSamplerHeap->Release(); }
    if (m_gpuSamplerHeap) { m_gpuSamplerHeap->Release(); }
    if (m_queue)   { m_queue->Release(); }
    if (m_device)  { m_device->Release(); }
    if (m_factory) { m_factory->Release(); }
}

// ════════════════════════════════════════════════════════════════
// 资源创建（Phase 2+ 实现）
// ════════════════════════════════════════════════════════════════

#define NOT_IMPL(msg) throw std::runtime_error("[d3d12] " msg " not implemented")

std::unique_ptr<RHIBuffer> D3D12RHIDevice::createBuffer(const BufferDesc& desc) {
    return std::unique_ptr<RHIBuffer>(new D3D12RHIBuffer(*this, desc));
}
std::unique_ptr<RHITexture> D3D12RHIDevice::createTexture(const TextureDesc& desc) {
    return std::unique_ptr<RHITexture>(new D3D12RHITexture(*this, desc));
}
std::unique_ptr<RHITextureView> D3D12RHIDevice::createTextureView(const RHITexture& tex, const TextureViewDesc& desc) {
    // createView 非 const（因可能创建 descriptor handle），安全的 const_cast
    return const_cast<D3D12RHITexture&>(static_cast<const D3D12RHITexture&>(tex)).createView(desc);
}
std::unique_ptr<RHIShader> D3D12RHIDevice::createShader(const ShaderDesc& desc, const void* bytecode, size_t size) {
    return std::make_unique<D3D12RHIShader>(desc, bytecode, size);
}
std::unique_ptr<RHISampler> D3D12RHIDevice::createSampler(const SamplerDesc& desc) {
    return std::make_unique<D3D12RHISampler>(*this, desc);
}
std::unique_ptr<RHISwapchain> D3D12RHIDevice::createSwapchain(void* nativeWindow, uint32_t w, uint32_t h) {
    return std::unique_ptr<RHISwapchain>(new D3D12RHISwapchain(*this, nativeWindow, w, h));
}
std::unique_ptr<RHIPipelineState> D3D12RHIDevice::createGraphicsPSO(const GraphicsPSODesc& desc) {
    return std::unique_ptr<RHIPipelineState>(new D3D12RHIPipelineState(*this, desc));
}
std::unique_ptr<RHIPipelineState> D3D12RHIDevice::createComputePSO(const ComputePSODesc& desc) {
    return std::unique_ptr<RHIPipelineState>(new D3D12RHIPipelineState(*this, desc));
}
std::unique_ptr<RHIPipelineState> D3D12RHIDevice::createRayTracingPSO(const RayTracingPSODesc&) {
    NOT_IMPL("createRayTracingPSO");
}
std::unique_ptr<RHIDescriptorSetLayout> D3D12RHIDevice::createDescriptorSetLayout(const DescSetLayoutDesc& desc) {
    return std::unique_ptr<RHIDescriptorSetLayout>(new D3D12RHIDescriptorSetLayout(desc));
}
std::unique_ptr<RHIDescriptorSet> D3D12RHIDevice::createDescriptorSet(const RHIDescriptorSetLayout& layout) {
    return std::unique_ptr<RHIDescriptorSet>(new D3D12RHIDescriptorSet(
        *this, const_cast<D3D12RHIDescriptorSetLayout&>(
            static_cast<const D3D12RHIDescriptorSetLayout&>(layout))));
}
std::unique_ptr<RHICommandPool> D3D12RHIDevice::createCommandPool() {
    return std::unique_ptr<RHICommandPool>(new D3D12RHICommandPool(*this));
}
std::unique_ptr<RHIFence> D3D12RHIDevice::createFence(bool signaled) {
    return std::unique_ptr<RHIFence>(new D3D12RHIFence(*this, signaled));
}
std::unique_ptr<RHISemaphore> D3D12RHIDevice::createSemaphore() {
    return std::unique_ptr<RHISemaphore>(new D3D12RHISemaphore());
}
std::unique_ptr<RHIQueryPool> D3D12RHIDevice::createQueryPool(uint32_t count) {
    return std::make_unique<D3D12RHIQueryPool>(*this, count);
}

// ════════════════════════════════════════════════════════════════
// 提交 / 呈现 / 同步
// ════════════════════════════════════════════════════════════════

void D3D12RHIDevice::submit(const SubmitDesc& desc) {
    ID3D12CommandList* lists[] = {
        static_cast<D3D12RHICommandBuffer*>(
            const_cast<RHICommandBuffer*>(desc.commandBuffer))->cmdList()
    };
    m_queue->ExecuteCommandLists(1, lists);
    if (desc.signalFence) {
        auto* fence = static_cast<D3D12RHIFence*>(
            const_cast<RHIFence*>(desc.signalFence));
        m_queue->Signal(fence->fence(), ++fence->value());
    }
}

void D3D12RHIDevice::present(const RHISwapchain& swapchain, const RHISemaphore*) {
    auto& d3dSwap = static_cast<const D3D12RHISwapchain&>(swapchain);
    // 提交命令队列前先确保所有提交完成
    // Phase 5: 使用 fence 精确同步
    d3dSwap.presentCurrentFrame();
    resetDescriptorHeap();  // 每帧重置描述符堆
    resetSamplerHeap();     // 每帧重置采样器堆
}

void D3D12RHIDevice::waitForFence(const RHIFence& fence, uint64_t timeoutNs) {
    auto& d3d12Fence = static_cast<const D3D12RHIFence&>(fence);
    const_cast<D3D12RHIFence&>(d3d12Fence).wait(timeoutNs);
}

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
    // nativeWindow 是 Window*，需要通过 GLFW 获取真正的 Win32 HWND
    auto* win = static_cast<somegi::Window*>(nativeWindow);
    return glfwGetWin32Window(win->handle());
}

} // namespace rhi
} // namespace somegi
