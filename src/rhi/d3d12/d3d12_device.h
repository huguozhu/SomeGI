// rhi/d3d12/d3d12_device.h — D3D12 后端设备（骨架）
#pragma once
#include "../base/device.h"
#include <memory>

struct ID3D12Device5;
struct ID3D12CommandQueue;
struct IDXGIFactory6;
struct HWND__;
typedef HWND__* HWND;

namespace somegi {
namespace rhi {

// D3D12 后端设备：封装 ID3D12Device + 命令队列 + 交换链工厂
class D3D12RHIDevice : public RHIDevice {
public:
    D3D12RHIDevice();
    ~D3D12RHIDevice() override;

    // ── RHIDevice 接口 ──
    Backend backend() const override { return Backend::D3D12; }
    DeviceLimits limits() const override { return m_limits; }

    std::unique_ptr<RHIBuffer> createBuffer(const BufferDesc& desc) override;
    std::unique_ptr<RHITexture> createTexture(const TextureDesc& desc) override;
    std::unique_ptr<RHITextureView> createTextureView(const RHITexture& tex,
                                                       const TextureViewDesc& desc) override;
    std::unique_ptr<RHIShader> createShader(const ShaderDesc& desc,
                                             const void* bytecode, size_t size) override;
    std::unique_ptr<RHISampler> createSampler(const SamplerDesc& desc) override;
    std::unique_ptr<RHISwapchain> createSwapchain(void* nativeWindow,
                                                   uint32_t width, uint32_t height) override;
    std::unique_ptr<RHIPipelineState> createGraphicsPSO(const GraphicsPSODesc& desc) override;
    std::unique_ptr<RHIPipelineState> createComputePSO(const ComputePSODesc& desc) override;
    std::unique_ptr<RHIPipelineState> createRayTracingPSO(const RayTracingPSODesc& desc) override;
    std::unique_ptr<RHIDescriptorSetLayout> createDescriptorSetLayout(
        const DescSetLayoutDesc& desc) override;
    std::unique_ptr<RHIDescriptorSet> createDescriptorSet(
        const RHIDescriptorSetLayout& layout) override;
    std::unique_ptr<RHICommandPool> createCommandPool() override;
    std::unique_ptr<RHIFence> createFence(bool signaled = false) override;
    std::unique_ptr<RHISemaphore> createSemaphore() override;
    std::unique_ptr<RHIQueryPool> createQueryPool(uint32_t count) override;

    void submit(const SubmitDesc& desc) override;
    void present(const RHISwapchain& swapchain, const RHISemaphore* waitSemaphore) override;
    void waitForFence(const RHIFence& fence, uint64_t timeoutNs) override;
    void waitIdle() override;

    void* nativeDevice() const override { return m_device; }
    void* nativePhysicalDevice() const override { return nullptr; }
    void* nativeQueue() const override { return m_queue; }
    uint32_t queueFamily() const override { return 0; }

    // ── D3D12 特定 ──
    ID3D12Device5*  device()       { return m_device; }
    ID3D12CommandQueue* commandQueue() { return m_queue; }
    IDXGIFactory6*  dxgiFactory()  { return m_factory; }

    static HWND getHwnd(void* nativeWindow);

private:
    ID3D12Device5*      m_device  = nullptr;
    ID3D12CommandQueue* m_queue   = nullptr;
    IDXGIFactory6*      m_factory = nullptr;
    DeviceLimits m_limits{};
};

} // namespace rhi
} // namespace somegi
