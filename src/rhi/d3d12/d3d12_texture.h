// rhi/d3d12/d3d12_texture.h — D3D12 纹理和纹理视图
#pragma once
#include "../base/texture.h"
#include "../base/shader.h"
#include "../base/sampler.h"
#include <d3d12.h>

namespace somegi {
namespace rhi {

class D3D12RHIDevice;

// D3D12 纹理
class D3D12RHITexture : public RHITexture {
public:
    D3D12RHITexture(D3D12RHIDevice& device, const TextureDesc& desc);
    ~D3D12RHITexture() override;

    std::unique_ptr<RHITextureView> createView(const TextureViewDesc& desc) override;
    Format format() const override { return m_desc.format; }
    uint32_t width() const override  { return m_desc.width; }
    uint32_t height() const override { return m_desc.height; }
    uint32_t mipLevels() const override { return m_desc.mipLevels; }
    void* nativeHandle() const override { return (void*)m_resource; }

    ID3D12Resource* resource() const { return m_resource; }
    D3D12_RESOURCE_STATES defaultState() const { return m_defaultState; }
    DXGI_FORMAT dxgiFormat() const { return m_dxgiFormat; }

private:
    D3D12RHIDevice& m_device;
    ID3D12Resource* m_resource = nullptr;
    TextureDesc m_desc{};
    DXGI_FORMAT m_dxgiFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    D3D12_RESOURCE_STATES m_defaultState = D3D12_RESOURCE_STATE_COMMON;
};

// D3D12 纹理视图（即 SRV/RTV/DSV/UAV descriptor handle）
class D3D12RHITextureView : public RHITextureView {
public:
    D3D12RHITextureView() = default;  // 用于 swapchain back buffer 视图
    D3D12RHITextureView(D3D12RHIDevice& device, D3D12RHITexture& texture,
                         const TextureViewDesc& desc);
    ~D3D12RHITextureView() override;
    void* nativeHandle() const override { return (void*)(uintptr_t)m_srvHandle.ptr; }

    D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle() const { return m_srvHandle; }
    D3D12_CPU_DESCRIPTOR_HANDLE rtvCpuHandle() const { return m_rtvHandle; }
    D3D12_CPU_DESCRIPTOR_HANDLE dsvCpuHandle() const { return m_dsvHandle; }
    D3D12_GPU_DESCRIPTOR_HANDLE srvGpuHandle() const { return m_srvGpuHandle; }
    bool isRenderTarget() const { return m_isRTV; }
    bool isDepthStencil() const { return m_isDSV; }

private:
    friend class D3D12RHISwapchain;
    friend class D3D12RHICommandBuffer;
    D3D12_CPU_DESCRIPTOR_HANDLE m_srvHandle{};
    D3D12_CPU_DESCRIPTOR_HANDLE m_rtvHandle{};
    D3D12_CPU_DESCRIPTOR_HANDLE m_dsvHandle{};
    D3D12_GPU_DESCRIPTOR_HANDLE m_srvGpuHandle{};
    bool m_isRTV = false;
    bool m_isDSV = false;
};

// D3D12 采样器
class D3D12RHISampler : public RHISampler {
public:
    D3D12RHISampler(const SamplerDesc& desc);
    void* nativeHandle() const override { return (void*)(uintptr_t)m_cpuHandle.ptr; }
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle() const { return m_cpuHandle; }
private:
    D3D12_CPU_DESCRIPTOR_HANDLE m_cpuHandle{};
};

// D3D12 Shader（DXIL bytecode 包装）
class D3D12RHIShader : public RHIShader {
public:
    D3D12RHIShader(const ShaderDesc& desc, const void* bytecode, size_t size);
    ~D3D12RHIShader() override;
    ShaderStage stage() const override { return m_stage; }
    const char* entryPoint() const override { return m_entryPoint.c_str(); }
    void* nativeHandle() const override { return (void*)m_bytecode.data(); }
    const void* bytecodeData() const { return m_bytecode.data(); }
    size_t bytecodeSize() const { return m_bytecode.size(); }

private:
    ShaderStage m_stage;
    std::string m_entryPoint;
    std::vector<uint8_t> m_bytecode;
};

} // namespace rhi
} // namespace somegi
