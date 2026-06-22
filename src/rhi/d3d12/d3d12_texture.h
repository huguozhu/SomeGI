// rhi/d3d12/d3d12_texture.h — D3D12 纹理和纹理视图
#pragma once
#include "../base/texture.h"
#include "../base/shader.h"
#include "../base/sampler.h"
#include <d3d12.h>

// 共享 D3D12 映射函数
namespace somegi { namespace rhi {
inline D3D12_COMPARISON_FUNC toD3D12Cmp(CompareFunc f) {
    switch (f) {
        case CompareFunc::Never:        return D3D12_COMPARISON_FUNC_NEVER;
        case CompareFunc::Less:         return D3D12_COMPARISON_FUNC_LESS;
        case CompareFunc::Equal:        return D3D12_COMPARISON_FUNC_EQUAL;
        case CompareFunc::LessEqual:    return D3D12_COMPARISON_FUNC_LESS_EQUAL;
        case CompareFunc::Greater:      return D3D12_COMPARISON_FUNC_GREATER;
        case CompareFunc::NotEqual:     return D3D12_COMPARISON_FUNC_NOT_EQUAL;
        case CompareFunc::GreaterEqual: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
        case CompareFunc::Always:       return D3D12_COMPARISON_FUNC_ALWAYS;
        default: return D3D12_COMPARISON_FUNC_LESS;
    }
}
}}

namespace somegi {
namespace rhi {

class D3D12RHIDevice;

// D3D12 纹理
class D3D12RHITexture : public RHITexture {
public:
    D3D12RHITexture(D3D12RHIDevice& device, const TextureDesc& desc);
    ~D3D12RHITexture() override;

    // 非拥有型包装：不创建/销毁 ID3D12Resource，用于包装 swapchain back buffer
    static std::unique_ptr<RHITexture> createNonOwning(D3D12RHIDevice& device, ID3D12Resource* resource,
                                                        Format format, uint32_t width, uint32_t height,
                                                        uint32_t mipLevels = 1);

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
    bool m_ownsResource = true;  // 是否在析构时释放资源

    // 空构造函（createNonOwning 专用，不创建资源）
    explicit D3D12RHITexture(D3D12RHIDevice& d) : m_device(d) {}
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

#include "d3d12_sampler.h"

#include "d3d12_shader.h"

} // namespace rhi
} // namespace somegi
