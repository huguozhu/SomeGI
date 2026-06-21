// rhi/d3d12/d3d12_texture.cpp — D3D12 纹理/视图/Shader 实现
#include "d3d12_texture.h"
#include "d3d12_device.h"
#include <stdexcept>
#include <cstdio>

namespace somegi {
namespace rhi {

// ── Format 映射 ──
static DXGI_FORMAT toDxgiFormat(Format f) {
    switch (f) {
        case Format::R8_UNORM:            return DXGI_FORMAT_R8_UNORM;
        case Format::R8G8B8A8_UNORM:      return DXGI_FORMAT_R8G8B8A8_UNORM;
        case Format::R16G16B16A16_SFLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case Format::R32_UINT:            return DXGI_FORMAT_R32_UINT;
        case Format::R32_SFLOAT:          return DXGI_FORMAT_R32_FLOAT;
        case Format::R32G32_SFLOAT:       return DXGI_FORMAT_R32G32_FLOAT;
        case Format::D32_SFLOAT:          return DXGI_FORMAT_D32_FLOAT;
        case Format::B8G8R8A8_UNORM:      return DXGI_FORMAT_B8G8R8A8_UNORM;
        default: return DXGI_FORMAT_R8G8B8A8_UNORM;
    }
}

// ════════════════════════════════════════════════════════════════
// D3D12RHITexture
// ════════════════════════════════════════════════════════════════

D3D12RHITexture::D3D12RHITexture(D3D12RHIDevice& device, const TextureDesc& desc)
    : m_device(device), m_desc(desc), m_dxgiFormat(toDxgiFormat(desc.format)) {

    // 推断默认状态
    bool isDS = ((uint32_t)desc.usage & (uint32_t)TextureUsage::DepthStencil) != 0;
    bool isRT = ((uint32_t)desc.usage & (uint32_t)TextureUsage::ColorAttachment) != 0;
    m_defaultState = isDS ? D3D12_RESOURCE_STATE_DEPTH_WRITE
                   : isRT ? D3D12_RESOURCE_STATE_RENDER_TARGET
                   : D3D12_RESOURCE_STATE_COMMON;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = desc.width;
    rd.Height = desc.height;
    rd.DepthOrArraySize = (UINT16)desc.arrayLayers;
    rd.MipLevels = (UINT16)desc.mipLevels;
    rd.Format = m_dxgiFormat;
    rd.SampleDesc = { (UINT)desc.samples, 0 };
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_NONE;

    if (isDS)
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    if (isRT || ((uint32_t)desc.usage & (uint32_t)TextureUsage::Storage))
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
                  | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    if (FAILED(device.device()->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_resource)))) {
        throw std::runtime_error("[d3d12] CreateCommittedResource(texture) failed");
    }

    if (desc.debugName) {
        wchar_t wname[128];
        MultiByteToWideChar(CP_UTF8, 0, desc.debugName, -1, wname, 128);
        m_resource->SetName(wname);
    }
    // 注册初始资源状态
    device.trackResourceState(m_resource, D3D12_RESOURCE_STATE_COMMON);
}

D3D12RHITexture::~D3D12RHITexture() {
    if (m_resource) {
        m_device.removeResourceState(m_resource);
        m_resource->Release();
    }
}

std::unique_ptr<RHITextureView> D3D12RHITexture::createView(const TextureViewDesc& desc) {
    return std::unique_ptr<RHITextureView>(new D3D12RHITextureView(m_device, *this, desc));
}

// ════════════════════════════════════════════════════════════════
// D3D12RHITextureView
// ════════════════════════════════════════════════════════════════

D3D12RHITextureView::D3D12RHITextureView(D3D12RHIDevice& device,
                                           D3D12RHITexture& texture,
                                           const TextureViewDesc& desc) {
    bool isDS = (texture.defaultState() == D3D12_RESOURCE_STATE_DEPTH_WRITE);
    m_isDSV = isDS;
    m_isRTV = !isDS && (texture.defaultState() == D3D12_RESOURCE_STATE_RENDER_TARGET);

    if (m_isDSV && device.cpuDsvHeap()) {
        // 分配 DSV
        static uint32_t dsvIdx = 0;
        m_dsvHandle = device.cpuDsvHeap()->GetCPUDescriptorHandleForHeapStart();
        m_dsvHandle.ptr += dsvIdx++ * device.cpuDsvIncrement();

        D3D12_DEPTH_STENCIL_VIEW_DESC dsvd{};
        dsvd.Format = texture.dxgiFormat();
        dsvd.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        device.device()->CreateDepthStencilView(texture.resource(), &dsvd, m_dsvHandle);
    } else if (m_isRTV && device.cpuRtvHeap()) {
        // 分配 RTV
        static uint32_t rtvIdx = 0;
        m_rtvHandle = device.cpuRtvHeap()->GetCPUDescriptorHandleForHeapStart();
        m_rtvHandle.ptr += rtvIdx++ * device.cpuRtvIncrement();

        D3D12_RENDER_TARGET_VIEW_DESC rtvd{};
        rtvd.Format = texture.dxgiFormat();
        rtvd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device.device()->CreateRenderTargetView(texture.resource(), &rtvd, m_rtvHandle);
    } else if (device.cpuSrvUavHeap()) {
        // 分配 SRV
        static uint32_t srvIdx = 0;
        m_srvHandle = device.cpuSrvUavHeap()->GetCPUDescriptorHandleForHeapStart();
        m_srvHandle.ptr += srvIdx++ * device.cpuSrvIncrement();

        D3D12_SHADER_RESOURCE_VIEW_DESC srvd{};
        srvd.Format = texture.dxgiFormat();
        srvd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvd.Texture2D.MipLevels = desc.mipCount ? desc.mipCount : texture.mipLevels();
        srvd.Texture2D.MostDetailedMip = desc.baseMip;
        device.device()->CreateShaderResourceView(texture.resource(), &srvd, m_srvHandle);
    }
}

D3D12RHITextureView::~D3D12RHITextureView() = default;

// ════════════════════════════════════════════════════════════════

} // namespace rhi
} // namespace somegi
