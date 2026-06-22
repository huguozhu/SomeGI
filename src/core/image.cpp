#include "image.h"
#include "device.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_texture.h"
#include <utility>

namespace somegi {

static rhi::Format toRhiFormat(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UNORM:                   return rhi::Format::R8_UNORM;
        case VK_FORMAT_R8G8B8A8_UNORM:             return rhi::Format::R8G8B8A8_UNORM;
        case VK_FORMAT_R16_SFLOAT:                 return rhi::Format::R16_SFLOAT;
        case VK_FORMAT_R16G16_SFLOAT:              return rhi::Format::R16G16_SFLOAT;
        case VK_FORMAT_R16G16B16A16_SFLOAT:        return rhi::Format::R16G16B16A16_SFLOAT;
        case VK_FORMAT_R32_UINT:                   return rhi::Format::R32_UINT;
        case VK_FORMAT_R32_SFLOAT:                 return rhi::Format::R32_SFLOAT;
        case VK_FORMAT_R32G32_SFLOAT:              return rhi::Format::R32G32_SFLOAT;
        case VK_FORMAT_R32G32B32A32_UINT:          return rhi::Format::R32G32B32A32_UINT;
        case VK_FORMAT_D32_SFLOAT:                 return rhi::Format::D32_SFLOAT;
        case VK_FORMAT_R8G8B8A8_SRGB:              return rhi::Format::R8G8B8A8_SRGB;
        case VK_FORMAT_R32G32B32A32_SFLOAT:        return rhi::Format::R32G32B32A32_SFLOAT;
        case VK_FORMAT_B8G8R8A8_UNORM:             return rhi::Format::B8G8R8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_SRGB:              return rhi::Format::B8G8R8A8_SRGB;
        default: return rhi::Format::Unknown;
    }
}

static rhi::TextureDesc toRhiDesc(const ImageDesc& d) {
    rhi::TextureDesc td;
    td.format    = toRhiFormat(d.format);
    td.width     = d.extent.width;
    td.height    = d.extent.height;
    td.depth     = d.extent.depth;
    td.mipLevels = d.mipLevels;
    td.arrayLayers = d.arrayLayers;
    td.samples   = d.samples;
    td.isCubemap = (d.flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0;
    td.debugName = "Image";
    // 映射 usage
    rhi::TextureUsage u = rhi::TextureUsage::Sampled;
    if (d.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)     u = u | rhi::TextureUsage::ColorAttachment;
    if (d.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) u = u | rhi::TextureUsage::DepthStencil;
    if (d.usage & VK_IMAGE_USAGE_STORAGE_BIT)              u = u | rhi::TextureUsage::Storage;
    if (d.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)         u = u | rhi::TextureUsage::TransferSrc;
    if (d.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)         u = u | rhi::TextureUsage::TransferDst;
    td.usage = u;
    return td;
}

void Image::init(rhi::RHIDevice& device, const ImageDesc& desc) {
    m_desc = desc;
    m_rhiTexture = device.createTexture(toRhiDesc(desc));
    rhi::TextureViewDesc vd;
    vd.format   = toRhiFormat(desc.format);
    vd.mipCount = desc.mipLevels;
    vd.layerCount = desc.arrayLayers;
    if (desc.arrayLayers == 6 && (desc.flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT))
        vd.viewType = rhi::TextureViewType::TextureCube;
    else if (desc.arrayLayers > 1)
        vd.viewType = rhi::TextureViewType::Texture2DArray;
    m_rhiView = device.createTextureView(*m_rhiTexture, vd);
}

Image::Image(rhi::VkRHIDevice& vkDev, const ImageDesc& desc) { init(vkDev, desc); }

Image::Image(Device& d, const ImageDesc& desc) { init(d.rhiDev(), desc); }

Image::~Image() = default;

void Image::swap(Image& o) noexcept {
    std::swap(m_rhiTexture, o.m_rhiTexture);
    std::swap(m_rhiView, o.m_rhiView);
    std::swap(m_desc, o.m_desc);
}

void Image::reset() {
    m_rhiView.reset();
    m_rhiTexture.reset();
    m_desc = {};
}

VkImage Image::image() const {
    return m_rhiTexture ? (VkImage)(uintptr_t)m_rhiTexture->nativeHandle() : VK_NULL_HANDLE;
}

VkImageView Image::view() const {
    return m_rhiView ? (VkImageView)(uintptr_t)m_rhiView->nativeHandle() : VK_NULL_HANDLE;
}

}
