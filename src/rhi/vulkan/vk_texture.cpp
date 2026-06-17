// rhi/vulkan/vk_texture.cpp
#include "vk_texture.h"
#include <VulkanMemoryAllocator/vk_mem_alloc.h>

namespace somegi {
namespace rhi {

static VkFormat toVkFormat(Format f) {
    switch (f) {
        case Format::R8_UNORM:          return VK_FORMAT_R8_UNORM;
        case Format::R8G8B8A8_UNORM:    return VK_FORMAT_R8G8B8A8_UNORM;
        case Format::R16G16B16A16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case Format::R32_UINT:          return VK_FORMAT_R32_UINT;
        case Format::R32_SFLOAT:        return VK_FORMAT_R32_SFLOAT;
        case Format::D32_SFLOAT:        return VK_FORMAT_D32_SFLOAT;
        case Format::B8G8R8A8_UNORM:    return VK_FORMAT_B8G8R8A8_UNORM;
        default: return VK_FORMAT_R8G8B8A8_UNORM;
    }
}

static VkImageUsageFlags toVkUsage(TextureUsage u) {
    VkImageUsageFlags f = 0;
    if ((uint32_t)u & (uint32_t)TextureUsage::Sampled)         f |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if ((uint32_t)u & (uint32_t)TextureUsage::Storage)         f |= VK_IMAGE_USAGE_STORAGE_BIT;
    if ((uint32_t)u & (uint32_t)TextureUsage::ColorAttachment) f |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if ((uint32_t)u & (uint32_t)TextureUsage::DepthStencil)    f |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if ((uint32_t)u & (uint32_t)TextureUsage::TransferSrc)     f |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if ((uint32_t)u & (uint32_t)TextureUsage::TransferDst)     f |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return f;
}

VkRHITexture::VkRHITexture(VkRHIDevice& d, const TextureDesc& desc) : m_device(d), m_desc(desc) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = toVkFormat(desc.format);
    ci.extent = {desc.width, desc.height, desc.depth};
    ci.mipLevels = desc.mipLevels;
    ci.arrayLayers = desc.arrayLayers;
    ci.samples = (VkSampleCountFlagBits)desc.samples;
    ci.usage = toVkUsage(desc.usage);
    if (desc.isCubemap) ci.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vmaCreateImage(d.vma(), &ci, &ai, &m_image, &m_allocation, nullptr);
}

std::unique_ptr<RHITexture> VkRHITexture::create(VkRHIDevice& device, const TextureDesc& desc) {
    return std::unique_ptr<RHITexture>(new VkRHITexture(device, desc));
}

VkRHITexture::~VkRHITexture() { if (m_image) vmaDestroyImage(m_device.vma(), m_image, m_allocation); }

std::unique_ptr<RHITextureView> VkRHITexture::createView(const TextureViewDesc& desc) {
    return VkRHITextureView::create(m_device, *this, desc);
}

// View
std::unique_ptr<RHITextureView> VkRHITextureView::create(VkRHIDevice& device, const RHITexture& tex, const TextureViewDesc& desc) {
    auto v = std::unique_ptr<VkRHITextureView>(new VkRHITextureView(device));
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ci.image = (VkImage)(uintptr_t)tex.nativeHandle();
    ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ci.format = desc.format != Format::Unknown ? toVkFormat(desc.format) : toVkFormat(tex.format());
    ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, desc.baseMip, desc.mipCount, desc.baseLayer, desc.layerCount};
    vkCreateImageView(device.vkDevice(), &ci, nullptr, &v->m_view);
    return v;
}

VkRHITextureView::~VkRHITextureView() { if (m_view) vkDestroyImageView(m_device.vkDevice(), m_view, nullptr); }

} // namespace rhi
} // namespace somegi
