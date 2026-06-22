#include "image.h"
#include "device.h"
#include "rhi/vulkan/vk_device.h"
#include <utility>

namespace somegi {

Image::Image(rhi::VkRHIDevice& vkDev, const ImageDesc& desc) : m_rhiDev(&vkDev), m_desc(desc) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.flags = desc.flags;
    ci.imageType = desc.type;
    ci.format = desc.format;
    ci.extent = desc.extent;
    ci.mipLevels = desc.mipLevels;
    ci.arrayLayers = desc.arrayLayers;
    ci.samples = desc.samples;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = desc.usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VK_CHECK(vmaCreateImage(m_rhiDev->vma(), &ci, &aci, &m_image, &m_allocation, nullptr));

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = m_image;
    if (desc.arrayLayers == 6 && (desc.flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT))
        vi.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    else if (desc.type == VK_IMAGE_TYPE_2D && desc.arrayLayers > 1)
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    else if (desc.type == VK_IMAGE_TYPE_2D)
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    else
        vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vi.format = desc.format;
    vi.subresourceRange = {desc.aspect, 0, desc.mipLevels, 0, desc.arrayLayers};
    VK_CHECK(vkCreateImageView(m_rhiDev->vkDevice(), &vi, nullptr, &m_view));
}

Image::Image(Device& d, const ImageDesc& desc) : m_rhiDev(&d.rhiDev()), m_desc(desc) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.flags = desc.flags;
    ci.imageType = desc.type;
    ci.format = desc.format;
    ci.extent = desc.extent;
    ci.mipLevels = desc.mipLevels;
    ci.arrayLayers = desc.arrayLayers;
    ci.samples = desc.samples;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = desc.usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VK_CHECK(vmaCreateImage(m_rhiDev->vma(), &ci, &aci, &m_image, &m_allocation, nullptr));

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = m_image;
    if (desc.arrayLayers == 6 && (desc.flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT))
        vi.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    else if (desc.type == VK_IMAGE_TYPE_2D && desc.arrayLayers > 1)
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    else if (desc.type == VK_IMAGE_TYPE_2D)
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    else
        vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vi.format = desc.format;
    vi.subresourceRange = {desc.aspect, 0, desc.mipLevels, 0, desc.arrayLayers};
    VK_CHECK(vkCreateImageView(m_rhiDev->vkDevice(), &vi, nullptr, &m_view));
}

Image::~Image() { reset(); }

void Image::swap(Image& o) noexcept {
    std::swap(m_rhiDev, o.m_rhiDev);
    std::swap(m_image, o.m_image);
    std::swap(m_view, o.m_view);
    std::swap(m_allocation, o.m_allocation);
    std::swap(m_desc, o.m_desc);
}

void Image::reset() {
    if (m_rhiDev) {
        if (m_view)  vkDestroyImageView(m_rhiDev->vkDevice(), m_view, nullptr);
        if (m_image) vmaDestroyImage(m_rhiDev->vma(), m_image, m_allocation);
    }
    m_rhiDev = nullptr;
    m_image = VK_NULL_HANDLE; m_view = VK_NULL_HANDLE; m_allocation = VK_NULL_HANDLE;
    m_desc = {};
}

}
