#include "image.h"
#include "device.h"
#include "allocator.h"
#include <utility>

namespace somegi {

Image::Image(Device& d, const ImageDesc& desc) : m_device(&d), m_desc(desc) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.flags = desc.flags;
    ci.imageType = desc.type;
    ci.format = desc.format;
    ci.extent = desc.extent;
    ci.mipLevels = desc.mipLevels;
    ci.arrayLayers = desc.arrayLayers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = desc.usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(d.device(), &ci, nullptr, &m_image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(d.device(), m_image, &req);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(d.physicalDevice(), req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(d.device(), &ai, nullptr, &m_memory));
    VK_CHECK(vkBindImageMemory(d.device(), m_image, m_memory, 0));

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
    VK_CHECK(vkCreateImageView(d.device(), &vi, nullptr, &m_view));
}

Image::~Image() { reset(); }

void Image::swap(Image& o) noexcept {
    std::swap(m_device, o.m_device);
    std::swap(m_image, o.m_image);
    std::swap(m_view, o.m_view);
    std::swap(m_memory, o.m_memory);
    std::swap(m_desc, o.m_desc);
}

void Image::reset() {
    if (m_device) {
        if (m_view)   vkDestroyImageView(m_device->device(), m_view, nullptr);
        if (m_image)  vkDestroyImage(m_device->device(), m_image, nullptr);
        if (m_memory) vkFreeMemory(m_device->device(), m_memory, nullptr);
    }
    m_device = nullptr;
    m_image = VK_NULL_HANDLE; m_view = VK_NULL_HANDLE; m_memory = VK_NULL_HANDLE;
    m_desc = {};
}

}
