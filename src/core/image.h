#pragma once
#include "vk_common.h"

namespace somegi {
class Device;

struct ImageDesc {
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkExtent3D extent = {1, 1, 1};
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkImageType type = VK_IMAGE_TYPE_2D;
    VkImageUsageFlags usage = 0;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageCreateFlags flags = 0;
};

class Image {
public:
    Image() = default;
    Image(Device& d, const ImageDesc& desc);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& o) noexcept { swap(o); }
    Image& operator=(Image&& o) noexcept { if (this != &o) { reset(); swap(o); } return *this; }

    void swap(Image& o) noexcept;
    void reset();

    VkImage image() const { return m_image; }
    VkImageView view() const { return m_view; }
    VkFormat format() const { return m_desc.format; }
    VkExtent3D extent() const { return m_desc.extent; }
    const ImageDesc& desc() const { return m_desc; }
    uint32_t mipLevels() const { return m_desc.mipLevels; }
    uint32_t arrayLayers() const { return m_desc.arrayLayers; }

private:
    Device* m_device = nullptr;
    VkImage m_image = VK_NULL_HANDLE;
    VkImageView m_view = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    ImageDesc m_desc{};
};

}
