#pragma once
#include "vk_common.h"
#include "rhi/base/device.h"
#include "rhi/base/texture.h"
#include <memory>

namespace somegi {
namespace rhi { class VkRHIDevice; }
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
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
};

class Image {
public:
    Image() = default;
    Image(Device& d, const ImageDesc& desc);
    Image(rhi::VkRHIDevice& vkDev, const ImageDesc& desc);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& o) noexcept { swap(o); }
    Image& operator=(Image&& o) noexcept { if (this != &o) { reset(); swap(o); } return *this; }

    void swap(Image& o) noexcept;
    void reset();

    VkImage image() const;
    VkImageView view() const;
    VkFormat format() const { return m_desc.format; }
    VkExtent3D extent() const { return m_desc.extent; }
    const ImageDesc& desc() const { return m_desc; }
    uint32_t mipLevels() const { return m_desc.mipLevels; }
    uint32_t arrayLayers() const { return m_desc.arrayLayers; }

private:
    void init(rhi::RHIDevice& device, const ImageDesc& desc);

    std::unique_ptr<rhi::RHITexture> m_rhiTexture;     // RHI 拥有实际 VkImage
    std::unique_ptr<rhi::RHITextureView> m_rhiView;     // RHI 拥有实际 VkImageView
    ImageDesc m_desc{};
};

}
