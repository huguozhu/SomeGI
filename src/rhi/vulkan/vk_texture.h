// rhi/vulkan/vk_texture.h
#pragma once
#include "../texture.h"
#include "vk_device.h"

namespace somegi {
namespace rhi {

class VkRHITexture : public RHITexture {
public:
    static std::unique_ptr<RHITexture> create(VkRHIDevice& device, const TextureDesc& desc);
    ~VkRHITexture() override;
    std::unique_ptr<RHITextureView> createView(const TextureViewDesc& desc) override;
    Format format() const override { return m_desc.format; }
    uint32_t width() const override { return m_desc.width; }
    uint32_t height() const override { return m_desc.height; }
    uint32_t mipLevels() const override { return m_desc.mipLevels; }
    void* nativeHandle() const override { return (void*)m_image; }
private:
    VkRHIDevice& m_device;
    VkImage m_image = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    TextureDesc m_desc{};
    VkRHITexture(VkRHIDevice& d, const TextureDesc& desc);
};

class VkRHITextureView : public RHITextureView {
public:
    VkRHITextureView(VkRHIDevice& d) : m_device(d) {}
    static std::unique_ptr<RHITextureView> create(VkRHIDevice& device, const RHITexture& tex, const TextureViewDesc& desc);
    ~VkRHITextureView() override;
    void* nativeHandle() const override { return (void*)m_view; }
    void setView(VkImageView v) { m_view = v; }
    VkImageView vkView() const { return m_view; }
private:
    VkRHIDevice& m_device;
    VkImageView m_view = VK_NULL_HANDLE;
};

} // namespace rhi
} // namespace somegi
