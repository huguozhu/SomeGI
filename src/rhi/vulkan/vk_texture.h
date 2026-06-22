// rhi/vulkan/vk_texture.h
#pragma once
#include "../base/texture.h"
#include "vk_device.h"

namespace somegi {
namespace rhi {

// VkFormat → RHI Format 反向映射（用于非拥有型纹理包装）
Format toRhiFormat(VkFormat vkFmt);

// 根据 RHI Format 推断 Vulkan image aspect（用于 barrier / view creation）
inline VkImageAspectFlags toVkAspect(Format f) {
    switch (f) {
        case Format::D32_SFLOAT:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        // 未来：D24_UNORM_S8_UINT → VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
        default:
            return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

class VkRHITexture : public RHITexture {
public:
    static std::unique_ptr<RHITexture> create(VkRHIDevice& device, const TextureDesc& desc);
    // 非拥有型包装：不销毁 VkImage，用于临时包装已有 Vulkan 纹理
    static std::unique_ptr<RHITexture> createNonOwning(VkRHIDevice& device, VkImage image,
                                                        Format format, uint32_t width, uint32_t height,
                                                        uint32_t mipLevels = 1);
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
    bool m_ownsImage = true;
    VkRHITexture(VkRHIDevice& d, const TextureDesc& desc);
    VkRHITexture(VkRHIDevice& d, VkImage image, Format format,
                 uint32_t width, uint32_t height, uint32_t mipLevels);
};

class VkRHITextureView : public RHITextureView {
public:
    VkRHITextureView(VkRHIDevice& d) : m_vkDev(d.vkDevice()) {}
    static std::unique_ptr<RHITextureView> create(VkRHIDevice& device, const RHITexture& tex, const TextureViewDesc& desc);
    // 非拥有型包装：析构时不销毁 VkImageView（用于 FGResources 资源池包装）
    static std::unique_ptr<RHITextureView> createNonOwning(VkRHIDevice& device, VkImageView view) {
        auto v = std::unique_ptr<VkRHITextureView>(new VkRHITextureView(device));
        v->m_ownsView = false;
        v->m_view = view;
        return v;
    }
    // 从原生 Vulkan 句柄创建并拥有 VkImageView（适配仍用原生 VkImage 的旧代码路径）
    // 内部调用 vkCreateImageView，析构时自动 vkDestroyImageView
    static std::unique_ptr<RHITextureView> createNonOwning(VkDevice device, const VkImageViewCreateInfo& ci);
    // 便捷重载：接受 VkRHIDevice& 替代裸 VkDevice
    static std::unique_ptr<RHITextureView> createNonOwning(VkRHIDevice& device, const VkImageViewCreateInfo& ci) {
        return createNonOwning(device.vkDevice(), ci);
    }
    ~VkRHITextureView() override;
    void* nativeHandle() const override { return (void*)m_view; }
    void setView(VkImageView v) { m_view = v; }
    VkImageView vkView() const { return m_view; }
private:
    VkRHITextureView(VkDevice d) : m_vkDev(d) {}
    VkDevice m_vkDev = VK_NULL_HANDLE;
    VkImageView m_view = VK_NULL_HANDLE;
    bool m_ownsView = true;
};

} // namespace rhi
} // namespace somegi
