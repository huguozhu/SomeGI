// rhi/vulkan/vk_swapchain.h
#pragma once
#include "../base/swapchain.h"
#include "vk_device.h"
#include "vk_texture.h"
#include "vk_fence.h"

namespace somegi {
namespace rhi {

class VkRHISwapchain : public RHISwapchain {
public:
    static std::unique_ptr<RHISwapchain> create(VkRHIDevice& device, void* nativeWindow, uint32_t width, uint32_t height);
    ~VkRHISwapchain() override;

    SwapchainFrame acquireNextFrame() override;
    void present(const SwapchainFrame& frame) override;
    void recreate() override;
    bool hdrAvailable() const override { return m_hdrAvailable; }
    bool hdrEnabled() const override { return m_hdrEnabled; }
    void setHdrEnabled(bool v) override { m_hdrEnabled = v; }
    Format format() const override { return m_format; }
    uint32_t width() const override { return m_extent.width; }
    uint32_t height() const override { return m_extent.height; }
    VkImage vkImage(uint32_t frameInFlight) const { return m_images[frameInFlight % m_images.size()]; }

private:
    VkRHIDevice& m_device;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkSurfaceFormatKHR m_surfaceFormat{};
    VkExtent2D m_extent{};
    Format m_format = Format::B8G8R8A8_UNORM;
    bool m_hdrAvailable = false;
    bool m_hdrEnabled = false;
    struct Sync { VkSemaphore imageAvailable, renderFinished; VkFence inFlight; };
    std::vector<Sync> m_syncs;
    std::vector<VkImage> m_images;
    std::vector<std::unique_ptr<VkRHITextureView>> m_views;
    uint32_t m_frameIndex = 0;

    VkRHISwapchain(VkRHIDevice& d) : m_device(d) {}
    void createSwapchain(uint32_t w, uint32_t h);
};

} // namespace rhi
} // namespace somegi
