#pragma once
#include "vk_common.h"
#include <vector>

namespace somegi {

class Device;
class Window;

struct FrameSync {
    VkSemaphore imageAvailable = VK_NULL_HANDLE;
    VkFence     inFlight = VK_NULL_HANDLE;
};

struct AcquiredFrame {
    uint32_t imageIndex = 0;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    const FrameSync* sync = nullptr;            // for inFlight + imageAvailable
    VkSemaphore renderFinished = VK_NULL_HANDLE; // per-swapchain-image
    uint32_t frameInFlight = 0;
    bool needsResize = false;
};

class Swapchain {
public:
    Swapchain(Device& device, Window& window);
    ~Swapchain();

    AcquiredFrame acquireNextFrame();
    void present(const AcquiredFrame& frame);
    void recreate();

    VkFormat format() const { return m_format; }
    VkExtent2D extent() const { return m_extent; }
    bool hdrAvailable() const { return m_hdrAvailable; }
    bool hdrEnabled() const { return m_hdrEnabled; }
    void setHdrEnabled(bool on);

private:
    void create();
    void destroy();

    Device& m_device;
    Window& m_window;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_format = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR m_colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    bool m_hdrAvailable = false;
    bool m_hdrEnabled = false;
    VkExtent2D m_extent{};
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_views;
    std::vector<VkSemaphore> m_renderFinished;  // per swapchain image
    std::vector<FrameSync> m_frameSync;
    uint32_t m_currentFrame = 0;
};

}
