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

    // Void* 访问器（便于逐步迁移到 RHI 通用类型）
    void* nativeImage() const             { return (void*)image; }
    void* nativeView() const              { return (void*)view; }
    void* nativeImageAvailable() const    { return sync ? (void*)sync->imageAvailable : nullptr; }
    void* nativeRenderFinished() const    { return (void*)renderFinished; }
    void* nativeInFlightFence() const     { return sync ? (void*)sync->inFlight : nullptr; }
    VkExtent2D nativeExtent() const       { return extent; }
};

namespace rhi { class VkRHIDevice; }

class Swapchain {
public:
    Swapchain(Device& device, Window& window);
    Swapchain(rhi::VkRHIDevice& vkDev, Window& window);
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
    void init(VkDevice device, VkPhysicalDevice physDev, VkInstance instance, VkQueue gq);
    void create();
    void destroy();

    Window& m_window;
    VkDevice m_vkDevice = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
    VkInstance m_vkInstance = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_format = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR m_colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    bool m_hdrAvailable = false;
    bool m_hdrEnabled = false;
    VkExtent2D m_extent{};
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_views;
    std::vector<VkSemaphore> m_renderFinished;
    std::vector<FrameSync> m_frameSync;
    uint32_t m_currentFrame = 0;
};

}
