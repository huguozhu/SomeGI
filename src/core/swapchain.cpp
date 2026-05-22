#include "swapchain.h"
#include "device.h"
#include "window.h"
#include <VkBootstrap.h>

namespace somegi {

Swapchain::Swapchain(Device& device, Window& window) : m_device(device), m_window(window) {
    m_frameSync.resize(kFramesInFlight);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
    for (auto& s : m_frameSync) {
        VK_CHECK(vkCreateSemaphore(m_device.device(), &sci, nullptr, &s.imageAvailable));
        VK_CHECK(vkCreateFence(m_device.device(), &fci, nullptr, &s.inFlight));
    }

    // Query HDR support via surface formats
    {
        uint32_t count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_device.physicalDevice(), m_device.surface(), &count, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_device.physicalDevice(), m_device.surface(), &count, formats.data());
        for (auto& fmt : formats) {
            if (fmt.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
                fmt.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) {
                m_hdrAvailable = true;
                std::printf("[swapchain] HDR scRGB (R16G16B16A16_SFLOAT) available\n");
                break;
            }
        }
        if (!m_hdrAvailable) std::printf("[swapchain] HDR not available, using SDR\n");
    }

    create();
}

Swapchain::~Swapchain() {
    destroy();
    for (auto& s : m_frameSync) {
        vkDestroySemaphore(m_device.device(), s.imageAvailable, nullptr);
        vkDestroyFence(m_device.device(), s.inFlight, nullptr);
    }
}

void Swapchain::create() {
    vkb::SwapchainBuilder sb{m_device.physicalDevice(), m_device.device(), m_device.surface()};

    VkSurfaceFormatKHR desired{};
    if (m_hdrEnabled && m_hdrAvailable) {
        desired = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT};
    } else {
        desired = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    }

    VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (m_hdrEnabled && m_hdrAvailable) {
        usage |= VK_IMAGE_USAGE_STORAGE_BIT; // scRGB output directly
    }

    auto ret = sb.set_desired_format(desired)
                .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
                .set_desired_extent(m_window.width(), m_window.height())
                .set_image_usage_flags(usage)
                .build();
    if (!ret) throw std::runtime_error("Swapchain: " + ret.error().message());
    auto vkbsc = ret.value();
    m_swapchain = vkbsc.swapchain;
    m_format = vkbsc.image_format;
    m_colorSpace = vkbsc.color_space;
    m_extent = vkbsc.extent;
    m_images = vkbsc.get_images().value();
    m_views = vkbsc.get_image_views().value();

    m_renderFinished.resize(m_images.size());
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (auto& s : m_renderFinished) {
        VK_CHECK(vkCreateSemaphore(m_device.device(), &sci, nullptr, &s));
    }
}

void Swapchain::destroy() {
    for (auto s : m_renderFinished) vkDestroySemaphore(m_device.device(), s, nullptr);
    m_renderFinished.clear();
    for (auto v : m_views) vkDestroyImageView(m_device.device(), v, nullptr);
    m_views.clear();
    if (m_swapchain) {
        vkDestroySwapchainKHR(m_device.device(), m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
}

void Swapchain::recreate() {
    m_device.waitIdle();
    destroy();
    create();
}

AcquiredFrame Swapchain::acquireNextFrame() {
    auto& s = m_frameSync[m_currentFrame];
    vkWaitForFences(m_device.device(), 1, &s.inFlight, VK_TRUE, UINT64_MAX);

    AcquiredFrame f{};
    f.frameInFlight = m_currentFrame;
    f.sync = &s;
    f.format = m_format;
    f.extent = m_extent;

    VkResult r = vkAcquireNextImageKHR(m_device.device(), m_swapchain, UINT64_MAX,
                                        s.imageAvailable, VK_NULL_HANDLE, &f.imageIndex);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        f.needsResize = true;
        return f;
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("vkAcquireNextImageKHR failed");
    }
    vkResetFences(m_device.device(), 1, &s.inFlight);
    f.image = m_images[f.imageIndex];
    f.view  = m_views[f.imageIndex];
    f.renderFinished = m_renderFinished[f.imageIndex];
    return f;
}

void Swapchain::present(const AcquiredFrame& frame) {
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &frame.renderFinished;
    pi.swapchainCount = 1;
    pi.pSwapchains = &m_swapchain;
    pi.pImageIndices = &frame.imageIndex;
    VkResult r = vkQueuePresentKHR(m_device.graphicsQueue(), &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || m_window.resized()) {
        recreate();
    } else if (r != VK_SUCCESS) {
        throw std::runtime_error("vkQueuePresentKHR failed");
    }
    m_currentFrame = (m_currentFrame + 1) % kFramesInFlight;
}

void Swapchain::setHdrEnabled(bool on) {
    if (on == m_hdrEnabled || (on && !m_hdrAvailable)) return;
    m_hdrEnabled = on;
    recreate();
}

}
