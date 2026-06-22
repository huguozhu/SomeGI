// rhi/vulkan/vk_swapchain.cpp
#include "vk_swapchain.h"
#include <core/window.h>
#include <stdexcept>

namespace somegi {
namespace rhi {

std::unique_ptr<RHISwapchain> VkRHISwapchain::create(VkRHIDevice& device, void* nativeWindow, uint32_t width, uint32_t height) {
    auto sc = std::unique_ptr<VkRHISwapchain>(new VkRHISwapchain(device));
    sc->createSwapchain(width, height);
    return sc;
}

void VkRHISwapchain::createSwapchain(uint32_t w, uint32_t h) {
    // surface 已在 VkRHIDevice 初始化时创建，此处直接使用
    VkSurfaceKHR surface = m_device.surface();

    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_device.vkPhysicalDevice(), surface, &caps);
    m_extent = {w, h};

    // 选择格式
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_device.vkPhysicalDevice(), surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_device.vkPhysicalDevice(), surface, &formatCount, formats.data());
    m_surfaceFormat = formats[0];
    for (auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            m_surfaceFormat = f; break;
        }
    }
    m_format = (m_surfaceFormat.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) ? Format::R16G16B16A16_SFLOAT : Format::B8G8R8A8_UNORM;

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface;
    ci.minImageCount = 2;
    ci.imageFormat = m_surfaceFormat.format;
    ci.imageColorSpace = m_surfaceFormat.colorSpace;
    ci.imageExtent = m_extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    ci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(m_device.vkDevice(), &ci, nullptr, &m_swapchain));

    uint32_t imgCount;
    vkGetSwapchainImagesKHR(m_device.vkDevice(), m_swapchain, &imgCount, nullptr);
    m_images.resize(imgCount);
    vkGetSwapchainImagesKHR(m_device.vkDevice(), m_swapchain, &imgCount, m_images.data());

    // 创建 image views
    m_views.resize(imgCount);
    for (uint32_t i = 0; i < imgCount; ++i) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = m_images[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = m_surfaceFormat.format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view;
        vkCreateImageView(m_device.vkDevice(), &vci, nullptr, &view);
        m_views[i] = std::unique_ptr<VkRHITextureView>(new VkRHITextureView(m_device));
        m_views[i]->setView(view);
    }

    // 创建同步对象
    m_syncs.resize(imgCount);
    for (auto& s : m_syncs) {
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(m_device.vkDevice(), &sci, nullptr, &s.imageAvailable);
        vkCreateSemaphore(m_device.vkDevice(), &sci, nullptr, &s.renderFinished);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
        vkCreateFence(m_device.vkDevice(), &fci, nullptr, &s.inFlight);
    }
}

VkRHISwapchain::~VkRHISwapchain() {
    for (auto& s : m_syncs) {
        vkDestroySemaphore(m_device.vkDevice(), s.imageAvailable, nullptr);
        vkDestroySemaphore(m_device.vkDevice(), s.renderFinished, nullptr);
        vkDestroyFence(m_device.vkDevice(), s.inFlight, nullptr);
    }
    m_views.clear();
    if (m_swapchain) vkDestroySwapchainKHR(m_device.vkDevice(), m_swapchain, nullptr);
}

SwapchainFrame VkRHISwapchain::acquireNextFrame() {
    SwapchainFrame f;
    f.frameInFlight = m_frameIndex % (uint32_t)m_syncs.size();
    auto& sync = m_syncs[f.frameInFlight];
    vkWaitForFences(m_device.vkDevice(), 1, &sync.inFlight, VK_TRUE, UINT64_MAX);
    vkResetFences(m_device.vkDevice(), 1, &sync.inFlight);

    uint32_t imgIdx;
    VkResult r = vkAcquireNextImageKHR(m_device.vkDevice(), m_swapchain, UINT64_MAX, sync.imageAvailable, VK_NULL_HANDLE, &imgIdx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) { f.needsResize = true; return f; }
    VK_CHECK(r);

    f.imageIndex = imgIdx;
    f.texture = nullptr; // swapchain images are not RHITexture wrappers
    // 使用非拥有型包装，避免 SwapchainFrame 析构时误销毁 swapchain 内部的 VkImageView
    f.view = VkRHITextureView::createNonOwning(m_device, m_views[imgIdx]->vkView());
    f.width = m_extent.width; f.height = m_extent.height;
    f.imageAvailable = VkRHISemaphore::createNonOwning(m_device, sync.imageAvailable);
    f.renderFinished = VkRHISemaphore::createNonOwning(m_device, sync.renderFinished);
    f.inFlightFence = &sync.inFlight;
    m_frameIndex++;
    return f;
}

void VkRHISwapchain::present(const SwapchainFrame& frame) {
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    VkSemaphore sig = m_syncs[frame.frameInFlight].renderFinished;
    pi.pWaitSemaphores = &sig;
    pi.swapchainCount = 1;
    pi.pSwapchains = &m_swapchain;
    pi.pImageIndices = &frame.imageIndex;
    vkQueuePresentKHR(m_device.vkQueue(), &pi);
}

void VkRHISwapchain::recreate() {
    vkDeviceWaitIdle(m_device.vkDevice());
    for (auto& s : m_syncs) {
        vkDestroySemaphore(m_device.vkDevice(), s.imageAvailable, nullptr);
        vkDestroySemaphore(m_device.vkDevice(), s.renderFinished, nullptr);
        vkDestroyFence(m_device.vkDevice(), s.inFlight, nullptr);
    }
    m_views.clear();
    vkDestroySwapchainKHR(m_device.vkDevice(), m_swapchain, nullptr);
    createSwapchain(m_extent.width, m_extent.height);
}

} // namespace rhi
} // namespace somegi
