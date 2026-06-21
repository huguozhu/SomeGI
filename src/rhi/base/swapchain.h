// rhi/swapchain.h
#pragma once
#include "common.h"
#include <memory>

namespace somegi {
namespace rhi {

class RHISemaphore;
class RHITexture;
class RHITextureView;

struct SwapchainFrame {
    uint32_t frameInFlight;
    std::unique_ptr<RHITexture> texture;
    std::unique_ptr<RHITextureView> view;
    void* imageAvailable = nullptr;   // VkSemaphore：submit 时 wait
    void* renderFinished = nullptr;   // VkSemaphore：submit 时 signal
    void* inFlightFence = nullptr;    // VkFence：GPU 完成后 CPU 同步
    bool needsResize = false;
    uint32_t width = 0, height = 0;
};

class RHISwapchain {
public:
    virtual ~RHISwapchain() = default;
    virtual SwapchainFrame acquireNextFrame() = 0;
    virtual void present(const SwapchainFrame& frame) = 0;
    virtual void recreate() = 0;
    virtual bool hdrAvailable() const = 0;
    virtual bool hdrEnabled() const = 0;
    virtual void setHdrEnabled(bool) = 0;
    virtual Format format() const = 0;
    virtual uint32_t width() const = 0;
    virtual uint32_t height() const = 0;
};

} // namespace rhi
} // namespace somegi
