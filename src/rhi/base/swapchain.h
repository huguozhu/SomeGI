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
    RHISemaphore* imageAvailable = nullptr;  // 由 swapchain 管理
    RHISemaphore* renderFinished = nullptr;
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
