// rhi/texture.h
#pragma once
#include "common.h"
#include <memory>

namespace somegi {
namespace rhi {

class RHITextureView {
public:
    virtual ~RHITextureView() = default;
    virtual void* nativeHandle() const = 0;
};

class RHITexture {
public:
    virtual ~RHITexture() = default;
    virtual std::unique_ptr<RHITextureView> createView(const TextureViewDesc& desc) = 0;
    virtual Format format() const = 0;
    virtual uint32_t width() const = 0;
    virtual uint32_t height() const = 0;
    virtual uint32_t mipLevels() const = 0;
    virtual void* nativeHandle() const = 0;
};

} // namespace rhi
} // namespace somegi
