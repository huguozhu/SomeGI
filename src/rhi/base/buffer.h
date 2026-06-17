// rhi/buffer.h
#pragma once
#include "common.h"

namespace somegi {
namespace rhi {

class RHIBuffer {
public:
    virtual ~RHIBuffer() = default;
    virtual void* map() = 0;
    virtual void unmap() = 0;
    virtual uint64_t size() const = 0;
    virtual uint64_t deviceAddress() const = 0;
    virtual void* nativeHandle() const = 0;
};

} // namespace rhi
} // namespace somegi
