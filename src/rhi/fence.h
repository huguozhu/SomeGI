// rhi/fence.h
#pragma once
#include "common.h"

namespace somegi {
namespace rhi {

class RHIFence {
public:
    virtual ~RHIFence() = default;
    virtual void wait(uint64_t timeoutNs = UINT64_MAX) = 0;
    virtual void reset() = 0;
    virtual void* nativeHandle() const = 0;
};

class RHISemaphore {
public:
    virtual ~RHISemaphore() = default;
    virtual void* nativeHandle() const = 0;
};

} // namespace rhi
} // namespace somegi
