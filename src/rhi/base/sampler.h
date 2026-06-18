// rhi/base/sampler.h — RHISampler 抽象
#pragma once
#include "common.h"

namespace somegi {
namespace rhi {

class RHISampler {
public:
    virtual ~RHISampler() = default;
    virtual void* nativeHandle() const = 0;
};

} // namespace rhi
} // namespace somegi
