// rhi/shader.h
#pragma once
#include "common.h"

namespace somegi {
namespace rhi {

class RHIShader {
public:
    virtual ~RHIShader() = default;
    virtual ShaderStage stage() const = 0;
    virtual void* nativeHandle() const = 0;
};

} // namespace rhi
} // namespace somegi
