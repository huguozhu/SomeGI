// rhi/shader.h
#pragma once
#include "common.h"

namespace somegi {
namespace rhi {

class RHIShader {
public:
    virtual ~RHIShader() = default;
    virtual ShaderStage stage() const = 0;
    virtual const char* entryPoint() const = 0;  // 用于 PSO 创建时指定入口函数名
    virtual void* nativeHandle() const = 0;
};

} // namespace rhi
} // namespace somegi
