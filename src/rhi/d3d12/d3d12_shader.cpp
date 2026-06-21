// rhi/d3d12/d3d12_shader.cpp — D3D12 Shader 实现
#include "d3d12_shader.h"
#include <cstring>

namespace somegi {
namespace rhi {

D3D12RHIShader::D3D12RHIShader(const ShaderDesc& desc, const void* bytecode, size_t size)
    : m_stage(desc.stage), m_entryPoint(desc.entryPoint ? desc.entryPoint : "main") {
    auto* data = static_cast<const uint8_t*>(bytecode);
    m_bytecode.assign(data, data + size);
}

D3D12RHIShader::~D3D12RHIShader() = default;

} // namespace rhi
} // namespace somegi
