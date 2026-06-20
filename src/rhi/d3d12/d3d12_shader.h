// rhi/d3d12/d3d12_shader.h — D3D12 Shader（DXIL bytecode 包装）
#pragma once
#include "../base/shader.h"
#include <vector>
#include <string>

namespace somegi {
namespace rhi {

class D3D12RHIShader : public RHIShader {
public:
    D3D12RHIShader(const ShaderDesc& desc, const void* bytecode, size_t size);
    ~D3D12RHIShader() override;
    ShaderStage stage() const override { return m_stage; }
    const char* entryPoint() const override { return m_entryPoint.c_str(); }
    void* nativeHandle() const override { return (void*)m_bytecode.data(); }
    const void* bytecodeData() const { return m_bytecode.data(); }
    size_t bytecodeSize() const { return m_bytecode.size(); }
private:
    ShaderStage m_stage;
    std::string m_entryPoint;
    std::vector<uint8_t> m_bytecode;
};

} // namespace rhi
} // namespace somegi
