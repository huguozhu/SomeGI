// rhi/vulkan/vk_shader.h
#pragma once
#include "../shader.h"
#include "vk_device.h"
#include <filesystem>

namespace somegi {
namespace rhi {

class VkRHIShader : public RHIShader {
public:
    static std::unique_ptr<RHIShader> create(VkRHIDevice& device, const ShaderDesc& desc, const void* bytecode, size_t size);
    static std::unique_ptr<RHIShader> createFromFile(VkRHIDevice& device, const ShaderDesc& desc, const std::filesystem::path& spvPath);
    ~VkRHIShader() override;
    ShaderStage stage() const override { return m_desc.stage; }
    void* nativeHandle() const override { return (void*)m_module; }
private:
    VkRHIDevice& m_device;
    VkShaderModule m_module = VK_NULL_HANDLE;
    ShaderDesc m_desc;
    VkRHIShader(VkRHIDevice& d, const ShaderDesc& desc) : m_device(d), m_desc(desc) {}
};

} // namespace rhi
} // namespace somegi
