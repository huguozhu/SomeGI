// rhi/vulkan/vk_shader.cpp
#include "vk_shader.h"
#include <core/vk_common.h>
#include <fstream>
#include <stdexcept>

namespace somegi {
namespace rhi {

std::unique_ptr<RHIShader> VkRHIShader::create(VkRHIDevice& device, const ShaderDesc& desc, const void* bytecode, size_t size) {
    auto s = std::unique_ptr<VkRHIShader>(new VkRHIShader(device, desc));
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = size;
    ci.pCode = (const uint32_t*)bytecode;
    VK_CHECK(vkCreateShaderModule(device.vkDevice(), &ci, nullptr, &s->m_module));
    return s;
}

std::unique_ptr<RHIShader> VkRHIShader::createFromFile(VkRHIDevice& device, const ShaderDesc& desc, const std::filesystem::path& spvPath) {
    std::ifstream file(spvPath, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open shader: " + spvPath.string());
    size_t size = (size_t)file.tellg();
    file.seekg(0);
    std::vector<uint32_t> code((size + 3) / 4);
    file.read((char*)code.data(), size);
    return create(device, desc, code.data(), size);
}

VkRHIShader::~VkRHIShader() { if (m_module) vkDestroyShaderModule(m_device.vkDevice(), m_module, nullptr); }

} // namespace rhi
} // namespace somegi
