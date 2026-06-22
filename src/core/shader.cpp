#include "shader.h"
#include "device.h"
#include "rhi/vulkan/vk_device.h"
#include <fstream>
#include <vector>

namespace somegi {

ShaderModule::ShaderModule(rhi::VkRHIDevice& vkDev, const std::filesystem::path& spvPath)
    : m_rhiDev(&vkDev) {
    std::ifstream f(spvPath, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("shader open failed: " + spvPath.string());
    auto size = static_cast<size_t>(f.tellg());
    if (size % 4) throw std::runtime_error("shader spv size not multiple of 4: " + spvPath.string());
    std::vector<uint32_t> data(size / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), size);

    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = size;
    ci.pCode = data.data();
    VK_CHECK(vkCreateShaderModule(m_rhiDev->vkDevice(), &ci, nullptr, &m_module));
}

ShaderModule::ShaderModule(Device& d, const std::filesystem::path& spvPath)
    : m_rhiDev(&d.rhiDev()) {
    std::ifstream f(spvPath, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("shader open failed: " + spvPath.string());
    auto size = static_cast<size_t>(f.tellg());
    if (size % 4) throw std::runtime_error("shader spv size not multiple of 4: " + spvPath.string());
    std::vector<uint32_t> data(size / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), size);

    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = size;
    ci.pCode = data.data();
    VK_CHECK(vkCreateShaderModule(m_rhiDev->vkDevice(), &ci, nullptr, &m_module));
}

ShaderModule::~ShaderModule() {
    if (m_rhiDev && m_module) vkDestroyShaderModule(m_rhiDev->vkDevice(), m_module, nullptr);
}

ShaderModule::ShaderModule(ShaderModule&& o) noexcept
    : m_rhiDev(o.m_rhiDev), m_module(o.m_module) {
    o.m_rhiDev = nullptr; o.m_module = VK_NULL_HANDLE;
}
ShaderModule& ShaderModule::operator=(ShaderModule&& o) noexcept {
    if (this != &o) {
        if (m_rhiDev && m_module) vkDestroyShaderModule(m_rhiDev->vkDevice(), m_module, nullptr);
        m_rhiDev = o.m_rhiDev; m_module = o.m_module;
        o.m_rhiDev = nullptr; o.m_module = VK_NULL_HANDLE;
    }
    return *this;
}

std::filesystem::path shaderDir() {
    return std::filesystem::path(SOMEGI_SHADER_DIR);
}

std::filesystem::path assetDir() {
    return std::filesystem::path(SOMEGI_ASSET_DIR);
}

}
