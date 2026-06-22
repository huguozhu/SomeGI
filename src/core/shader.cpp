#include "shader.h"
#include "device.h"
#include "rhi/vulkan/vk_device.h"
#include <fstream>
#include <vector>

namespace somegi {

ShaderModule::ShaderModule(rhi::VkRHIDevice& vkDev, const std::filesystem::path& spvPath) {
    std::ifstream f(spvPath, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("shader open failed: " + spvPath.string());
    auto size = static_cast<size_t>(f.tellg());
    if (size % 4) throw std::runtime_error("shader spv size not multiple of 4: " + spvPath.string());
    std::vector<uint32_t> data(size / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), size);

    rhi::ShaderDesc desc;
    desc.stage = rhi::ShaderStage::Fragment;  // 仅元数据，不影响 Vulkan 创建
    desc.format = rhi::ShaderFormat::SPIRV;
    desc.entryPoint = "main";
    m_rhiShader = vkDev.createShader(desc, data.data(), size);
    m_module = (VkShaderModule)(uintptr_t)m_rhiShader->nativeHandle();
}

ShaderModule::ShaderModule(Device& d, const std::filesystem::path& spvPath) {
    std::ifstream f(spvPath, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("shader open failed: " + spvPath.string());
    auto size = static_cast<size_t>(f.tellg());
    if (size % 4) throw std::runtime_error("shader spv size not multiple of 4: " + spvPath.string());
    std::vector<uint32_t> data(size / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), size);

    rhi::ShaderDesc desc;
    desc.stage = rhi::ShaderStage::Fragment;
    desc.format = rhi::ShaderFormat::SPIRV;
    desc.entryPoint = "main";
    m_rhiShader = d.rhiDev().createShader(desc, data.data(), size);
    m_module = (VkShaderModule)(uintptr_t)m_rhiShader->nativeHandle();
}

ShaderModule::~ShaderModule() = default;

ShaderModule::ShaderModule(ShaderModule&& o) noexcept
    : m_module(o.m_module), m_rhiShader(std::move(o.m_rhiShader)) {
    o.m_module = VK_NULL_HANDLE;
}
ShaderModule& ShaderModule::operator=(ShaderModule&& o) noexcept {
    if (this != &o) {
        m_module = o.m_module;
        m_rhiShader = std::move(o.m_rhiShader);
        o.m_module = VK_NULL_HANDLE;
    }
    return *this;
}

std::filesystem::path shaderDir() { return std::filesystem::path(SOMEGI_SHADER_DIR); }
std::filesystem::path assetDir() { return std::filesystem::path(SOMEGI_ASSET_DIR); }

}
