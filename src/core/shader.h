#pragma once
#include "vk_common.h"
#include <filesystem>

namespace somegi {

class Device;

class ShaderModule {
public:
    ShaderModule() = default;
    ShaderModule(Device& d, const std::filesystem::path& spvPath);
    ~ShaderModule();

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ShaderModule(ShaderModule&&) noexcept;
    ShaderModule& operator=(ShaderModule&&) noexcept;

    VkShaderModule handle() const { return m_module; }

private:
    Device* m_device = nullptr;
    VkShaderModule m_module = VK_NULL_HANDLE;
};

std::filesystem::path shaderDir();
std::filesystem::path assetDir();

}
