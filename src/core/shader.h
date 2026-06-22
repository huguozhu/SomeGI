#pragma once
#include "vk_common.h"
#include "path_util.h"
#include <filesystem>

namespace somegi {

class Device;

namespace rhi { class VkRHIDevice; }

class ShaderModule {
public:
    ShaderModule() = default;
    ShaderModule(Device& d, const std::filesystem::path& spvPath);
    ShaderModule(rhi::VkRHIDevice& vkDev, const std::filesystem::path& spvPath);
    ~ShaderModule();

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ShaderModule(ShaderModule&&) noexcept;
    ShaderModule& operator=(ShaderModule&&) noexcept;

    VkShaderModule handle() const { return m_module; }

private:
    rhi::VkRHIDevice* m_rhiDev = nullptr;  // RHI 设备指针（不拥有）
    VkShaderModule m_module = VK_NULL_HANDLE;
};

std::filesystem::path shaderDir();
std::filesystem::path assetDir();

}
