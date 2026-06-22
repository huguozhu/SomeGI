#pragma once
#include "vk_common.h"
#include "rhi/base/shader.h"
#include "path_util.h"
#include <filesystem>
#include <memory>

namespace somegi {

namespace rhi { class VkRHIDevice; }
class Device;

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
    VkShaderModule m_module = VK_NULL_HANDLE;  // 缓存原生句柄（兼容旧调用方）
    std::unique_ptr<rhi::RHIShader> m_rhiShader;  // RHI 拥有实际资源
};

std::filesystem::path shaderDir();
std::filesystem::path assetDir();

}
