// core/debug_dump.cpp
#include "debug_dump.h"
#include "device.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_command.h"

#include <stb/stb_image_write.h>

#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace somegi {

void DebugDump::init(Device& d, uint32_t maxWidth, uint32_t maxHeight) {
    destroy();
    VkDeviceSize size = static_cast<VkDeviceSize>(maxWidth) *
                        static_cast<VkDeviceSize>(maxHeight) * 16;  // 最大 16 字节/像素
    m_staging = Buffer(d, size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void DebugDump::destroy() {
    m_staging.reset();
    m_copied = false;
}

void DebugDump::recordCopy(rhi::RHICommandBuffer& rhiCmd, VkImage srcImage, VkFormat format,
                            uint32_t width, uint32_t height) {
    auto& vkCmd = static_cast<rhi::VkRHICommandBuffer&>(rhiCmd);
    auto& vkDev = vkCmd.device();

    auto srcTex = rhi::VkRHITexture::createNonOwning(vkDev, srcImage,
        rhi::toRhiFormat(format), width, height, 1);
    auto dstBuf = rhi::VkRHIBuffer::createNonOwning(vkDev, m_staging.handle(), m_staging.size());

    rhi::BufferTextureCopyRegion r;
    r.extentWidth = width;
    r.extentHeight = height;
    rhiCmd.copyTextureToBuffer(*srcTex, *dstBuf, r);

    m_imgExtent = {width, height};
    m_imgFormat = format;
    m_copied = true;
}

static uint32_t bytesPerPixel(VkFormat fmt) {
    switch (fmt) {
        case VK_FORMAT_R8G8B8A8_UNORM:  return 4;
        case VK_FORMAT_B8G8R8A8_UNORM:  return 4;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
        case VK_FORMAT_R32_SFLOAT:      return 4;
        case VK_FORMAT_D32_SFLOAT:      return 4;
        case VK_FORMAT_R16G16_SFLOAT:   return 4;
        default: return 4;
    }
}

void DebugDump::savePng(const std::string& filepath, VkFormat format,
                         uint32_t width, uint32_t height) {
    if (!m_copied) { std::fprintf(stderr, "[debug_dump] no copy recorded\n"); return; }
    if (!m_staging.mapped()) { std::fprintf(stderr, "[debug_dump] buffer not mapped\n"); return; }

    uint32_t bpp = bytesPerPixel(format);
    uint32_t pixelCount = width * height;
    std::vector<uint8_t> rgba(pixelCount * 4, 0);
    const uint8_t* src = static_cast<const uint8_t*>(m_staging.mapped());

    for (uint32_t i = 0; i < pixelCount; ++i) {
        uint8_t* d = &rgba[i * 4];
        const uint8_t* s = src + i * bpp;

        switch (format) {
            case VK_FORMAT_R8G8B8A8_UNORM:
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
                break;
            case VK_FORMAT_B8G8R8A8_UNORM:
                d[0] = s[2]; d[1] = s[1]; d[2] = s[0]; d[3] = 255;
                break;
            case VK_FORMAT_R16G16B16A16_SFLOAT: {
                // 半精度 float → uint8
                auto f16to32 = [](uint16_t h) -> float {
                    // 简单半精度解码
                    uint32_t sign = (h >> 15) & 1;
                    uint32_t exp  = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;
                    if (exp == 0) return (sign ? -1.f : 1.f) * std::ldexp((float)mant, -24);
                    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
                    float val = std::ldexp(1.0f + (float)mant / 1024.0f, (int)exp - 15);
                    return sign ? -val : val;
                };
                uint16_t r16, g16, b16;
                std::memcpy(&r16, s+0, 2); std::memcpy(&g16, s+2, 2); std::memcpy(&b16, s+4, 2);
                auto clamp = [](float v) { return (uint8_t)std::clamp((int)(v * 255.0f), 0, 255); };
                d[0] = clamp(f16to32(r16));
                d[1] = clamp(f16to32(g16));
                d[2] = clamp(f16to32(b16));
                d[3] = 255;
                break;
            }
            case VK_FORMAT_D32_SFLOAT: {
                float depth;
                std::memcpy(&depth, s, 4);
                // 深度可视化：近处亮，远处暗，clamp 到可见范围
                uint8_t v = (uint8_t)std::clamp((int)((1.0f - depth) * 255.0f), 0, 255);
                d[0] = v; d[1] = v; d[2] = v; d[3] = 255;
                break;
            }
            case VK_FORMAT_R16G16_SFLOAT: {
                auto f16to32 = [](uint16_t h) -> float {
                    uint32_t sign = (h >> 15) & 1; uint32_t exp = (h >> 10) & 0x1F; uint32_t mant = h & 0x3FF;
                    if (exp == 0) return (sign ? -1.f : 1.f) * std::ldexp((float)mant, -24);
                    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
                    float val = std::ldexp(1.0f + (float)mant / 1024.0f, (int)exp - 15);
                    return sign ? -val : val;
                };
                uint16_t r16, g16;
                std::memcpy(&r16, s+0, 2); std::memcpy(&g16, s+2, 2);
                auto clamp = [](float v) { return (uint8_t)std::clamp((int)(v * 255.0f), 0, 255); };
                d[0] = clamp(f16to32(r16)); d[1] = clamp(f16to32(g16)); d[2] = 0; d[3] = 255;
                break;
            }
            default:
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 255;
                break;
        }
    }

    // 确保输出目录存在
    std::error_code ec;
    std::filesystem::create_directories("debug_dump", ec);

    if (stbi_write_png(filepath.c_str(), (int)width, (int)height, 4, rgba.data(), (int)width * 4)) {
        std::printf("[debug_dump] saved %s (%ux%u)\n", filepath.c_str(), width, height);
    } else {
        std::fprintf(stderr, "[debug_dump] stbi_write_png failed: %s\n", filepath.c_str());
    }
}

} // namespace somegi
