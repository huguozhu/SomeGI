#include "screenshot.h"
#include "device.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <filesystem>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cinttypes>

namespace somegi {

void ScreenshotCapture::init(Device& d, VkExtent2D extent) {
    destroy();
    m_extent = extent;

    // B8G8R8A8_UNORM = 4 字节/像素
    VkDeviceSize size = static_cast<VkDeviceSize>(extent.width) *
                        static_cast<VkDeviceSize>(extent.height) * 4;
    m_staging = Buffer(d, size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    m_inited = true;
    std::printf("[screenshot] staging buffer created (%u x %u, %" PRIu64 " bytes)\n",
                extent.width, extent.height, static_cast<uint64_t>(size));
}

void ScreenshotCapture::destroy() {
    m_staging.reset();
    m_inited = false;
    copied = false;
}

void ScreenshotCapture::recordCopy(VkCommandBuffer cmd, VkImage srcImage, VkExtent2D extent) {
    // 过渡布局：先确保 srcImage 在 TRANSFER_SRC layout
    // （调用方需在 cb 外先转好，或在本函数内转——这里要求调用方预先保证）
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;   // 0 = 紧密打包
    region.bufferImageHeight = 0; // 0 = 紧密打包
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {extent.width, extent.height, 1};

    vkCmdCopyImageToBuffer(cmd,
        srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        m_staging.handle(), 1, &region);

    copied = true;
}

void ScreenshotCapture::savePng(const std::string& filepath, VkExtent2D extent) const {
    if (!copied) {
        std::fprintf(stderr, "[screenshot] savePng called but no copy recorded\n");
        return;
    }
    if (!m_staging.mapped()) {
        std::fprintf(stderr, "[screenshot] staging buffer not mapped\n");
        return;
    }

    // 确保输出目录存在
    std::error_code ec;
    std::filesystem::create_directories(outputDir, ec);

    // BGRA → RGBA 转换（ldrTonemap 格式为 VK_FORMAT_B8G8R8A8_UNORM）
    const uint32_t pixelCount = extent.width * extent.height;
    std::vector<uint8_t> rgba(pixelCount * 4);
    const uint8_t* src = static_cast<const uint8_t*>(m_staging.mapped());

    for (uint32_t i = 0; i < pixelCount; ++i) {
        rgba[i * 4 + 0] = src[i * 4 + 2];  // R ← B
        rgba[i * 4 + 1] = src[i * 4 + 1];  // G ← G
        rgba[i * 4 + 2] = src[i * 4 + 0];  // B ← R
        rgba[i * 4 + 3] = src[i * 4 + 3];  // A ← A
    }

    std::string fullPath = outputDir + "/" + filepath;
    if (stbi_write_png(fullPath.c_str(),
                       static_cast<int>(extent.width),
                       static_cast<int>(extent.height),
                       4, rgba.data(),
                       static_cast<int>(extent.width) * 4)) {
        std::printf("[screenshot] saved %s (%ux%u)\n",
                    fullPath.c_str(), extent.width, extent.height);
    } else {
        std::fprintf(stderr, "[screenshot] stbi_write_png failed: %s\n",
                     fullPath.c_str());
    }
}

} // namespace somegi
