// core/debug_dump.h — 开发调试：将 render target 纹理保存为 PNG 文件。
// 仅用于诊断渲染问题，发布版本应移除。
#pragma once
#include "vk_common.h"
#include "buffer.h"
#include <string>
#include <cstdio>

namespace somegi {
class Device;

class DebugDump {
public:
    // 初始化 staging buffer（根据最大纹理尺寸分配）
    void init(Device& d, uint32_t maxWidth, uint32_t maxHeight);
    void destroy();

    // 在 command buffer 中录制 vkCmdCopyImageToBuffer。
    // 调用方需确保 srcImage 已在 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    void recordCopy(VkCommandBuffer cmd, VkImage srcImage, VkFormat format,
                    uint32_t width, uint32_t height);

    // 保存最近一次复制的纹理为 PNG（调用前需确保 cmd 已执行完毕）
    void savePng(const std::string& filepath, VkFormat format,
                 uint32_t width, uint32_t height);

    bool hasCopy() const { return m_copied; }

private:
    Buffer m_staging;
    bool m_copied = false;
    VkExtent2D m_imgExtent{};
    VkFormat m_imgFormat = VK_FORMAT_UNDEFINED;
};

} // namespace somegi
