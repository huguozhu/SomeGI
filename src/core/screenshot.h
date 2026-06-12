#pragma once
#include "vk_common.h"
#include "buffer.h"
#include <string>

namespace somegi {

class Device;

// 截屏工具：从 GPU image 拷贝到 host-visible staging buffer，保存为 PNG 文件。
//
// 用法：
//   1. App 构造后（swapchain 创建完毕）调用 init()
//   2. 每帧 run() 末尾、submit fence 完成后：
//        if (m_screenshot.shouldCapture()) {
//            vkWaitForFences(...);
//            oneShotSubmit(..., [&](cmd) { m_screenshot.recordCopy(cmd, ...); });
//            m_screenshot.savePng(...);
//        }
//   3. swapchain resize 时调用一次 init() 重建 staging buffer
class ScreenshotCapture {
public:
    void init(Device& d, VkExtent2D extent);
    void destroy();

    // 在 command buffer 中录制 vkCmdCopyImageToBuffer（srcImage → staging）
    // 要求 srcImage 必须已在 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    void recordCopy(VkCommandBuffer cmd, VkImage srcImage, VkExtent2D extent);

    // 将 staging buffer 中的像素数据写为 PNG 文件。
    // filepath 是相对于 outputDir 的文件名（例如 "frame_0001.png"）
    void savePng(const std::string& filepath, VkExtent2D extent) const;

    // ---- 配置 ----
    int  captureInterval = 0;      // 0=禁用；N>0 时每 N 帧自动截图
    bool manualRequest = false;    // F12 手动触发（下一帧截图后自动清零）
    int  captureOneFrame = -1;     // --capture-frame N：仅截第 N 帧（截图后自动置 -1）
    std::string outputPrefix = "frame";   // 文件名前缀
    std::string outputDir = "screenshots"; // 输出目录（自动创建）

    // ---- 运行时状态 ----
    int  frameCount = 0;           // 帧计数器（每帧递增）
    bool copied = false;           // 当前帧已录制 copy 命令

    // 判断本帧是否需要截图（手动 / 间隔 / 指定帧）
    bool shouldCapture() const {
        if (manualRequest) return true;
        if (captureOneFrame >= 0 && frameCount == captureOneFrame) return true;
        if (captureInterval > 0 && frameCount > 0 && (frameCount % captureInterval) == 0)
            return true;
        return false;
    }

private:
    Buffer m_staging;
    bool m_inited = false;
    VkExtent2D m_extent{};
};

} // namespace somegi
