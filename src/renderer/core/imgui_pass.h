#pragma once
#include "core/vk_common.h"
#include <functional>

struct GLFWwindow;

namespace somegi {
class Device;

class ImGuiPass {
public:
    void init(Device& d, GLFWwindow* window, VkFormat swapchainFormat, uint32_t imageCount);
    void destroy();

    void newFrame();

    // 持久化：窗口尺寸变更时保存 ini + 控件样式
    void saveSettings();

    // body builds ImGui windows (panels). Call before render().
    void render(VkCommandBuffer cmd, VkImageView swapchainView, VkExtent2D extent);

private:
    // 从 assets/imgui_style.ini 恢复控件尺寸样式
    void loadStyle();
    Device* m_device = nullptr;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    bool m_inited = false;
};

}
