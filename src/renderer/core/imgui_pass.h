#pragma once
#include "core/vk_common.h"
#include "rhi/base/device.h"
#include <functional>

struct GLFWwindow;

namespace somegi {
class Device;
namespace rhi { class RHICommandBuffer; }

class ImGuiPass {
public:
    void init(Device& d, rhi::RHIDevice& rhiDevice, GLFWwindow* window, VkFormat swapchainFormat, uint32_t imageCount);
    void destroy();

    void newFrame();

    // 持久化：窗口尺寸变更时保存 ini + 控件样式
    void saveSettings();

    // RHI 路径：rendering pass 通过 RHI 接口，ImGui draw 仍用原生 VkCommandBuffer
    void render(rhi::RHICommandBuffer& cmd, VkImageView swapchainView, VkExtent2D extent);
    void render(VkCommandBuffer cmd, VkImageView swapchainView, VkExtent2D extent);

private:
    // 从 assets/imgui_style.ini 恢复控件尺寸样式
    void loadStyle();
    rhi::RHIDevice* m_rhiDevice = nullptr;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    bool m_inited = false;
};

}
