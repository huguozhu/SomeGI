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
    // body builds ImGui windows (panels). Call before render().
    void render(VkCommandBuffer cmd, VkImageView swapchainView, VkExtent2D extent);

private:
    Device* m_device = nullptr;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    bool m_inited = false;
};

}
