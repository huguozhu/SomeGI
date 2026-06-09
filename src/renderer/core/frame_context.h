#pragma once
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

namespace somegi {

// 逐帧渲染上下文 —— 打包每帧从 swapchain 获取的动态数据。
// App::run() 在一帧开始时赋值，供渲染管线 lambda 和 post-processing 消费。
// 注意：与 gi_technique.h 中的 FrameContext (GI prepare 用) 是不同的结构体。
struct RenderFrame {
    uint32_t    frameInFlight = 0;
    VkImageView swapView      = VK_NULL_HANDLE;
    VkImage     swapImage     = VK_NULL_HANDLE;
    VkExtent2D  swapExtent{};
    glm::mat4   proj{1.0f};
    glm::mat4   view{1.0f};
    glm::mat4   invViewProj{1.0f};
};

} // namespace somegi
