#pragma once

#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>

#define VK_CHECK(expr)                                                     \
    do {                                                                   \
        VkResult _r = (expr);                                              \
        if (_r != VK_SUCCESS) {                                            \
            throw std::runtime_error(std::string(#expr " failed: ") +      \
                                     std::to_string(static_cast<int>(_r)));\
        }                                                                  \
    } while (0)

namespace somegi {

constexpr uint32_t kFramesInFlight = 2;

// Vulkan 图像布局转换辅助函数 —— 构建并提交 VkImageMemoryBarrier2。
// 多个 .cpp 文件中存在重复定义，现已统一到此公共位置。
inline void transitionImage(VkCommandBuffer cmd, VkImage image,
                            VkImageAspectFlags aspect,
                            VkImageLayout oldLayout, VkImageLayout newLayout,
                            VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                            VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask  = srcStage;  b.srcAccessMask = srcAccess;
    b.dstStageMask  = dstStage;  b.dstAccessMask = dstAccess;
    b.oldLayout = oldLayout;     b.newLayout = newLayout;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};

    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}

}
