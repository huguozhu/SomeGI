#include "upload.h"
#include "core/device.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_command.h"

namespace somegi {

void oneShotSubmit(Device& d, VkCommandPool pool, std::function<void(VkCommandBuffer)> body) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(d.device(), &ai, &cmd));

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    body(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = cmd;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1; si.pCommandBufferInfos = &csi;

    VK_CHECK(vkQueueSubmit2(d.graphicsQueue(), 1, &si, VK_NULL_HANDLE));
    vkQueueWaitIdle(d.graphicsQueue());
    vkFreeCommandBuffers(d.device(), pool, 1, &cmd);
}

void oneShotSubmitRHI(rhi::RHIDevice& rhiDevice, Device& d, VkCommandPool pool,
                      std::function<void(rhi::RHICommandBuffer&)> body) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiDevice);
    oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
        rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);
        body(rhiCmd);
    });
}

}
