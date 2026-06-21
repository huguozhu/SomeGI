#include "upload.h"
#include "core/device.h"
#include "rhi/base/device.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_command.h"

namespace somegi {

void oneShotSubmit(rhi::RHIDevice& rhiDevice, std::function<void(VkCommandBuffer)> body) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiDevice);
    auto pool = rhiDevice.createCommandPool();
    auto* rhiCmd = pool->allocateRaw();
    rhiCmd->begin();
    VkCommandBuffer vkCmd = static_cast<VkCommandBuffer>(rhiCmd->nativeHandle());
    body(vkCmd);
    rhiCmd->end();
    rhiDevice.submit({rhiCmd});
    rhiDevice.waitIdle();
}

void oneShotSubmitRHI(rhi::RHIDevice& rhiDevice,
                      std::function<void(rhi::RHICommandBuffer&)> body) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiDevice);
    oneShotSubmit(rhiDevice, [&](VkCommandBuffer vkCmd) {
        rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
        body(rhiCmd);
    });
}

}
