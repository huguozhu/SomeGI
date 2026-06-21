#pragma once
#include "core/vk_common.h"
#include <functional>

namespace somegi {
class Device;

// 前向声明 RHI 类型（somegi::rhi 命名空间）
namespace rhi { class RHIDevice; class RHICommandBuffer; }

void oneShotSubmit(Device& d, VkCommandPool pool, std::function<void(VkCommandBuffer)> body);

// RHI 封装版：内部创建 RHICommandBuffer 包装，body 接收 RHI 命令缓冲
void oneShotSubmitRHI(rhi::RHIDevice& rhiDevice, Device& d, VkCommandPool pool,
                      std::function<void(rhi::RHICommandBuffer&)> body);

}
