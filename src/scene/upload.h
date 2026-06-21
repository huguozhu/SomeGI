#pragma once
#include "core/vk_common.h"
#include <functional>

namespace somegi {
class Device;

// 前向声明 RHI 类型（somegi::rhi 命名空间）
namespace rhi { class RHIDevice; class RHICommandBuffer; }

// oneShotSubmit: 一次性提交一个命令缓冲 → 等待空闲 → 释放
// 内部使用 RHI 管理命令池分配/提交/等待
void oneShotSubmit(rhi::RHIDevice& rhiDevice, std::function<void(VkCommandBuffer)> body);

// RHI 封装版：body 接收 RHI 命令缓冲，使用 RHI API 录制命令
void oneShotSubmitRHI(rhi::RHIDevice& rhiDevice,
                      std::function<void(rhi::RHICommandBuffer&)> body);

}
