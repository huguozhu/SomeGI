#pragma once
#include "core/vk_common.h"
#include <functional>

namespace somegi {
class Device;

void oneShotSubmit(Device& d, VkCommandPool pool, std::function<void(VkCommandBuffer)> body);

}
