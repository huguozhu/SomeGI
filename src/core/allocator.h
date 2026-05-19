#pragma once
#include "vk_common.h"

namespace somegi {

uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props);

}
