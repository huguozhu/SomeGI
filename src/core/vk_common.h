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
}
