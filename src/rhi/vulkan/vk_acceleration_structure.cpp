// rhi/vulkan/vk_acceleration_structure.cpp
#include "vk_acceleration_structure.h"

namespace somegi {
namespace rhi {

VkRHIAccelerationStructure::~VkRHIAccelerationStructure() {
    if (m_as && m_owns) {
        // 拥有型销毁（预留，当前仅 createNonOwning 路径，不执行）
        m_as = VK_NULL_HANDLE;
    }
}

} // namespace rhi
} // namespace somegi
