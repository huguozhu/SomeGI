// rhi/vulkan/vk_acceleration_structure.cpp
#include "vk_acceleration_structure.h"

namespace somegi {
namespace rhi {

VkRHIAccelerationStructure::~VkRHIAccelerationStructure() {
    if (m_as != VK_NULL_HANDLE && m_owns) {
        m_device.dispatch().destroyAccelerationStructureKHR(m_as, nullptr);
        m_as = VK_NULL_HANDLE;
    }
}

} // namespace rhi
} // namespace somegi
