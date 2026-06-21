// rhi/vulkan/vk_acceleration_structure.h — RHIAccelerationStructure 的 Vulkan 实现
#pragma once
#include "../base/acceleration_structure.h"
#include "vk_device.h"
#include <vulkan/vulkan.h>

namespace somegi {
namespace rhi {

class VkRHIAccelerationStructure : public RHIAccelerationStructure {
public:
    // 非拥有型包装（用于现有 VkAccelerationStructureKHR 的过渡，不管理生命周期）
    static std::unique_ptr<RHIAccelerationStructure> createNonOwning(VkRHIDevice& device, VkAccelerationStructureKHR as) {
        auto s = std::unique_ptr<VkRHIAccelerationStructure>(new VkRHIAccelerationStructure(device));
        s->m_owns = false;
        s->m_as = as;
        return s;
    }
    // 拥有型包装（m_tlasRHI 场景：SceneRtAS 创建 TLAS 后由 RHI wrapper 接管生命周期）
    static std::unique_ptr<RHIAccelerationStructure> createOwning(VkRHIDevice& device, VkAccelerationStructureKHR as) {
        auto s = std::unique_ptr<VkRHIAccelerationStructure>(new VkRHIAccelerationStructure(device));
        s->m_owns = true;
        s->m_as = as;
        return s;
    }
    ~VkRHIAccelerationStructure() override;
    void* nativeHandle() const override { return (void*)m_as; }
private:
    VkRHIDevice& m_device;
    VkAccelerationStructureKHR m_as = VK_NULL_HANDLE;
    bool m_owns = true;
    VkRHIAccelerationStructure(VkRHIDevice& d) : m_device(d) {}
};

} // namespace rhi
} // namespace somegi
