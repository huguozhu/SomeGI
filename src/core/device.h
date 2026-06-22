#pragma once
#include "vk_common.h"
#include "rhi/base/common.h"
#include <VkBootstrap.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <VulkanMemoryAllocator/vk_mem_alloc.h>

namespace somegi {
namespace rhi { class VkRHIDevice; }

class Window;

struct DeviceFeatures {
    bool rayTracing = false;
    bool meshShader = false;
    bool taskShader = false;
    bool accelStruct = false;
    bool rayQuery = false;
    uint32_t maxMeshOutputVertices = 256;
    uint32_t maxMeshOutputPrimitives = 256;
    uint32_t maxMeshWorkGroupSize = 128;
};

// ════════════════════════════════════════════════════════════════
// Device — RHI 过渡期薄包装器，所有方法委托给 rhi::VkRHIDevice
//
// 提供给仍使用 Device& 签名的 96 个 renderer/GI pass 文件。
// 不拥有任何 Vulkan 对象，析构为空操作。
// ════════════════════════════════════════════════════════════════
class Device {
public:
    Device(rhi::VkRHIDevice& vkDev);
    ~Device() = default;

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // ── 全部委托给 m_rhiDev，不直接调用 Vulkan API ──
    VkInstance instance() const;
    VkPhysicalDevice physicalDevice() const;
    VkDevice device() const;
    VkSurfaceKHR surface() const;
    VkQueue graphicsQueue() const;
    uint32_t graphicsQueueFamily() const;
    const DeviceFeatures& features() const { return m_features; }

    void waitIdle() const;

    float timestampPeriod() const { return m_timestampPeriod; }

    VkSampleCountFlags supportedSampleCounts() const { return m_supportedSampleCounts; }

    const vkb::DispatchTable& dispatch() const;

    PFN_vkCmdDrawMeshTasksEXT vkCmdDrawMeshTasksEXT = nullptr;

    VmaAllocator allocator() const;

    rhi::DeviceLimits limits() const { return m_limits; }

    // 直接访问底层 RHI 设备
    rhi::VkRHIDevice& rhiDev() const { return *m_rhiDev; }

private:
    rhi::VkRHIDevice* m_rhiDev;
    DeviceFeatures m_features{};
    float m_timestampPeriod = 1.0f;
    uint32_t m_supportedSampleCounts = 0;
    rhi::DeviceLimits m_limits{};
};

}
