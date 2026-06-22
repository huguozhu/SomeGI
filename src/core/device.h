#pragma once
#include "vk_common.h"
#include <VkBootstrap.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <VulkanMemoryAllocator/vk_mem_alloc.h>

namespace somegi {
namespace rhi { class VkRHIDevice; }

class Window;

struct DeviceFeatures {
    bool rayTracing = false;
    bool meshShader = false;   // VK_EXT_mesh_shader: Mesh Shader 支持
    bool taskShader = false;   // VK_EXT_mesh_shader: Task Shader 支持
    bool accelStruct = false;
    bool rayQuery = false;
    // Mesh Shader 属性限制（查询自 VkPhysicalDeviceMeshShaderPropertiesEXT）
    uint32_t maxMeshOutputVertices = 256;
    uint32_t maxMeshOutputPrimitives = 256;
    uint32_t maxMeshWorkGroupSize = 128;
};

// ════════════════════════════════════════════════════════════════
// Device — RHI 迁移过渡期的薄包装器
//
// 从 rhi::VkRHIDevice 提取原生 Vulkan 句柄，提供给仍使用
// Device& 签名的 96 个 renderer/GI pass 文件。这些文件后续将
// 逐步迁移到 rhi::RHIDevice。
//
// VkRHIDevice 拥有所有 Vulkan 对象的生命周期，Device 仅
// 持有一份拷贝（指针/值拷贝），析构时不销毁任何 Vulkan 对象。
// ════════════════════════════════════════════════════════════════
class Device {
public:
    // 从已存在的 VkRHIDevice 构造（VkRHIDevice 拥有对象生命周期）
    Device(rhi::VkRHIDevice& vkDev);

    // 不拥有 Vulkan 对象，析构不销毁任何东西
    ~Device() = default;

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    VkInstance instance() const { return m_instance; }
    VkPhysicalDevice physicalDevice() const { return m_physicalDevice; }
    VkDevice device() const { return m_device; }
    VkSurfaceKHR surface() const { return m_surface; }
    VkQueue graphicsQueue() const { return m_graphicsQueue; }
    uint32_t graphicsQueueFamily() const { return m_graphicsQueueFamily; }
    const DeviceFeatures& features() const { return m_features; }

    void waitIdle() const { vkDeviceWaitIdle(m_device); }

    float timestampPeriod() const { return m_timestampPeriod; }

    VkSampleCountFlags supportedSampleCounts() const {
        return m_supportedSampleCounts;
    }

    const vkb::DispatchTable& dispatch() const { return m_dispatch; }

    // Mesh Shader 扩展函数
    PFN_vkCmdDrawMeshTasksEXT vkCmdDrawMeshTasksEXT = nullptr;

    VmaAllocator allocator() const { return m_allocator; }

    // RHI 设备限制（从 VkRHIDevice 拷贝，供需要 DeviceLimits 的代码使用）
    rhi::DeviceLimits limits() const { return m_limits; }

private:
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueFamily = 0;
    DeviceFeatures m_features{};
    float m_timestampPeriod = 1.0f;
    uint32_t m_supportedSampleCounts = 0;
    vkb::DispatchTable m_dispatch{};
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    rhi::DeviceLimits m_limits{};
};

}
