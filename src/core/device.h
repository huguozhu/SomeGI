#pragma once
#include "vk_common.h"
#include <VkBootstrap.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <VulkanMemoryAllocator/vk_mem_alloc.h>

namespace somegi {

class Window;

struct DeviceFeatures {
    bool rayTracing = false;
    bool meshShader = false;   // VK_EXT_mesh_shader: Mesh Shader 支持
    bool taskShader = false;   // VK_EXT_mesh_shader: Task Shader 支持
    bool accelStruct = false;
    bool rayQuery = false;
};

class Device {
public:
    Device(Window& window, bool enableValidation);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    VkInstance instance() const { return m_instance.instance; }
    VkPhysicalDevice physicalDevice() const { return m_physicalDevice.physical_device; }
    VkDevice device() const { return m_device.device; }
    VkSurfaceKHR surface() const { return m_surface; }
    VkQueue graphicsQueue() const { return m_graphicsQueue; }
    uint32_t graphicsQueueFamily() const { return m_graphicsQueueFamily; }
    const DeviceFeatures& features() const { return m_features; }

    void waitIdle() const { vkDeviceWaitIdle(m_device.device); }

    // GPU 时间戳一个 tick 多少 ns。lim.timestampPeriod 来自 physical
    // device limits（Intel 一般 83.33ns，约等于 12 MHz）。
    float timestampPeriod() const { return m_physicalDevice.properties.limits.timestampPeriod; }

    // 查询 GPU 支持的 MSAA sample counts（color & depth 交集）
    VkSampleCountFlags supportedSampleCounts() const {
        auto& limits = m_physicalDevice.properties.limits;
        return limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts;
    }

    // vk-bootstrap dispatch table（加载了所有 extension 函数指针）。
    const vkb::DispatchTable& dispatch() const { return m_dispatch; }

    // Mesh Shader 扩展函数（vkGetDeviceProcAddr 加载）
    PFN_vkCmdDrawMeshTasksEXT vkCmdDrawMeshTasksEXT = nullptr;

    // VMA allocator
    VmaAllocator allocator() const { return m_allocator; }

private:
    vkb::Instance m_instance{};
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    vkb::PhysicalDevice m_physicalDevice{};
    vkb::Device m_device{};
    vkb::DispatchTable m_dispatch{};
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueFamily = 0;
    DeviceFeatures m_features{};
    VmaAllocator m_allocator = VK_NULL_HANDLE;
};

}
