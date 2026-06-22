#include "device.h"
#include "rhi/vulkan/vk_device.h"

namespace somegi {

Device::Device(rhi::VkRHIDevice& vkDev) {
    auto limits = vkDev.limits();

    m_instance = vkDev.vkInstance();
    m_physicalDevice = vkDev.vkPhysicalDevice();
    m_device = vkDev.vkDevice();
    m_surface = vkDev.surface();
    m_graphicsQueue = vkDev.vkQueue();
    m_graphicsQueueFamily = vkDev.queueFamily();
    m_dispatch = vkDev.dispatch();
    m_allocator = vkDev.vma();
    m_timestampPeriod = limits.timestampPeriod;
    m_supportedSampleCounts = limits.supportedSampleCounts;

    // 将 rhi::DeviceLimits 映射到 core::DeviceFeatures
    m_features.meshShader = limits.meshShaderSupported;
    m_features.taskShader = limits.taskShaderSupported;
    m_features.rayTracing = limits.rayTracingSupported;
    m_features.accelStruct = limits.accelStructSupported;
    m_features.rayQuery = limits.rayQuerySupported;
    m_features.maxMeshOutputVertices = limits.maxMeshOutputVertices;
    m_features.maxMeshOutputPrimitives = limits.maxMeshOutputPrimitives;
    m_features.maxMeshWorkGroupSize = limits.maxMeshWorkGroupSize;

    // 扩展函数指针
    vkCmdDrawMeshTasksEXT = vkDev.vkCmdDrawMeshTasksEXT;

    std::printf("[device] thin-wrapper initialized from VkRHIDevice\n");
}

}
