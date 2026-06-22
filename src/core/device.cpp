#include "device.h"
#include "rhi/vulkan/vk_device.h"

namespace somegi {

Device::Device(rhi::VkRHIDevice& vkDev) : m_rhiDev(&vkDev) {
    m_limits = vkDev.limits();
    m_timestampPeriod = m_limits.timestampPeriod;
    m_supportedSampleCounts = m_limits.supportedSampleCounts;

    m_features.meshShader = m_limits.meshShaderSupported;
    m_features.taskShader = m_limits.taskShaderSupported;
    m_features.rayTracing = m_limits.rayTracingSupported;
    m_features.accelStruct = m_limits.accelStructSupported;
    m_features.rayQuery = m_limits.rayQuerySupported;
    m_features.maxMeshOutputVertices = m_limits.maxMeshOutputVertices;
    m_features.maxMeshOutputPrimitives = m_limits.maxMeshOutputPrimitives;
    m_features.maxMeshWorkGroupSize = m_limits.maxMeshWorkGroupSize;

    vkCmdDrawMeshTasksEXT = vkDev.vkCmdDrawMeshTasksEXT;

    std::printf("[device] thin-wrapper delegates to VkRHIDevice\n");
}

VkInstance Device::instance() const          { return m_rhiDev->vkInstance(); }
VkPhysicalDevice Device::physicalDevice() const { return m_rhiDev->vkPhysicalDevice(); }
VkDevice Device::device() const              { return m_rhiDev->vkDevice(); }
VkSurfaceKHR Device::surface() const         { return m_rhiDev->surface(); }
VkQueue Device::graphicsQueue() const        { return m_rhiDev->vkQueue(); }
uint32_t Device::graphicsQueueFamily() const { return m_rhiDev->queueFamily(); }
const vkb::DispatchTable& Device::dispatch() const { return m_rhiDev->dispatch(); }
VmaAllocator Device::allocator() const       { return m_rhiDev->vma(); }
void Device::waitIdle() const                { m_rhiDev->waitIdle(); }

}
