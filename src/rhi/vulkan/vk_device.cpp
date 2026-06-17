// rhi/vulkan/vk_device.cpp
#include "vk_device.h"
#include "vk_buffer.h"
#include "vk_texture.h"
#include "vk_shader.h"
#include "vk_swapchain.h"
#include "vk_pso.h"
#include "vk_descriptor.h"
#include "vk_command.h"
#include "vk_fence.h"
#include "vk_query_pool.h"
#include <core/window.h>  // Window::createSurface
#include <cstdio>

namespace somegi {
namespace rhi {

// ════════════════════════════════════════════════════════════════
// 工厂方法
// ════════════════════════════════════════════════════════════════
std::unique_ptr<RHIDevice> RHIDevice::createVulkan(void* nativeWindowHandle, bool enableValidation) {
    return std::make_unique<VkRHIDevice>(nativeWindowHandle, enableValidation);
}

// ════════════════════════════════════════════════════════════════
// VkRHIDevice 构造
// ════════════════════════════════════════════════════════════════
VkRHIDevice::VkRHIDevice(void* nativeWindowHandle, bool enableValidation) {
    // GPU-AV 环境变量（调试用）
    if (enableValidation) {
        _putenv_s("VK_LAYER_KHRONOS_VALIDATION_GPU_ASSISTED_EXT", "1");
        _putenv_s("VK_LAYER_KHRONOS_VALIDATION_SYNCHRONIZATION_VALIDATION_EXT", "1");
    }

    // Instance
    vkb::InstanceBuilder ib;
    ib.set_app_name("SomeGI")
      .require_api_version(1, 3, 0)
      .request_validation_layers(enableValidation)
      .use_default_debug_messenger();
    auto instRet = ib.build();
    if (!instRet) throw std::runtime_error("vkb::Instance: " + instRet.error().message());
    m_instance = instRet.value();

    // Surface
    auto* window = static_cast<Window*>(nativeWindowHandle);
    m_surface = window->createSurface(m_instance.instance);

    // Physical device
    vkb::PhysicalDeviceSelector ps{m_instance};
    ps.set_minimum_version(1, 3)
      .set_surface(m_surface)
      .require_present()
      .add_required_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    auto pdRet = ps.select();
    if (!pdRet) throw std::runtime_error("vkb::PhysicalDevice: " + pdRet.error().message());
    m_physicalDevice = pdRet.value();

    // 查询扩展支持
    bool hasAS = false, hasRQ = false, hasDHO = false;
    for (auto& e : m_physicalDevice.get_available_extensions()) {
        if (e == VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) hasAS = true;
        if (e == VK_KHR_RAY_QUERY_EXTENSION_NAME) hasRQ = true;
        if (e == VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) hasDHO = true;
    }
    if (hasAS && hasRQ && hasDHO) {
        m_physicalDevice.enable_extension_if_present(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        m_physicalDevice.enable_extension_if_present(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        m_physicalDevice.enable_extension_if_present(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        m_limits.rayTracingSupported = true;
    }

    // Device fault
    m_physicalDevice.enable_extension_if_present(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);

    // Mesh shader
    bool meshAvail = false, taskAvail = false;
    for (auto& e : m_physicalDevice.get_available_extensions()) {
        if (e == VK_EXT_MESH_SHADER_EXTENSION_NAME) meshAvail = true;
    }
    if (meshAvail) m_physicalDevice.enable_extension_if_present(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    m_limits.meshShaderSupported = meshAvail;

    // Features
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    f13.maintenance4 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.descriptorIndexing = VK_TRUE;
    f12.runtimeDescriptorArray = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rqFeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceMeshShaderFeaturesEXT msFeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};

    vkb::DeviceBuilder db{m_physicalDevice};
    if (hasAS && hasRQ && hasDHO) {
        asFeat.accelerationStructure = VK_TRUE;
        rqFeat.rayQuery = VK_TRUE;
        rqFeat.pNext = &asFeat;
        db.add_pNext(&rqFeat);
    }
    if (meshAvail) {
        msFeat.meshShader = VK_TRUE;
        msFeat.taskShader = meshAvail ? VK_TRUE : VK_FALSE;
        m_limits.meshShaderSupported = true;
        db.add_pNext(&msFeat);
    }

    auto devRet = db.build();
    if (!devRet) throw std::runtime_error("vkb::Device: " + devRet.error().message());
    m_device = devRet.value();
    m_dispatch = m_device.make_table();

    m_graphicsQueue = m_device.get_queue(vkb::QueueType::graphics).value();
    m_graphicsQueueFamily = m_device.get_queue_index(vkb::QueueType::graphics).value();

    // VMA
    VmaVulkanFunctions vmaFuncs{};
    vmaFuncs.vkGetInstanceProcAddr = m_instance.fp_vkGetInstanceProcAddr;
    vmaFuncs.vkGetDeviceProcAddr = m_device.fp_vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo vmaCI{};
    vmaCI.vulkanApiVersion = VK_API_VERSION_1_3;
    vmaCI.physicalDevice = m_physicalDevice.physical_device;
    vmaCI.device = m_device.device;
    vmaCI.instance = m_instance.instance;
    vmaCI.pVulkanFunctions = &vmaFuncs;
    vmaCI.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&vmaCI, &m_allocator);

    // Limits
    auto& props = m_physicalDevice.properties;
    m_limits.maxTextureSize = props.limits.maxImageDimension2D;
    m_limits.maxSampledTextures = props.limits.maxPerStageDescriptorSampledImages;
    m_limits.maxUniformBufferSize = props.limits.maxUniformBufferRange;
    m_limits.maxStorageBufferSize = props.limits.maxStorageBufferRange;
    m_limits.maxPushConstantsSize = props.limits.maxPushConstantsSize;
    m_limits.timestampPeriod = props.limits.timestampPeriod;

    std::printf("[rhi] Vulkan device created\n");
}

VkRHIDevice::~VkRHIDevice() {
    vmaDestroyAllocator(m_allocator);
    vkDestroySurfaceKHR(m_instance.instance, m_surface, nullptr);
}

DeviceLimits VkRHIDevice::limits() const { return m_limits; }
void VkRHIDevice::waitIdle() { vkDeviceWaitIdle(m_device.device); }

// ════════════════════════════════════════════════════════════════
// 资源创建（桩实现，指向具体 vulkan/ 文件）
// ════════════════════════════════════════════════════════════════

std::unique_ptr<RHIBuffer> VkRHIDevice::createBuffer(const BufferDesc& desc) {
    return VkRHIBuffer::create(*this, desc);
}
std::unique_ptr<RHITexture> VkRHIDevice::createTexture(const TextureDesc& desc) {
    return VkRHITexture::create(*this, desc);
}
std::unique_ptr<RHITextureView> VkRHIDevice::createTextureView(const RHITexture& tex, const TextureViewDesc& desc) {
    return VkRHITextureView::create(*this, tex, desc);
}
std::unique_ptr<RHIShader> VkRHIDevice::createShader(const ShaderDesc& desc, const void* bytecode, size_t size) {
    return VkRHIShader::create(*this, desc, bytecode, size);
}
std::unique_ptr<RHISwapchain> VkRHIDevice::createSwapchain(void* nativeWindow, uint32_t width, uint32_t height) {
    return VkRHISwapchain::create(*this, nativeWindow, width, height);
}
std::unique_ptr<RHIPipelineState> VkRHIDevice::createGraphicsPSO(const GraphicsPSODesc& desc) {
    return VkRHIPipelineState::createGraphics(*this, desc);
}
std::unique_ptr<RHIPipelineState> VkRHIDevice::createComputePSO(const ComputePSODesc& desc) {
    return VkRHIPipelineState::createCompute(*this, desc);
}
std::unique_ptr<RHIDescriptorSetLayout> VkRHIDevice::createDescriptorSetLayout(const DescSetLayoutDesc& desc) {
    return VkRHIDescSetLayout::create(*this, desc);
}
std::unique_ptr<RHIDescriptorSet> VkRHIDevice::createDescriptorSet(const RHIDescriptorSetLayout& layout) {
    return VkRHIDescSet::create(*this, layout);
}
std::unique_ptr<RHICommandPool> VkRHIDevice::createCommandPool() {
    return VkRHICommandPool::create(*this);
}
std::unique_ptr<RHIFence> VkRHIDevice::createFence(bool signaled) {
    return VkRHIFence::create(*this, signaled);
}
std::unique_ptr<RHISemaphore> VkRHIDevice::createSemaphore() {
    return VkRHISemaphore::create(*this);
}
std::unique_ptr<RHIQueryPool> VkRHIDevice::createQueryPool(uint32_t count) {
    return VkRHIQueryPool::create(*this, count);
}

} // namespace rhi
} // namespace somegi
