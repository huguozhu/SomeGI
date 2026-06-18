// rhi/vulkan/vk_device.cpp
#include "vk_device.h"
#include "vk_buffer.h"
#include "vk_texture.h"
#include "vk_shader.h"
#include "vk_sampler.h"
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
// 构造函数 1：独立创建（拥有所有 Vulkan 句柄）
// ════════════════════════════════════════════════════════════════
VkRHIDevice::VkRHIDevice(void* nativeWindowHandle, bool enableValidation)
    : m_ownsDevice(true)
{
    // GPU-AV 环境变量（调试用）
    if (enableValidation) {
        _putenv_s("VK_LAYER_KHRONOS_VALIDATION_GPU_ASSISTED_EXT", "1");
        _putenv_s("VK_LAYER_KHRONOS_VALIDATION_SYNCHRONIZATION_VALIDATION_EXT", "1");
    }

    // ── Instance ──
    vkb::InstanceBuilder ib;
    ib.set_app_name("SomeGI")
      .require_api_version(1, 3, 0)
      .request_validation_layers(enableValidation)
      .use_default_debug_messenger();
    auto instRet = ib.build();
    if (!instRet) throw std::runtime_error("vkb::Instance: " + instRet.error().message());

    vkb::Instance vkbInst = instRet.value();
    m_vkInstance = vkbInst.instance;
    m_debugMessenger = vkbInst.debug_messenger;

    // ── Surface ──
    auto* window = static_cast<Window*>(nativeWindowHandle);
    m_surface = window->createSurface(m_vkInstance);

    // ── Physical Device ──
    vkb::PhysicalDeviceSelector ps{vkbInst};
    ps.set_minimum_version(1, 3)
      .set_surface(m_surface)
      .require_present()
      .add_required_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    auto pdRet = ps.select();
    if (!pdRet) throw std::runtime_error("vkb::PhysicalDevice: " + pdRet.error().message());
    vkb::PhysicalDevice vkbPD = pdRet.value();
    m_vkPhysicalDevice = vkbPD.physical_device;

    // 查询扩展支持（在已选中的 Physical Device 上查询）
    bool hasAS = false, hasRQ = false, hasDHO = false;
    for (auto& e : vkbPD.get_available_extensions()) {
        if (e == VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) hasAS = true;
        if (e == VK_KHR_RAY_QUERY_EXTENSION_NAME) hasRQ = true;
        if (e == VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) hasDHO = true;
    }
    if (hasAS && hasRQ && hasDHO) {
        vkbPD.enable_extension_if_present(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        vkbPD.enable_extension_if_present(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        vkbPD.enable_extension_if_present(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    }

    // Device fault
    vkbPD.enable_extension_if_present(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);

    // Mesh shader
    bool meshAvail = false;
    for (auto& e : vkbPD.get_available_extensions()) {
        if (e == VK_EXT_MESH_SHADER_EXTENSION_NAME) meshAvail = true;
    }
    if (meshAvail) vkbPD.enable_extension_if_present(VK_EXT_MESH_SHADER_EXTENSION_NAME);

    // ── Features ──
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

    vkb::DeviceBuilder db{vkbPD};
    if (hasAS && hasRQ && hasDHO) {
        asFeat.accelerationStructure = VK_TRUE;
        rqFeat.rayQuery = VK_TRUE;
        rqFeat.pNext = &asFeat;
        db.add_pNext(&rqFeat);
    }
    if (meshAvail) {
        msFeat.meshShader = VK_TRUE;
        msFeat.taskShader = meshAvail ? VK_TRUE : VK_FALSE;
        db.add_pNext(&msFeat);
    }

    auto devRet = db.build();
    if (!devRet) throw std::runtime_error("vkb::Device: " + devRet.error().message());
    vkb::Device vkbDevice = devRet.value();
    m_vkDevice = vkbDevice.device;
    m_dispatch = vkbDevice.make_table();

    m_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    m_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // ── VMA ──
    VmaVulkanFunctions vmaFuncs{};
    vmaFuncs.vkGetInstanceProcAddr = vkbInst.fp_vkGetInstanceProcAddr;
    vmaFuncs.vkGetDeviceProcAddr = vkbInst.fp_vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo vmaCI{};
    vmaCI.vulkanApiVersion = VK_API_VERSION_1_3;
    vmaCI.physicalDevice = m_vkPhysicalDevice;
    vmaCI.device = m_vkDevice;
    vmaCI.instance = m_vkInstance;
    vmaCI.pVulkanFunctions = &vmaFuncs;
    vmaCI.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&vmaCI, &m_allocator);

    // ── Limits ──
    auto& props = vkbPD.properties;
    m_limits.maxTextureSize = props.limits.maxImageDimension2D;
    m_limits.maxSampledTextures = props.limits.maxPerStageDescriptorSampledImages;
    m_limits.maxUniformBufferSize = props.limits.maxUniformBufferRange;
    m_limits.maxStorageBufferSize = props.limits.maxStorageBufferRange;
    m_limits.maxPushConstantsSize = props.limits.maxPushConstantsSize;
    m_limits.timestampPeriod = props.limits.timestampPeriod;
    m_limits.meshShaderSupported = meshAvail;
    m_limits.rayTracingSupported = (hasAS && hasRQ && hasDHO);

    // ── 提取完毕，清除 vkb 管理的句柄防止其析构时销毁 Vulkan 对象 ──
    // C++ 局部变量按声明逆序析构：vkbDevice → vkbPD → vkbInst
    // vkDestroyDevice(VK_NULL_HANDLE) / vkDestroyInstance(VK_NULL_HANDLE) 都是 no-op
    const_cast<VkDevice&>(vkbDevice.device) = VK_NULL_HANDLE;
    const_cast<VkInstance&>(vkbInst.instance) = VK_NULL_HANDLE;
    const_cast<VkDebugUtilsMessengerEXT&>(vkbInst.debug_messenger) = VK_NULL_HANDLE;
    // vkb 对象离开作用域，不会真正销毁 Vulkan 对象

    std::printf("[rhi] Vulkan device created\n");
}

// ════════════════════════════════════════════════════════════════
// 构造函数 2：从已有 core::Device 共享句柄（不拥有 Vulkan 对象）
// ════════════════════════════════════════════════════════════════
VkRHIDevice::VkRHIDevice(VkInstance instance, VkPhysicalDevice physicalDevice,
                         VkDevice device, VmaAllocator allocator,
                         VkQueue queue, uint32_t queueFamily,
                         const vkb::DispatchTable& dispatch,
                         const DeviceLimits& limits)
    : m_ownsDevice(false)
    , m_vkInstance(instance)
    , m_vkPhysicalDevice(physicalDevice)
    , m_vkDevice(device)
    , m_graphicsQueue(queue)
    , m_graphicsQueueFamily(queueFamily)
    , m_allocator(allocator)
    , m_limits(limits)
    , m_dispatch(dispatch)
{
    std::printf("[rhi] Vulkan device wrapped (shared with core::Device)\n");
}

// ════════════════════════════════════════════════════════════════
// 析构
// ════════════════════════════════════════════════════════════════
VkRHIDevice::~VkRHIDevice() {
    if (!m_ownsDevice) {
        // 共享模式：不销毁任何 Vulkan 对象（由 core::Device 管理生命周期）
        return;
    }

    // 独立模式：按创建顺序的逆序销毁
    vmaDestroyAllocator(m_allocator);

    // vkDestroyDevice 会隐式清理 device 级别的资源
    if (m_vkDevice) vkDestroyDevice(m_vkDevice, nullptr);

    if (m_surface) vkDestroySurfaceKHR(m_vkInstance, m_surface, nullptr);

    // 调试回调需要在 instance 销毁前清理
    // destroyDebugUtilsMessengerEXT 是 instance 级函数，不在设备级 DispatchTable 中
    if (m_debugMessenger) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_vkInstance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(m_vkInstance, m_debugMessenger, nullptr);
    }

    if (m_vkInstance) vkDestroyInstance(m_vkInstance, nullptr);
}

DeviceLimits VkRHIDevice::limits() const { return m_limits; }

// ════════════════════════════════════════════════════════════════
// 队列提交 + 呈现 + 同步
// ════════════════════════════════════════════════════════════════
void VkRHIDevice::submit(const SubmitDesc& desc) {
    VkCommandBufferSubmitInfo cbsi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbsi.commandBuffer = (VkCommandBuffer)(uintptr_t)desc.commandBuffer->nativeHandle();

    VkSemaphoreSubmitInfo waitSI{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    VkSemaphoreSubmitInfo sigSI{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    if (desc.waitSemaphore) {
        waitSI.semaphore = (VkSemaphore)(uintptr_t)desc.waitSemaphore->nativeHandle();
        waitSI.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
    if (desc.signalSemaphore) {
        sigSI.semaphore = (VkSemaphore)(uintptr_t)desc.signalSemaphore->nativeHandle();
        sigSI.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }

    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbsi;
    si.waitSemaphoreInfoCount = desc.waitSemaphore ? 1u : 0u;
    si.pWaitSemaphoreInfos = desc.waitSemaphore ? &waitSI : nullptr;
    si.signalSemaphoreInfoCount = desc.signalSemaphore ? 1u : 0u;
    si.pSignalSemaphoreInfos = desc.signalSemaphore ? &sigSI : nullptr;

    VkFence vkFence = desc.signalFence ? (VkFence)(uintptr_t)desc.signalFence->nativeHandle() : VK_NULL_HANDLE;
    m_dispatch.queueSubmit2(m_graphicsQueue, 1, &si, vkFence);
}

void VkRHIDevice::present(const RHISwapchain& swapchain, const RHISemaphore* waitSemaphore) {
    // 通过 nativeHandle 获取 VkSwapchainKHR（内部实现细节）
    // 注意：RHISwapchain 目前没有公开 VkSwapchainKHR 的方法，
    // 这里通过现有的 SwapchainFrame 机制调用
    (void)swapchain;
    (void)waitSemaphore;
    // 实际的 present 由 VkRHISwapchain::present() 执行，
    // 此方法作为接口占位，完整实现在 Phase 2 迁移交换链时完成。
}

void VkRHIDevice::waitForFence(const RHIFence& fence, uint64_t timeoutNs) {
    VkFence vkFence = (VkFence)(uintptr_t)fence.nativeHandle();
    m_dispatch.waitForFences(1, &vkFence, VK_TRUE, timeoutNs);
}

void VkRHIDevice::waitIdle() {
    if (m_vkDevice) vkDeviceWaitIdle(m_vkDevice);
}

// ════════════════════════════════════════════════════════════════
// 资源创建（委托给具体 vulkan/ 文件中的实现类）
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
std::unique_ptr<RHISampler> VkRHIDevice::createSampler(const SamplerDesc& desc) {
    return VkRHISampler::create(*this, desc);
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
std::unique_ptr<RHIPipelineState> VkRHIDevice::createRayTracingPSO(const RayTracingPSODesc& desc) {
    return VkRHIPipelineState::createRayTracing(*this, desc);
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
