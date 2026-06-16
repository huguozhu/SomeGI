#include "device.h"
#include "window.h"

namespace somegi {

Device::Device(Window& window, bool enableValidation) {
    // GPU-Assisted Validation: 检测 shader 越界/未初始化描述符。
    // 注意：GPU-AV 在 UNDEFINED oldLayout 过渡后读取图像时可能触发 device lost
    // （正确行为——它检测到了驱动丢弃的内容）。调试时可按需启用。
    if (enableValidation) {
        _putenv_s("VK_LAYER_KHRONOS_VALIDATION_GPU_ASSISTED_EXT", "1");
        _putenv_s("VK_LAYER_KHRONOS_VALIDATION_SYNCHRONIZATION_VALIDATION_EXT", "1");
    }

    vkb::InstanceBuilder ib;
    ib.set_app_name("SomeGI")
      .require_api_version(1, 3, 0)
      .request_validation_layers(enableValidation)
      .use_default_debug_messenger();
    auto instRet = ib.build();
    if (!instRet) throw std::runtime_error("vkb::Instance: " + instRet.error().message());
    m_instance = instRet.value();

    m_surface = window.createSurface(m_instance.instance);

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    f13.maintenance4 = VK_TRUE;
    f13.shaderDemoteToHelperInvocation = VK_TRUE;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.descriptorIndexing = VK_TRUE;
    f12.runtimeDescriptorArray = VK_TRUE;
    f12.descriptorBindingPartiallyBound = VK_TRUE;
    f12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    f12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    f12.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    f12.drawIndirectCount = VK_TRUE;

    // Slang emits the SPIR-V DrawParameters capability for shaders that touch
    // SV_VertexID / gl_BaseVertex (e.g. the skybox fullscreen triangle).
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.shaderDrawParameters = VK_TRUE;

    vkb::PhysicalDeviceSelector ps{m_instance};
    auto pdRet = ps.set_minimum_version(1, 3)
                    .set_surface(m_surface)
                    .set_required_features_13(f13)
                    .set_required_features_12(f12)
                    .set_required_features_11(f11)
                    .select();
    if (!pdRet) throw std::runtime_error("vkb::PhysicalDevice: " + pdRet.error().message());
    m_physicalDevice = pdRet.value();

    // Detect RT extensions from the physical device's available extensions.
    bool hasAccelStruct = false, hasRayQuery = false, hasDeferredHost = false;
    {
        auto availableExts = m_physicalDevice.get_available_extensions();
        for (auto& e : availableExts) {
            if (e == VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)  hasAccelStruct = true;
            if (e == VK_KHR_RAY_QUERY_EXTENSION_NAME)               hasRayQuery    = true;
            if (e == VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) hasDeferredHost = true;
            if (e == VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)    m_features.rayTracing = true;
            if (e == VK_EXT_MESH_SHADER_EXTENSION_NAME)             m_features.meshShader = true;  // 仅表示扩展可用，实际能力待查
        }
    }

    // Enable RT extensions on the physical device BEFORE constructing DeviceBuilder,
    // because DeviceBuilder copies the physical device at construction time.
    if (hasAccelStruct && hasRayQuery && hasDeferredHost) {
        m_physicalDevice.enable_extension_if_present(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        m_physicalDevice.enable_extension_if_present(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        m_physicalDevice.enable_extension_if_present(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        m_features.accelStruct = true;
        m_features.rayQuery = true;
    }

    // VK_EXT_device_fault: 获取 device lost 后的详细诊断信息
    bool hasDeviceFault = m_physicalDevice.enable_extension_if_present(
        VK_EXT_DEVICE_FAULT_EXTENSION_NAME);

    // Enable Mesh Shader extension + query actual feature support
    bool meshShaderAvail = false, taskShaderAvail = false;
    if (m_features.meshShader) {
        VkPhysicalDeviceMeshShaderFeaturesEXT meshQuery{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &meshQuery;
        vkGetPhysicalDeviceFeatures2(m_physicalDevice.physical_device, &f2);
        meshShaderAvail = meshQuery.meshShader == VK_TRUE;
        taskShaderAvail = meshQuery.taskShader == VK_TRUE;
        std::printf("[device] Mesh Shader query: mesh=%d task=%d\n", meshShaderAvail, taskShaderAvail);
    }
    if (meshShaderAvail) {
        m_physicalDevice.enable_extension_if_present(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    }
    m_features.meshShader = meshShaderAvail;
    m_features.taskShader = taskShaderAvail;

    // 查询 Mesh Shader 属性限制
    if (meshShaderAvail) {
        VkPhysicalDeviceMeshShaderPropertiesEXT meshProps{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &meshProps;
        vkGetPhysicalDeviceProperties2(m_physicalDevice.physical_device, &p2);
        m_features.maxMeshOutputVertices   = meshProps.maxMeshOutputVertices;
        m_features.maxMeshOutputPrimitives = meshProps.maxMeshOutputPrimitives;
        m_features.maxMeshWorkGroupSize    = meshProps.maxMeshWorkGroupSize[0];
        std::printf("[device] Mesh limits: maxVerts=%u maxPrims=%u maxWG=%u\n",
            meshProps.maxMeshOutputVertices, meshProps.maxMeshOutputPrimitives,
            meshProps.maxMeshWorkGroupSize[0]);
    }

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rqFeat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};

    vkb::DeviceBuilder db{m_physicalDevice};

    if (m_features.accelStruct && m_features.rayQuery) {
        asFeat.accelerationStructure = VK_TRUE;
        rqFeat.rayQuery = VK_TRUE;
        // Chain: rqFeat → asFeat → VkDeviceCreateInfo.pNext
        rqFeat.pNext = &asFeat;
        db.add_pNext(&rqFeat);
        std::printf("[device] HW ray tracing enabled (AS + RQ + DHO)\n");
    } else {
        std::printf("[device] HW ray tracing NOT available (AS=%d RQ=%d DH=%d)\n",
                    hasAccelStruct, hasRayQuery, hasDeferredHost);
    }

    // Mesh Shader: chain feature struct，仅启用 GPU 实际支持的特性
    VkPhysicalDeviceMeshShaderFeaturesEXT msFeat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    if (meshShaderAvail || taskShaderAvail) {
        msFeat.meshShader = meshShaderAvail ? VK_TRUE : VK_FALSE;
        msFeat.taskShader = taskShaderAvail ? VK_TRUE : VK_FALSE;
        db.add_pNext(&msFeat);
        std::printf("[device] Mesh Shader: mesh=%d task=%d\n", meshShaderAvail, taskShaderAvail);
    } else {
        std::printf("[device] Mesh Shader NOT available\n");
    }

    auto devRet = db.build();
    if (!devRet) throw std::runtime_error("vkb::Device: " + devRet.error().message());
    m_device = devRet.value();

    m_graphicsQueue = m_device.get_queue(vkb::QueueType::graphics).value();
    m_graphicsQueueFamily = m_device.get_queue_index(vkb::QueueType::graphics).value();

    // Load extension function pointers from dispatch table.
    m_dispatch = m_device.make_table();

    // Mesh Shader 扩展函数（不包含在 vkb::DispatchTable 中）
    if (meshShaderAvail) {
        vkCmdDrawMeshTasksEXT = (PFN_vkCmdDrawMeshTasksEXT)
            vkGetDeviceProcAddr(m_device.device, "vkCmdDrawMeshTasksEXT");
    }

    // Create VMA allocator
    VmaVulkanFunctions vmaFuncs{};
    vmaFuncs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vmaFuncs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo aci{};
    aci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    aci.physicalDevice = m_physicalDevice.physical_device;
    aci.device = m_device.device;
    aci.instance = m_instance.instance;
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    aci.pVulkanFunctions = &vmaFuncs;
    vmaCreateAllocator(&aci, &m_allocator);
}

Device::~Device() {
    if (m_allocator) vmaDestroyAllocator(m_allocator);
    if (m_device.device) vkb::destroy_device(m_device);
    if (m_surface) vkDestroySurfaceKHR(m_instance.instance, m_surface, nullptr);
    if (m_instance.instance) vkb::destroy_instance(m_instance);
}

}
