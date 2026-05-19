#include "device.h"
#include "window.h"

namespace somegi {

Device::Device(Window& window, bool enableValidation) {
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

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.descriptorIndexing = VK_TRUE;
    f12.runtimeDescriptorArray = VK_TRUE;
    f12.descriptorBindingPartiallyBound = VK_TRUE;
    f12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

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

    vkb::DeviceBuilder db{m_physicalDevice};
    auto devRet = db.build();
    if (!devRet) throw std::runtime_error("vkb::Device: " + devRet.error().message());
    m_device = devRet.value();

    m_graphicsQueue = m_device.get_queue(vkb::QueueType::graphics).value();
    m_graphicsQueueFamily = m_device.get_queue_index(vkb::QueueType::graphics).value();

    auto exts = m_physicalDevice.get_extensions();
    for (auto& e : exts) {
        if (e == VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) m_features.rayTracing = true;
        if (e == VK_EXT_MESH_SHADER_EXTENSION_NAME) m_features.meshShader = true;
    }
}

Device::~Device() {
    if (m_device.device) vkb::destroy_device(m_device);
    if (m_surface) vkDestroySurfaceKHR(m_instance.instance, m_surface, nullptr);
    if (m_instance.instance) vkb::destroy_instance(m_instance);
}

}
