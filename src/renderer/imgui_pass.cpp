#include "imgui_pass.h"
#include "core/device.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

namespace somegi {

void ImGuiPass::init(Device& d, GLFWwindow* window, VkFormat swapchainFormat, uint32_t imageCount) {
    m_device = &d;

    VkDescriptorPoolSize ps[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024},
    };
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets = 1024;
    pci.poolSizeCount = 1; pci.pPoolSizes = ps;
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo info{};
    info.Instance = d.instance();
    info.PhysicalDevice = d.physicalDevice();
    info.Device = d.device();
    info.QueueFamily = d.graphicsQueueFamily();
    info.Queue = d.graphicsQueue();
    info.DescriptorPool = m_pool;
    info.MinImageCount = imageCount;
    info.ImageCount = imageCount;
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.UseDynamicRendering = true;

    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &swapchainFormat;
    info.PipelineRenderingCreateInfo = rci;

    if (!ImGui_ImplVulkan_Init(&info)) {
        throw std::runtime_error("ImGui_ImplVulkan_Init failed");
    }
    m_inited = true;
}

void ImGuiPass::destroy() {
    if (!m_inited) return;
    vkDeviceWaitIdle(m_device->device());
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (m_pool) vkDestroyDescriptorPool(m_device->device(), m_pool, nullptr);
    m_pool = VK_NULL_HANDLE;
    m_inited = false;
}

void ImGuiPass::newFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiPass::render(VkCommandBuffer cmd, VkImageView swapchainView, VkExtent2D extent) {
    ImGui::Render();

    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = swapchainView;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0,0}, extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    vkCmdBeginRendering(cmd, &ri);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

}
