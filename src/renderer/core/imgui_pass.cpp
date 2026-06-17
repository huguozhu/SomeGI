#include "renderer/core/imgui_pass.h"
#include "core/device.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

namespace somegi {

void ImGuiPass::init(Device& d, rhi::RHIDevice&, GLFWwindow* window, VkFormat swapchainFormat, uint32_t imageCount) {
    m_device = &d;

    VkDescriptorPoolSize ps[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024},
        {VK_DESCRIPTOR_TYPE_SAMPLER,                1024},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          1024},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1024},
    };
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets = 1024;
    pci.poolSizeCount = (uint32_t)(sizeof(ps) / sizeof(ps[0])); pci.pPoolSizes = ps;
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = "assets/imgui.ini";
    ImGui::StyleColorsDark();
    loadStyle();  // 从文件恢复控件尺寸样式

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
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &swapchainFormat;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = rci;

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

void ImGuiPass::saveSettings() {
    ImGuiIO& io = ImGui::GetIO();

    // 窗口尺寸/位置变更时立即写入 IniFilename
    if (io.WantSaveIniSettings && io.IniFilename) {
        ImGui::SaveIniSettingsToDisk(io.IniFilename);
    }

    // 保存 ImGuiStyle 控件尺寸到独立文件
    FILE* f = fopen("assets/imgui_style.ini", "w");
    if (!f) return;

    ImGuiStyle& s = ImGui::GetStyle();
    fprintf(f, "# ImGui Style — 控件尺寸/间距（每次变更自动保存）\n");
    fprintf(f, "[Style]\n");
    fprintf(f, "Alpha=%.3f\n", s.Alpha);
    fprintf(f, "DisabledAlpha=%.3f\n", s.DisabledAlpha);
    fprintf(f, "WindowPadding=%.1f,%.1f\n", s.WindowPadding.x, s.WindowPadding.y);
    fprintf(f, "WindowRounding=%.1f\n", s.WindowRounding);
    fprintf(f, "WindowBorderSize=%.1f\n", s.WindowBorderSize);
    fprintf(f, "WindowMinSize=%.1f,%.1f\n", s.WindowMinSize.x, s.WindowMinSize.y);
    fprintf(f, "ChildRounding=%.1f\n", s.ChildRounding);
    fprintf(f, "ChildBorderSize=%.1f\n", s.ChildBorderSize);
    fprintf(f, "PopupRounding=%.1f\n", s.PopupRounding);
    fprintf(f, "PopupBorderSize=%.1f\n", s.PopupBorderSize);
    fprintf(f, "FramePadding=%.1f,%.1f\n", s.FramePadding.x, s.FramePadding.y);
    fprintf(f, "FrameRounding=%.1f\n", s.FrameRounding);
    fprintf(f, "FrameBorderSize=%.1f\n", s.FrameBorderSize);
    fprintf(f, "ItemSpacing=%.1f,%.1f\n", s.ItemSpacing.x, s.ItemSpacing.y);
    fprintf(f, "ItemInnerSpacing=%.1f,%.1f\n", s.ItemInnerSpacing.x, s.ItemInnerSpacing.y);
    fprintf(f, "CellPadding=%.1f,%.1f\n", s.CellPadding.x, s.CellPadding.y);
    fprintf(f, "TouchExtraPadding=%.1f,%.1f\n", s.TouchExtraPadding.x, s.TouchExtraPadding.y);
    fprintf(f, "IndentSpacing=%.1f\n", s.IndentSpacing);
    fprintf(f, "ColumnsMinSpacing=%.1f\n", s.ColumnsMinSpacing);
    fprintf(f, "ScrollbarSize=%.1f\n", s.ScrollbarSize);
    fprintf(f, "ScrollbarRounding=%.1f\n", s.ScrollbarRounding);
    fprintf(f, "GrabMinSize=%.1f\n", s.GrabMinSize);
    fprintf(f, "GrabRounding=%.1f\n", s.GrabRounding);
    fprintf(f, "TabRounding=%.1f\n", s.TabRounding);
    fprintf(f, "TabBorderSize=%.1f\n", s.TabBorderSize);
    fprintf(f, "TabCloseButtonMinWidthUnselected=%.1f\n", s.TabCloseButtonMinWidthUnselected);
    fprintf(f, "TabCloseButtonMinWidthSelected=%.1f\n", s.TabCloseButtonMinWidthSelected);
    fprintf(f, "SeparatorTextPadding=%.1f,%.1f\n", s.SeparatorTextPadding.x, s.SeparatorTextPadding.y);
    fprintf(f, "DisplayWindowPadding=%.1f,%.1f\n", s.DisplayWindowPadding.x, s.DisplayWindowPadding.y);
    fprintf(f, "DisplaySafeAreaPadding=%.1f,%.1f\n", s.DisplaySafeAreaPadding.x, s.DisplaySafeAreaPadding.y);
    fprintf(f, "MouseCursorScale=%.1f\n", s.MouseCursorScale);
    fprintf(f, "CurveTessellationTol=%.1f\n", s.CurveTessellationTol);
    fprintf(f, "CircleTessellationMaxError=%.1f\n", s.CircleTessellationMaxError);
    fclose(f);
}

void ImGuiPass::loadStyle() {
    FILE* f = fopen("assets/imgui_style.ini", "r");
    if (!f) return;  // 首次运行无文件，使用默认样式

    ImGuiStyle& s = ImGui::GetStyle();
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        // 跳过注释和空行
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        char key[64] = {};
        float v0 = 0, v1 = 0;
        int n = sscanf(line, "%63[^=]=%f,%f", key, &v0, &v1);
        if (n < 2) continue;  // 不匹配的行跳过

        // Vec2 类型（两个 float）
        #define SET_VEC2(name) else if (strcmp(key, #name) == 0) { s.name = ImVec2(v0, v1); }
        #define SET_FLT(name)  else if (strcmp(key, #name) == 0) { s.name = v0; }

        if (0) {}
        SET_VEC2(WindowPadding)
        SET_VEC2(WindowMinSize)
        SET_VEC2(FramePadding)
        SET_VEC2(ItemSpacing)
        SET_VEC2(ItemInnerSpacing)
        SET_VEC2(CellPadding)
        SET_VEC2(TouchExtraPadding)
        SET_VEC2(SeparatorTextPadding)
        SET_VEC2(DisplayWindowPadding)
        SET_VEC2(DisplaySafeAreaPadding)
        SET_FLT(Alpha)
        SET_FLT(DisabledAlpha)
        SET_FLT(WindowRounding)
        SET_FLT(WindowBorderSize)
        SET_FLT(ChildRounding)
        SET_FLT(ChildBorderSize)
        SET_FLT(PopupRounding)
        SET_FLT(PopupBorderSize)
        SET_FLT(FrameRounding)
        SET_FLT(FrameBorderSize)
        SET_FLT(IndentSpacing)
        SET_FLT(ColumnsMinSpacing)
        SET_FLT(ScrollbarSize)
        SET_FLT(ScrollbarRounding)
        SET_FLT(GrabMinSize)
        SET_FLT(GrabRounding)
        SET_FLT(TabRounding)
        SET_FLT(TabBorderSize)
        SET_FLT(TabCloseButtonMinWidthUnselected)
        SET_FLT(TabCloseButtonMinWidthSelected)
        SET_FLT(MouseCursorScale)
        SET_FLT(CurveTessellationTol)
        SET_FLT(CircleTessellationMaxError)

        #undef SET_VEC2
        #undef SET_FLT
    }
    fclose(f);
}

void ImGuiPass::render(VkCommandBuffer cmd, VkImageView swapchainView, VkExtent2D extent) {
    ImGui::Render();

    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = swapchainView;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.clearValue.color = {{0.1f, 0.1f, 0.1f, 1.0f}};
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
