// rhi_imgui.cpp — RHI 无关的 ImGui 渲染封装实现
#include "rhi_imgui.h"

#include "rhi/base/device.h"
#include "rhi/base/command_buffer.h"
#include "rhi/base/swapchain.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/d3d12/d3d12_device.h"
#include "rhi/d3d12/d3d12_command.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#ifdef WIN32
#include <imgui_impl_dx12.h>
#endif

#include <stdexcept>
#include <cstdio>

// ════════════════════════════════════════════════════════════════
// 格式转换辅助
// ════════════════════════════════════════════════════════════════

static VkFormat toVkFormat(somegi::rhi::Format fmt) {
    using F = somegi::rhi::Format;
    switch (fmt) {
        case F::B8G8R8A8_UNORM:    return VK_FORMAT_B8G8R8A8_UNORM;
        case F::B8G8R8A8_SRGB:     return VK_FORMAT_B8G8R8A8_SRGB;
        case F::R8G8B8A8_UNORM:    return VK_FORMAT_R8G8B8A8_UNORM;
        case F::R8G8B8A8_SRGB:     return VK_FORMAT_R8G8B8A8_SRGB;
        case F::R16G16B16A16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        default: return VK_FORMAT_B8G8R8A8_UNORM;
    }
}

#ifdef WIN32
static DXGI_FORMAT toDxgiFormat(somegi::rhi::Format fmt) {
    using F = somegi::rhi::Format;
    switch (fmt) {
        case F::B8G8R8A8_UNORM:    return DXGI_FORMAT_B8G8R8A8_UNORM;
        case F::B8G8R8A8_SRGB:     return DXGI_FORMAT_B8G8R8A8_UNORM; // D3D12 交换链不支持 _SRGB
        case F::R8G8B8A8_UNORM:    return DXGI_FORMAT_R8G8B8A8_UNORM;
        case F::R8G8B8A8_SRGB:     return DXGI_FORMAT_R8G8B8A8_UNORM;
        case F::R16G16B16A16_SFLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        default: return DXGI_FORMAT_B8G8R8A8_UNORM;
    }
}
#endif

// ════════════════════════════════════════════════════════════════
// 构造 / 析构
// ════════════════════════════════════════════════════════════════

RhiImGuiRenderer::~RhiImGuiRenderer() {
    shutdown();
}

// ════════════════════════════════════════════════════════════════
// 公共接口
// ════════════════════════════════════════════════════════════════

void RhiImGuiRenderer::init(somegi::rhi::RHIDevice& device, GLFWwindow* window,
                             somegi::rhi::Format swapchainFormat, uint32_t imageCount) {
    if (m_inited) shutdown();

    m_backend = device.backend();

    // ImGui 上下文只创建一次（shutdown 时也不销毁，仅销毁后端）
    static bool s_contextCreated = false;
    if (!s_contextCreated) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::GetIO().IniFilename = nullptr; // 不保存窗口布局
        ImGui::StyleColorsDark();

        // ── 加载中文字体（优先微软雅黑，回退黑体/宋体） ──
        ImFontConfig fontCfg{};
        fontCfg.MergeMode = false;
        // 简体中文常用字符 + 默认 ASCII
        const ImWchar* chineseRange = ImGui::GetIO().Fonts->GetGlyphRangesChineseSimplifiedCommon();

        const char* fontPaths[] = {
            "C:/Windows/Fonts/msyh.ttc",    // 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  // 黑体
            "C:/Windows/Fonts/simsun.ttc",  // 宋体
        };
        bool fontLoaded = false;
        for (const char* path : fontPaths) {
            if (ImGui::GetIO().Fonts->AddFontFromFileTTF(path, 18.0f, &fontCfg, chineseRange)) {
                std::printf("[RhiImGui] 加载字体: %s\n", path);
                fontLoaded = true;
                break;
            }
        }
        if (!fontLoaded) {
            // 无中文字体时回退默认字体（中文将显示为方框）
            ImGui::GetIO().Fonts->AddFontDefault();
            std::printf("[RhiImGui] 未找到中文字体，使用默认字体（中文显示为方框）\n");
        }

        s_contextCreated = true;
    }

    switch (m_backend) {
        case somegi::rhi::Backend::Vulkan:
            initVulkan(device, window, swapchainFormat, imageCount);
            break;
        case somegi::rhi::Backend::D3D12:
            initD3D12(device, window, swapchainFormat, imageCount);
            break;
        default:
            throw std::runtime_error("[RhiImGui] 不支持的后端");
    }

    m_inited = true;
    std::printf("[RhiImGui] 初始化完成 (backend=%s)\n",
                m_backend == somegi::rhi::Backend::Vulkan ? "Vulkan" : "D3D12");
}

void RhiImGuiRenderer::shutdown() {
    if (!m_inited) return;

    // 等待设备空闲，确保 GPU 不再使用 ImGui 资源
    // 调用者应在 shutdown 前调用 device.waitIdle()

    switch (m_backend) {
        case somegi::rhi::Backend::Vulkan: {
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            if (m_vkPool && m_nativeDevice) {
                vkDestroyDescriptorPool((VkDevice)m_nativeDevice, (VkDescriptorPool)m_vkPool, nullptr);
                m_vkPool = nullptr;
                m_nativeDevice = nullptr;
            }
            break;
        }
        case somegi::rhi::Backend::D3D12: {
#ifdef WIN32
            ImGui_ImplDX12_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            if (m_d3dInfo) {
                delete static_cast<ImGui_ImplDX12_InitInfo*>(m_d3dInfo);
                m_d3dInfo = nullptr;
            }
            m_nativeDevice = nullptr;
#endif
            break;
        }
        default: break;
    }

    m_inited = false;
    std::printf("[RhiImGui] shutdown 完成\n");
}

void RhiImGuiRenderer::newFrame() {
    if (!m_inited) return;

    switch (m_backend) {
        case somegi::rhi::Backend::Vulkan:
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            break;
        case somegi::rhi::Backend::D3D12:
#ifdef WIN32
            ImGui_ImplDX12_NewFrame();
            ImGui_ImplGlfw_NewFrame();
#endif
            break;
        default: break;
    }
    ImGui::NewFrame();
}

void RhiImGuiRenderer::render(somegi::rhi::RHICommandBuffer& cmd) {
    if (!m_inited) return;

    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();
    if (!drawData || drawData->CmdListsCount == 0) return;

    // 通过原生句柄桥接：ImGui 后端直接录制到原生命令缓冲
    switch (m_backend) {
        case somegi::rhi::Backend::Vulkan:
            ImGui_ImplVulkan_RenderDrawData(drawData,
                (VkCommandBuffer)(uintptr_t)cmd.nativeHandle());
            break;
        case somegi::rhi::Backend::D3D12:
#ifdef WIN32
            ImGui_ImplDX12_RenderDrawData(drawData,
                (ID3D12GraphicsCommandList*)cmd.nativeHandle());
#endif
            break;
        default: break;
    }
}

// ════════════════════════════════════════════════════════════════
// Vulkan 后端初始化
// ════════════════════════════════════════════════════════════════

void RhiImGuiRenderer::initVulkan(somegi::rhi::RHIDevice& device, GLFWwindow* window,
                                   somegi::rhi::Format swapchainFormat, uint32_t imageCount) {
    auto& vkDev = static_cast<somegi::rhi::VkRHIDevice&>(device);

    // ── 创建描述符池（ImGui 内部需要多种描述符类型，必须足够的 maxSets） ──
    VkDescriptorPoolSize ps[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16},
        {VK_DESCRIPTOR_TYPE_SAMPLER,                16},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          16},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          16},
    };
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets = 64;
    pci.poolSizeCount = (uint32_t)(sizeof(ps) / sizeof(ps[0]));
    pci.pPoolSizes = ps;

    VkDescriptorPool pool;
    if (vkCreateDescriptorPool(vkDev.vkDevice(), &pci, nullptr, &pool) != VK_SUCCESS) {
        throw std::runtime_error("[RhiImGui] 创建 VkDescriptorPool 失败");
    }
    m_vkPool = pool;
    m_nativeDevice = (void*)vkDev.vkDevice();

    // ── 初始化 GLFW + Vulkan 后端 ──
    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo info{};
    info.Instance = vkDev.vkInstance();
    info.PhysicalDevice = vkDev.vkPhysicalDevice();
    info.Device = vkDev.vkDevice();
    info.QueueFamily = vkDev.queueFamily();
    info.Queue = static_cast<VkQueue>(vkDev.nativeQueue());
    info.DescriptorPool = pool;
    info.MinImageCount = imageCount;
    info.ImageCount = imageCount;
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    VkFormat vkFmt = toVkFormat(swapchainFormat);
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &vkFmt;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = rci;

    if (!ImGui_ImplVulkan_Init(&info)) {
        throw std::runtime_error("[RhiImGui] ImGui_ImplVulkan_Init 失败");
    }
    // 字体纹理在首次 NewFrame() 时自动上传
}

// ════════════════════════════════════════════════════════════════
// D3D12 后端初始化
// ════════════════════════════════════════════════════════════════

#ifdef WIN32

// D3D12 SRV 描述符分配回调（从 GPU 描述符堆的持久区域分配）
static void ImGuiD3D12_SrvAlloc(ImGui_ImplDX12_InitInfo* info,
                                 D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu,
                                 D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu) {
    auto* d3dDev = static_cast<somegi::rhi::D3D12RHIDevice*>(info->UserData);
    auto alloc = d3dDev->allocPersistentDescriptors(1);
    *out_cpu = alloc.cpu;
    *out_gpu = alloc.gpu;
}

// D3D12 SRV 描述符释放回调（持久区域不释放，因为堆在设备析构时统一释放）
static void ImGuiD3D12_SrvFree(ImGui_ImplDX12_InitInfo*,
                                D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE) {
    // 持久分配无需逐帧释放
}

#endif // WIN32

void RhiImGuiRenderer::initD3D12(somegi::rhi::RHIDevice& device, GLFWwindow* window,
                                  somegi::rhi::Format swapchainFormat, uint32_t imageCount) {
#ifdef WIN32
    auto& d3dDev = static_cast<somegi::rhi::D3D12RHIDevice&>(device);

    // ── 初始化 GLFW 后端（非 Vulkan/OpenGL 模式） ──
    ImGui_ImplGlfw_InitForOther(window, true);

    // ── 构造 D3D12 初始化信息 ──
    m_d3dInfo = new ImGui_ImplDX12_InitInfo();
    auto& info = *static_cast<ImGui_ImplDX12_InitInfo*>(m_d3dInfo);
    info.Device = d3dDev.device();
    info.CommandQueue = d3dDev.commandQueue();
    info.NumFramesInFlight = (int)imageCount;
    info.RTVFormat = toDxgiFormat(swapchainFormat);
    info.DSVFormat = DXGI_FORMAT_UNKNOWN;  // 无深度缓冲
    info.UserData = &d3dDev;
    info.SrvDescriptorHeap = d3dDev.gpuDescriptorHeap();
    info.SrvDescriptorAllocFn = ImGuiD3D12_SrvAlloc;
    info.SrvDescriptorFreeFn = ImGuiD3D12_SrvFree;

    if (!ImGui_ImplDX12_Init(&info)) {
        delete static_cast<ImGui_ImplDX12_InitInfo*>(m_d3dInfo);
        m_d3dInfo = nullptr;
        throw std::runtime_error("[RhiImGui] ImGui_ImplDX12_Init 失败");
    }
    m_nativeDevice = (void*)d3dDev.device();
#else
    (void)device; (void)window; (void)swapchainFormat; (void)imageCount;
    throw std::runtime_error("[RhiImGui] D3D12 仅在 Windows 上支持");
#endif
}
