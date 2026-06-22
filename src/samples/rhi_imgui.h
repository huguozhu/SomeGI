// rhi_imgui.h — RHI 无关的 ImGui 渲染封装
// 支持 Vulkan 和 D3D12 后端，运行时切换
#pragma once

#include "rhi/base/common.h"  // Backend, Format
#include <memory>

struct GLFWwindow;

namespace somegi {
namespace rhi {
    class RHIDevice;
    class RHICommandBuffer;
    class RHISwapchain;
    struct SwapchainFrame;
}
}

// ImGui 渲染封装：管理 ImGui 上下文 + 后端初始化和每帧渲染
// 后端切换流程：shutdown() → init(newDevice, ...) → 正常帧循环
class RhiImGuiRenderer {
public:
    ~RhiImGuiRenderer();

    // 初始化 ImGui + 选择后端渲染器
    void init(somegi::rhi::RHIDevice& device, GLFWwindow* window,
              somegi::rhi::Format swapchainFormat, uint32_t imageCount);

    // 销毁当前后端渲染器（保留 ImGui 上下文和应用状态）
    void shutdown();

    bool isInitialized() const { return m_inited; }

    // 单帧生命周期：newFrame 在渲染前调用，render 在 beginRendering 之后调用
    void newFrame();
    void render(somegi::rhi::RHICommandBuffer& cmd);

private:
    void initVulkan(somegi::rhi::RHIDevice& device, GLFWwindow* window,
                    somegi::rhi::Format swapchainFormat, uint32_t imageCount);
    void initD3D12(somegi::rhi::RHIDevice& device, GLFWwindow* window,
                   somegi::rhi::Format swapchainFormat, uint32_t imageCount);

    bool m_inited = false;
    somegi::rhi::Backend m_backend{};

    // Vulkan 专用句柄（用 void* 避免头文件暴露 Vulkan 类型）
    void* m_vkPool = nullptr;       // VkDescriptorPool
    void* m_nativeDevice = nullptr; // VkDevice 或 ID3D12Device*

    // D3D12 专用：存储 ImGui_ImplDX12_InitInfo（不完整类型，cpp 内构造/析构）
    void* m_d3dInfo = nullptr;      // ImGui_ImplDX12_InitInfo*
};
