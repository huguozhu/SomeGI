// RHI 跨后端示例 — 纯 RHI 抽象层绘制三角形 + ImGui 控制面板
// 支持 Vulkan 和 D3D12 后端，运行时通过下拉菜单热切换

#include "rhi/base/device.h"
#include "rhi/base/swapchain.h"
#include "rhi/base/shader.h"
#include "rhi/base/command_buffer.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/texture.h"
#include "rhi/base/fence.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_fence.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/d3d12/d3d12_device.h"
#include "rhi/d3d12/d3d12_swapchain.h"
#include "rhi/d3d12/d3d12_texture.h"
#include "rhi/d3d12/d3d12_command.h"

#include "rhi_imgui.h"
#include "core/window.h"

#include <imgui.h>

#include <cstdio>
#include <fstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>

#ifdef WIN32
#include <d3dcompiler.h>
#pragma comment(lib, "d3dcompiler.lib")
#endif

using namespace somegi;

// ════════════════════════════════════════════════════════════════
// 后端选择状态（全局，因为 ImGui 回调需要静态访问）
// ════════════════════════════════════════════════════════════════
static rhi::Backend  s_currentBackend = rhi::Backend::Vulkan;
static bool          s_pendingSwitch  = false;   // 本帧检测到切换请求

// ════════════════════════════════════════════════════════════════
// SPIR-V 文件加载（Vulkan 专用）
// ════════════════════════════════════════════════════════════════
static std::vector<uint32_t> loadSpv(const char* name) {
    std::ifstream f(name, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("无法打开文件: ") + name);
    auto size = static_cast<size_t>(f.tellg());
    std::vector<uint32_t> data(size / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

// ════════════════════════════════════════════════════════════════
// HLSL → DXIL 运行时编译（D3D12 专用）
// ════════════════════════════════════════════════════════════════
#ifdef WIN32
// 与 GLSL triangle.vert 等价的 HLSL 顶点着色器
static const char* kTriangleVS_HLSL = R"(
struct VSInput  { uint   vertexId : SV_VertexID; };
struct VSOutput { float4 position : SV_POSITION; float3 color : COLOR0; };

VSOutput main(VSInput input) {
    float2 positions[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };
    float3 colors[3]    = { float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0) };

    VSOutput output;
    output.position = float4(positions[input.vertexId], 0.0, 1.0);
    output.color    = colors[input.vertexId];
    return output;
}
)";

// 与 GLSL triangle.frag 等价的 HLSL 片段着色器
static const char* kTrianglePS_HLSL = R"(
struct PSInput { float4 position : SV_POSITION; float3 color : COLOR0; };

float4 main(PSInput input) : SV_TARGET {
    return float4(input.color, 1.0);
}
)";

// 编译 HLSL 源码为 DXIL bytecode
// 使用动态加载，避免 d3dcompiler DLL 导入表解析问题
static std::vector<uint8_t> compileHlsl(const char* source, const char* entryPoint,
                                         const char* target) {
    // 动态加载 d3dcompiler DLL（避免导入表解析问题）
    HMODULE hD3DCompiler = LoadLibraryA("D3DCompiler_47.dll");
    if (!hD3DCompiler) {
        throw std::runtime_error("[D3D12] 无法加载 D3DCompiler_47.dll");
    }

    using D3DCompileFunc = HRESULT (*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*,
                                        ID3DInclude*, LPCSTR, LPCSTR, UINT, UINT,
                                        ID3DBlob**, ID3DBlob**);
    auto pD3DCompile = (D3DCompileFunc)GetProcAddress(hD3DCompiler, "D3DCompile");
    if (!pD3DCompile) {
        FreeLibrary(hD3DCompiler);
        throw std::runtime_error("[D3D12] 无法找到 D3DCompile 函数");
    }

    ID3DBlob* codeBlob = nullptr;
    ID3DBlob* errBlob  = nullptr;

    UINT compileFlags = 0;
#if defined(_DEBUG)
    compileFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    HRESULT hr = pD3DCompile(source, strlen(source), nullptr, nullptr, nullptr,
                              entryPoint, target, compileFlags, 0, &codeBlob, &errBlob);
    FreeLibrary(hD3DCompiler);

    if (FAILED(hr)) {
        std::string err = "HLSL 编译失败 (unknown error)";
        if (errBlob) {
            err = std::string(static_cast<const char*>(errBlob->GetBufferPointer()),
                              errBlob->GetBufferSize());
            errBlob->Release();
        }
        throw std::runtime_error("[D3D12] " + err);
    }
    if (errBlob) errBlob->Release();

    std::vector<uint8_t> bytecode(codeBlob->GetBufferSize());
    memcpy(bytecode.data(), codeBlob->GetBufferPointer(), bytecode.size());
    codeBlob->Release();
    return bytecode;
}
#endif

// ════════════════════════════════════════════════════════════════
// 资源创建辅助 — 根据后端创建着色器
// ════════════════════════════════════════════════════════════════
struct TriangleResources {
    std::unique_ptr<rhi::RHIShader> vs;
    std::unique_ptr<rhi::RHIShader> fs;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> setLayout;
    std::unique_ptr<rhi::RHIDescriptorSet> emptySet;
    std::unique_ptr<rhi::RHIPipelineState> pso;

    void createShaders(rhi::RHIDevice& device) {
        rhi::ShaderDesc sd;
        sd.entryPoint = "main";

        switch (device.backend()) {
            case rhi::Backend::Vulkan: {
                auto vertSpv = loadSpv("triangle.vert.spv");
                auto fragSpv = loadSpv("triangle.frag.spv");
                sd.stage = rhi::ShaderStage::Vertex;
                vs = device.createShader(sd, vertSpv.data(), vertSpv.size() * 4);
                sd.stage = rhi::ShaderStage::Fragment;
                fs = device.createShader(sd, fragSpv.data(), fragSpv.size() * 4);
                break;
            }
            case rhi::Backend::D3D12: {
#ifdef WIN32
                // D3DCompile 编译 HLSL→DXBC（vs_5_0/ps_5_0），D3D12 可消费 DXBC
                auto vsDxil = compileHlsl(kTriangleVS_HLSL, "main", "vs_5_0");
                auto fsDxil = compileHlsl(kTrianglePS_HLSL, "main", "ps_5_0");
                sd.stage = rhi::ShaderStage::Vertex;
                sd.format = rhi::ShaderFormat::DXIL;
                vs = device.createShader(sd, vsDxil.data(), vsDxil.size());
                sd.stage = rhi::ShaderStage::Fragment;
                fs = device.createShader(sd, fsDxil.data(), fsDxil.size());
#endif
                break;
            }
            default: break;
        }
    }

    void createPSO(rhi::RHIDevice& device, rhi::Format swapchainFmt) {
        rhi::DescSetLayoutDesc ld; ld.debugName = "Triangle";
        setLayout = device.createDescriptorSetLayout(ld);
        emptySet  = device.createDescriptorSet(*setLayout);

        rhi::GraphicsPSODesc pd; pd.debugName = "Triangle";
        pd.vertexShader   = vs.get();
        pd.fragmentShader = fs.get();
        pd.topology       = rhi::PrimitiveTopology::TriangleList;
        pd.rasterization  = { rhi::FillMode::Solid, rhi::CullMode::None, false };
        pd.renderTargets.colorFormats = { swapchainFmt };
        pd.descriptorSetLayouts = { setLayout.get() };
        pso = device.createGraphicsPSO(pd);
    }

    // 销毁前按顺序释放（依赖倒序）
    void reset() {
        pso.reset();
        emptySet.reset();
        setLayout.reset();
        fs.reset();
        vs.reset();
    }
};

// ════════════════════════════════════════════════════════════════
// 主函数
// ════════════════════════════════════════════════════════════════
int main() {
    // ── 创建窗口 ──
    WindowDesc wd; wd.title = "RHI Triangle"; wd.width = 800; wd.height = 600;
    Window window(wd);

    // ── 全局资源持有者（unique_ptr 自动管理生命周期） ──
    std::unique_ptr<rhi::RHIDevice>       rhiDevice;
    std::unique_ptr<rhi::RHISwapchain>    swapchain;
    std::unique_ptr<rhi::RHICommandPool>  cmdPool;
    rhi::RHICommandBuffer*                cmd0 = nullptr;  // 双缓冲命令缓冲
    rhi::RHICommandBuffer*                cmd1 = nullptr;
    TriangleResources                     triRes;
    RhiImGuiRenderer                      imgui;
    uint32_t                              frameIdx = 0;

    // ── 创建设备和所有资源的函数（首次启动 + 后端切换复用） ──
    auto createAllResources = [&]() {
        rhiDevice = rhi::RHIDevice::create(s_currentBackend, &window, false);
        swapchain = rhiDevice->createSwapchain(&window, wd.width, wd.height);

        triRes.createShaders(*rhiDevice);
        triRes.createPSO(*rhiDevice, swapchain->format());

        cmdPool = rhiDevice->createCommandPool();
        cmd0 = cmdPool->allocateRaw();
        cmd1 = cmdPool->allocateRaw();

        imgui.init(*rhiDevice, window.handle(), swapchain->format(), 2);
        frameIdx = 0;
    };

    // ── 销毁所有资源的函数（后端切换前调用） ──
    auto destroyAllResources = [&]() {
        if (rhiDevice) rhiDevice->waitIdle();
        imgui.shutdown();
        // 先释放命令缓冲（在 cmdPool 析构前）
        delete cmd0; cmd0 = nullptr;
        delete cmd1; cmd1 = nullptr;
        cmdPool.reset();
        triRes.reset();
        swapchain.reset();
        rhiDevice.reset();
    };

    // ── 首次创建 ──
    createAllResources();

    // ════════════════════════════════════════════════════════════
    // 主渲染循环
    // ════════════════════════════════════════════════════════════
    std::printf("[triangle] 进入渲染循环\n");
    while (!window.shouldClose()) {
        window.pollEvents();

        // ── ImGui 新帧 ──
        imgui.newFrame();

        // ── 控制面板 UI ──
        {
            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(220, 130), ImGuiCond_FirstUseEver);
            ImGui::Begin("RHI 控制面板", nullptr,
                         ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize);

            int backendIdx = (s_currentBackend == rhi::Backend::Vulkan) ? 0 : 1;
            if (ImGui::Combo("渲染后端", &backendIdx, "Vulkan\0D3D12\0")) {
                rhi::Backend newBackend = (backendIdx == 0) ? rhi::Backend::Vulkan
                                                            : rhi::Backend::D3D12;
                if (newBackend != s_currentBackend) {
                    s_currentBackend = newBackend;
                    s_pendingSwitch  = true;
                }
            }

            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Text("帧数: %u", frameIdx);
            ImGui::Text("后端: %s", s_currentBackend == rhi::Backend::Vulkan ? "Vulkan" : "D3D12");
            ImGui::End();
        }

        // ── 后端热切换 ──
        if (s_pendingSwitch) {
            s_pendingSwitch = false;
            std::printf("[triangle] 切换后端 → %s\n",
                        s_currentBackend == rhi::Backend::Vulkan ? "Vulkan" : "D3D12");
            destroyAllResources();
            createAllResources();
            continue;  // 跳过本帧渲染
        }

        // ── 获取下一帧 ──
        auto frame = swapchain->acquireNextFrame();
        if (frame.needsResize) {
            swapchain->recreate();
            continue;
        }

        // ── 选择命令缓冲（双缓冲交替） ──
        auto& cmd = (frame.frameInFlight == 0) ? *cmd0 : *cmd1;
        cmd.reset();
        cmd.begin();

        // ── 屏障：转换到颜色附件布局（frame.texture 由 swapchain 提供） ──
        rhi::TextureLayout initialLayout = (s_currentBackend == rhi::Backend::Vulkan)
            ? rhi::TextureLayout::Undefined : rhi::TextureLayout::Present;
        cmd.textureBarrier(*frame.texture, initialLayout, rhi::TextureLayout::ColorAttachment);

        // ── 渲染通道 1：清除背景 + 绘制三角形 ──
        {
            rhi::RenderingAttachmentInfo color{};
            color.view    = frame.view.get();
            color.loadOp  = rhi::AttachmentLoadOp::Clear;
            color.storeOp = rhi::AttachmentStoreOp::Store;
            color.clearColor[0] = 0.1f;
            color.clearColor[2] = 0.3f;

            cmd.beginRendering(&color, 1, nullptr, frame.width, frame.height);
            cmd.bindPipelineState(*triRes.pso);
            cmd.bindDescriptorSet(0, *triRes.emptySet);
            cmd.setViewport(0, 0, (float)frame.width, (float)frame.height);
            cmd.setScissor(0, 0, frame.width, frame.height);
            cmd.draw(3, 0, 0);
            cmd.endRendering();
        }

        // ── 渲染通道 2：ImGui 叠加（Load 保留三角形内容） ──
        {
            rhi::RenderingAttachmentInfo color{};
            color.view    = frame.view.get();
            color.loadOp  = rhi::AttachmentLoadOp::Load;  // 保留上一通道内容
            color.storeOp = rhi::AttachmentStoreOp::Store;

            cmd.beginRendering(&color, 1, nullptr, frame.width, frame.height);
            imgui.render(cmd);
            cmd.endRendering();
        }

        // ── 屏障：转换到呈现布局 ──
        cmd.textureBarrier(*frame.texture, rhi::TextureLayout::ColorAttachment,
                           rhi::TextureLayout::Present);

        cmd.end();

        // ── 提交 ──
        // Vulkan 专用：非拥有型 fence 包装器（提交前创建，等待后销毁）
        std::unique_ptr<rhi::RHIFence> inFlightWrapper;
        {
            rhi::SubmitDesc sd;
            sd.commandBuffer = &cmd;

            if (s_currentBackend == rhi::Backend::Vulkan) {
                // Vulkan: 显式同步 (acquire→submit→present)
                sd.waitSemaphore   = frame.imageAvailable.get();
                sd.signalSemaphore = frame.renderFinished.get();
                // 关键：必须 signal 交换链的 inFlightFence
                // 否则 acquireNextFrame 中的 vkWaitForFences 会永久阻塞
                if (frame.inFlightFence) {
                    inFlightWrapper = rhi::VkRHIFence::createNonOwning(
                        static_cast<rhi::VkRHIDevice&>(*rhiDevice),
                        *static_cast<VkFence*>(frame.inFlightFence));
                    sd.signalFence = inFlightWrapper.get();
                }
            }
            rhiDevice->submit(sd);
        }
        // ── 呈现 ──
        if (s_currentBackend == rhi::Backend::Vulkan) {
            swapchain->present(frame);
            // 等待 GPU 完成（inFlightFence 已由 queueSubmit2 signal）
            inFlightWrapper->wait();
        } else {
            // D3D12: swapchain.present() 负责 Present + fence signal
            swapchain->present(frame);
            // 重置描述符堆（每帧线性分配器归零）
            auto& d3dDev = static_cast<rhi::D3D12RHIDevice&>(*rhiDevice);
            d3dDev.resetDescriptorHeap();
            d3dDev.resetSamplerHeap();
            // D3D12: acquireNextFrame 通过内部 fence 等待，无需显式 wait
        }
        ++frameIdx;
    }

    // ── 清理 ──
    std::printf("[triangle] 退出，共 %u 帧\n", frameIdx);
    destroyAllResources();
    return 0;
}
