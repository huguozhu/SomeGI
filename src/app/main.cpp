#include "app.h"
#include "core/screenshot.h"
#include "core/shader.h"
#include "rhi/base/device.h"
#include "rhi/base/swapchain.h"
#include "rhi/base/command_buffer.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>

namespace {

// CLI 参数解析结果
struct CliConfig {
    int captureInterval = 0;   // --capture-interval N
    int captureFrame = -1;     // --capture-frame N
    const char* captureDir = "screenshots";
    int shadowMethod = -1;     // --shadow-method N（-1 表示使用默认值）
    bool exitAfterCapture = false;  // --exit-after-capture
    // 回归测试
    bool captureRef = false;          // --capture-ref：生成参考图
    bool captureCompare = false;      // --capture-compare：截帧对比
    double refThreshold = 40.0;       // --ref-threshold N
    // 后端选择
    const char* backend = "vulkan";   // --backend vulkan|d3d12
};

CliConfig parseCli(int argc, char** argv) {
    CliConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--capture-interval") == 0 && i + 1 < argc) {
            cfg.captureInterval = std::atoi(argv[++i]);
            if (cfg.captureInterval < 1) cfg.captureInterval = 0;
        } else if (std::strcmp(argv[i], "--capture-frame") == 0 && i + 1 < argc) {
            cfg.captureFrame = std::atoi(argv[++i]);
            if (cfg.captureFrame < 0) cfg.captureFrame = -1;
        } else if (std::strcmp(argv[i], "--capture-dir") == 0 && i + 1 < argc) {
            cfg.captureDir = argv[++i];
        } else if (std::strcmp(argv[i], "--shadow-method") == 0 && i + 1 < argc) {
            cfg.shadowMethod = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--exit-after-capture") == 0) {
            cfg.exitAfterCapture = true;
        } else if (std::strcmp(argv[i], "--capture-ref") == 0) {
            cfg.captureRef = true;
        } else if (std::strcmp(argv[i], "--capture-compare") == 0) {
            cfg.captureCompare = true;
        } else if (std::strcmp(argv[i], "--ref-threshold") == 0 && i + 1 < argc) {
            cfg.refThreshold = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            cfg.backend = argv[++i];
        } else {
            // 兜底：旧的 gltf 路径参数（已废弃），忽略
        }
    }
    return cfg;
}

} // anonymous namespace

// 全局 CLI 配置，供 App::run() 在构造后读取
static CliConfig g_cliConfig;

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    g_cliConfig = parseCli(argc, argv);

    try {
        std::error_code ec;
        std::filesystem::current_path(SOMEGI_PROJECT_DIR, ec);
        if (ec) {
            std::fprintf(stderr, "warn: failed to chdir to project root: %s\n", ec.message().c_str());
        }

        // 打印截图配置（如果启用）
        if (g_cliConfig.captureInterval > 0) {
            std::printf("[init] auto screenshot every %d frame(s) → %s/\n",
                        g_cliConfig.captureInterval, g_cliConfig.captureDir);
        }
        if (g_cliConfig.captureFrame >= 0) {
            std::printf("[init] capture frame %d → %s/\n",
                        g_cliConfig.captureFrame, g_cliConfig.captureDir);
        }

        std::printf("CWD : %s\n", std::filesystem::current_path().string().c_str());

        // D3D12 测试路径：clear + present 循环
        if (std::strcmp(g_cliConfig.backend, "d3d12") == 0) {
            std::printf("[init] D3D12 backend selected — running clear test\n");
            // 创建独立 GLFW 窗口（不使用 Vulkan）
            glfwInit();
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            GLFWwindow* win = glfwCreateWindow(800, 600, "SomeGI D3D12", nullptr, nullptr);
            if (!win) { std::fprintf(stderr, "GLFW window creation failed\n"); return 1; }

            HWND hwnd = glfwGetWin32Window(win);
            auto d3d12Device = somegi::rhi::RHIDevice::create(somegi::rhi::Backend::D3D12, hwnd, false);
            auto swapchain = d3d12Device->createSwapchain(hwnd, 800, 600);
            auto cmdPool = d3d12Device->createCommandPool();
            std::unique_ptr<somegi::rhi::RHICommandBuffer> cmdBuf(cmdPool->allocateRaw());

            std::printf("[d3d12] clear loop: press ESC or close window to exit\n");
            while (!glfwWindowShouldClose(win)) {
                glfwPollEvents();
                if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                    glfwSetWindowShouldClose(win, GLFW_TRUE);

                auto frame = swapchain->acquireNextFrame();
                if (frame.needsResize) continue;

                cmdBuf->begin();

                // 清除 back buffer（蓝色背景）
                somegi::rhi::RenderingAttachmentInfo colorAttach{};
                colorAttach.view = frame.view.get();
                colorAttach.loadOp = somegi::rhi::AttachmentLoadOp::Clear;
                colorAttach.storeOp = somegi::rhi::AttachmentStoreOp::Store;
                colorAttach.clearColor[0] = 0.1f;
                colorAttach.clearColor[1] = 0.2f;
                colorAttach.clearColor[2] = 0.4f;
                colorAttach.clearColor[3] = 1.0f;

                cmdBuf->beginRendering(&colorAttach, 1, nullptr, frame.width, frame.height);
                cmdBuf->endRendering();

                cmdBuf->end();

                somegi::rhi::SubmitDesc sd{};
                sd.commandBuffer = cmdBuf.get();
                d3d12Device->submit(sd);

                swapchain->present(frame);
            }

            d3d12Device->waitIdle();
            glfwDestroyWindow(win);
            glfwTerminate();
            std::printf("[d3d12] test complete\n");
            return 0;
        }

        somegi::App app;
        app.setBackend(g_cliConfig.backend);
        app.setScreenshotConfig(
            g_cliConfig.captureInterval,
            g_cliConfig.captureFrame,
            g_cliConfig.captureDir);
        if (g_cliConfig.shadowMethod >= 0)
            app.setInitialShadowMethod(g_cliConfig.shadowMethod);
        if (g_cliConfig.exitAfterCapture)
            app.setExitAfterCapture(true);
        if (g_cliConfig.captureRef)
            app.setCaptureRefMode(true);
        if (g_cliConfig.captureCompare)
            app.setCaptureCompareMode(true, g_cliConfig.refThreshold);
        app.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
