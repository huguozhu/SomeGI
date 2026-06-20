#include "app.h"
#include "core/screenshot.h"
#include "core/shader.h"
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
    const char* backend = "d3d12";   // --backend vulkan|d3d12
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

        // 根据后端选择创建 App（D3D12 跳过 Vulkan 初始化）
        std::unique_ptr<somegi::App> app;
        if (std::strcmp(g_cliConfig.backend, "d3d12") == 0) {
            app.reset(new somegi::App(somegi::App::ForD3D12{}));
        } else {
            app.reset(new somegi::App());
        }
        app->setBackend(g_cliConfig.backend);
        app->setScreenshotConfig(
            g_cliConfig.captureInterval,
            g_cliConfig.captureFrame,
            g_cliConfig.captureDir);
        if (g_cliConfig.shadowMethod >= 0)
            app->setInitialShadowMethod(g_cliConfig.shadowMethod);
        if (g_cliConfig.exitAfterCapture)
            app->setExitAfterCapture(true);
        if (g_cliConfig.captureRef)
            app->setCaptureRefMode(true);
        if (g_cliConfig.captureCompare)
            app->setCaptureCompareMode(true, g_cliConfig.refThreshold);
        app->run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
