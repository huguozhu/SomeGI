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

        somegi::App app;
        app.setScreenshotConfig(
            g_cliConfig.captureInterval,
            g_cliConfig.captureFrame,
            g_cliConfig.captureDir);
        if (g_cliConfig.shadowMethod >= 0)
            app.setInitialShadowMethod(g_cliConfig.shadowMethod);
        app.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
