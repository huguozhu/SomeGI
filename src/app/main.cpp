#include "app.h"
#include "core/shader.h"
#include <cstdio>
#include <exception>
#include <filesystem>

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    try {
        std::error_code ec;
        std::filesystem::current_path(SOMEGI_PROJECT_DIR, ec);
        if (ec) {
            std::fprintf(stderr, "warn: failed to chdir to project root: %s\n", ec.message().c_str());
        }
        if (argc >= 2) {
            std::fprintf(stderr, "warn: CLI gltf path '%s' is no longer used; switch scenes via the ImGui dropdown.\n", argv[1]);
        }
        std::printf("CWD : %s\n", std::filesystem::current_path().string().c_str());

        somegi::App app;
        app.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
