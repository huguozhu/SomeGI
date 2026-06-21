#pragma once
#include <filesystem>

namespace somegi {

// 着色器/资源目录路径（由 CMake SOMEGI_SHADER_DIR / SOMEGI_ASSET_DIR 定义）
std::filesystem::path shaderDir();
std::filesystem::path assetDir();

} // namespace somegi
