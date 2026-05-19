#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <cstdint>

namespace somegi {

struct EnvCpu {
    int width = 0;
    int height = 0;
    std::vector<float> rgbaF32;  // 4 通道，HDR linear（R,G,B,1）
};

// 失败：返回 false，errOut 描述。成功：env 填好，errOut 空字符串。
bool loadHdrEquirect(const std::filesystem::path& path, EnvCpu& env, std::string& errOut);

// 找不到文件时使用：生成一个简单的天空梯度（512×256），让 baker 也能跑通。
void makeFallbackSky(EnvCpu& env);

}
