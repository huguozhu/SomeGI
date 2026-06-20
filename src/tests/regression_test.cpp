// regression_test.cpp — PNG 加载 + 像素级 PSNR 对比
#include "regression_test.h"
// STB_IMAGE_IMPLEMENTATION 已在 gltf_loader.cpp 定义，此处仅引用头文件
#include <stb/stb_image.h>

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <filesystem>

namespace somegi {

bool RegressionTest::loadPng(const std::string& path,
                              std::vector<uint8_t>& rgba,
                              int& width, int& height) {
    int channels = 0;
    stbi_uc* data = stbi_load(path.c_str(), &width, &height, &channels, 4); // 强制 RGBA
    if (!data) {
        std::fprintf(stderr, "[regress] stbi_load failed: %s\n", path.c_str());
        return false;
    }
    size_t size = static_cast<size_t>(width) * height * 4;
    rgba.assign(data, data + size);
    stbi_image_free(data);
    return true;
}

RegressResult RegressionTest::compare(const std::string& refName,
                                       const std::string& captureName) const {
    RegressResult r{};

    std::string refPath = m_refDir + "/" + refName + ".png";
    std::string capPath = m_outDir + "/" + captureName + ".png";

    std::vector<uint8_t> refPixels, capPixels;
    if (!loadPng(refPath, refPixels, r.refWidth, r.refHeight)) {
        std::fprintf(stderr, "[regress] failed to load reference: %s\n", refPath.c_str());
        return r;
    }
    if (!loadPng(capPath, capPixels, r.curWidth, r.curHeight)) {
        std::fprintf(stderr, "[regress] failed to load capture: %s\n", capPath.c_str());
        return r;
    }

    // 尺寸必须一致
    if (r.refWidth != r.curWidth || r.refHeight != r.curHeight) {
        std::fprintf(stderr, "[regress] size mismatch: ref=%dx%d cap=%dx%d\n",
                     r.refWidth, r.refHeight, r.curWidth, r.curHeight);
        return r;
    }

    r.totalPixels = static_cast<uint64_t>(r.refWidth) * r.refHeight;
    r.badPixels = 0;
    r.maxError = 0.0;
    double mse = 0.0;

    for (uint64_t i = 0; i < r.totalPixels; ++i) {
        for (int c = 0; c < 4; ++c) {
            double diff = static_cast<double>(refPixels[i * 4 + c]) -
                          static_cast<double>(capPixels[i * 4 + c]);
            mse += diff * diff;
            double absErr = std::abs(diff);
            if (absErr > r.maxError) r.maxError = absErr;
        }
    }

    mse /= (r.totalPixels * 4); // 4 通道平均
    if (mse > 0.0) {
        r.psnr = 20.0 * std::log10(255.0 / std::sqrt(mse));
    } else {
        r.psnr = 999.0; // 完全相同
    }

    // 统计差异超过 1 个色阶的像素
    double pixelMse = 0.0;
    for (uint64_t i = 0; i < r.totalPixels; ++i) {
        pixelMse = 0.0;
        for (int c = 0; c < 4; ++c) {
            double diff = static_cast<double>(refPixels[i * 4 + c]) -
                          static_cast<double>(capPixels[i * 4 + c]);
            pixelMse += diff * diff;
        }
        if (std::sqrt(pixelMse / 4.0) > 1.0) {
            ++r.badPixels;
        }
    }

    r.passed = (r.psnr >= m_threshold);
    std::printf("[regress] %s vs %s: PSNR=%.2fdB maxErr=%.1f bad=%llu/%llu %s\n",
                refName.c_str(), captureName.c_str(),
                r.psnr, r.maxError,
                (unsigned long long)r.badPixels, (unsigned long long)r.totalPixels,
                r.passed ? "PASS" : "FAIL");
    return r;
}

} // namespace somegi
