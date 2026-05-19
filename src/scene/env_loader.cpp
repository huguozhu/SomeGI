#include "env_loader.h"
#include <stb_image.h>  // STB_IMAGE_IMPLEMENTATION 已在 gltf_loader.cpp 定义
#include <string>

namespace somegi {

bool loadHdrEquirect(const std::filesystem::path& path, EnvCpu& env, std::string& errOut) {
    int w, h, c;
    float* pix = stbi_loadf(path.string().c_str(), &w, &h, &c, 3);
    if (!pix) {
        errOut = std::string("stbi_loadf failed: ") + path.string();
        return false;
    }
    env.width = w; env.height = h;
    env.rgbaF32.resize(size_t(w) * h * 4);
    for (int i = 0; i < w * h; ++i) {
        env.rgbaF32[i * 4 + 0] = pix[i * 3 + 0];
        env.rgbaF32[i * 4 + 1] = pix[i * 3 + 1];
        env.rgbaF32[i * 4 + 2] = pix[i * 3 + 2];
        env.rgbaF32[i * 4 + 3] = 1.0f;
    }
    stbi_image_free(pix);
    errOut.clear();
    return true;
}

void makeFallbackSky(EnvCpu& env) {
    env.width = 512; env.height = 256;
    env.rgbaF32.resize(size_t(env.width) * env.height * 4);
    for (int y = 0; y < env.height; ++y) {
        float t = float(y) / float(env.height - 1);  // 0=top, 1=bottom
        // 顶部偏蓝、地平线偏白、地面暗
        float r, g, b;
        if (t < 0.5f) {
            float k = t * 2.0f;
            r = 0.4f + 0.5f * k;
            g = 0.6f + 0.3f * k;
            b = 1.0f;
        } else {
            float k = (t - 0.5f) * 2.0f;
            r = 0.9f - 0.7f * k;
            g = 0.9f - 0.7f * k;
            b = 1.0f - 0.7f * k;
        }
        for (int x = 0; x < env.width; ++x) {
            int i = (y * env.width + x) * 4;
            env.rgbaF32[i + 0] = r;
            env.rgbaF32[i + 1] = g;
            env.rgbaF32[i + 2] = b;
            env.rgbaF32[i + 3] = 1.0f;
        }
    }
}

}
