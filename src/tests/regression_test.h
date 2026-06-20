// regression_test.h — 截帧回归测试：加载 PNG 对比像素，计算 PSNR，判定通过/失败
#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace somegi {

// 单次对比结果
struct RegressResult {
    bool   passed = false;
    double psnr   = 0.0;       // PSNR (dB)
    double maxError = 0.0;     // 单通道最大误差 [0, 255]
    uint64_t badPixels = 0;    // 超过容限的像素数
    uint64_t totalPixels = 0;
    int refWidth = 0, refHeight = 0;
    int curWidth = 0, curHeight = 0;
};

// 回归测试：加载参考图和当前截图，逐像素对比
// 用法：
//   RegressionTest test;
//   test.setThreshold(40.0);  // 低于 40dB 视为失败
//   RegressResult r = test.compare("ref_gbuffer.png", "captures/gbuffer_0001.png");
//   if (!r.passed) printf("REGRESSION: PSNR=%.2f\n", r.psnr);
class RegressionTest {
public:
    void setThreshold(double db) { m_threshold = db; }
    double threshold() const { return m_threshold; }

    // 设置参考图和输出图的根目录（默认：tests/ref/ 和 screenshots/）
    void setRefDir(const std::string& dir)   { m_refDir = dir; }
    void setOutDir(const std::string& dir)   { m_outDir = dir; }

    // 加载两张 PNG，对比所有像素
    // refName: 参考图路径（相对于 m_refDir，不含 .png 后缀）
    // captureName: 截图路径（相对于 m_outDir，不含 .png 后缀）
    RegressResult compare(const std::string& refName,
                          const std::string& captureName) const;

    // 便捷函数：加载 PNG 到 RGBA 像素数组
    static bool loadPng(const std::string& path,
                        std::vector<uint8_t>& rgba,
                        int& width, int& height);

private:
    double m_threshold = 40.0;      // PSNR 阈值 (dB)
    std::string m_refDir = "tests/ref";
    std::string m_outDir = "screenshots";
};

} // namespace somegi
