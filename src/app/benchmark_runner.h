#pragma once
#include <functional>
#include <vector>
#include <string>

namespace somegi {

class FrameRenderer;
class Device;

// 基准测试状态机 —— 遍历 GI×AA×AO 全组合，采集每组合的帧率和 GPU 时间。
// 通过 ApplyFn 回调通知 App 切换配置，App 不直接操纵测试内部状态。
class BenchmarkRunner {
public:
    struct Result { int gi, aa, ao; float fps, gpuMs; };

    // 当测试推进到下一组合时调用此回调
    using ApplyFn = std::function<void(int giIndex, int aaMethod, int aoMethod)>;

    // 开始新测试：重置状态，apply 第一组配置
    void start(ApplyFn apply);
    // 每帧调用：dt 为帧间隔(秒)，gpuMs 为 GPU 耗时(ms)
    void tick(float dt, float gpuMs, ApplyFn apply);
    // 测试是否运行中
    bool running() const { return m_running; }
    // 测试结果（完成后可读）
    const std::vector<Result>& results() const { return m_results; }

private:
    void finish();

    bool m_running    = false;
    bool m_collecting = false;
    float m_timer     = 0.0f;
    int   m_frameCnt  = 0;
    float m_gpuSum    = 0.0f;

    int m_gi = 0, m_aa = 0, m_ao = 0;
    std::vector<Result> m_results;
};

} // namespace somegi
