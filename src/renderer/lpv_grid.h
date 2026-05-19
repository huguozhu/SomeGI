#pragma once
#include "core/image.h"

// LPV 体素网格 —— 一阶 SH（4 系数）逐通道存到 3 张 RGBA16F 3D image：
//   lpvR : 红通道的 4 个 SH 系数（c00, c1-1, c10, c11）
//   lpvG : 绿通道
//   lpvB : 蓝通道
// 一组 grid 总显存 ≈ 3 × 32³ × 8B = 768 KB（RGBA16F = 8B）；
// ping-pong 两组 ≈ 1.5 MB。
//
// LpvResources 持有两组 grid，propagate 阶段每次迭代 src→dst 后 swap。
// 调用方约定：current() 是上次 propagate 的最终结果（lighting 评估时读这个）；
// next() 是即将被写的目标。

namespace somegi {
class Device;

struct LpvGrid {
    Image lpvR;
    Image lpvG;
    Image lpvB;

    void create(Device& d, uint32_t resolution);
    void destroy();
};

class LpvResources {
public:
    void create(Device& d, uint32_t resolution);
    void destroy();

    LpvGrid& current()       { return m_grids[m_curIdx]; }
    const LpvGrid& current() const { return m_grids[m_curIdx]; }
    LpvGrid& next()          { return m_grids[m_curIdx ^ 1]; }
    int curIdx() const { return (int)m_curIdx; }
    void swap() { m_curIdx ^= 1; }

    // B.8 Geometry Volume：每帧 inject 写入，propagate 读，存几何遮挡 SH
    // (单色 4 系数)。不参与 ping-pong，每帧 inject 阶段重 build。
    const Image& gv() const { return m_gv; }
    Image& gv()             { return m_gv; }

    uint32_t resolution() const { return m_resolution; }

private:
    LpvGrid m_grids[2];
    Image m_gv;
    uint32_t m_curIdx = 0;
    uint32_t m_resolution = 0;
};

}
