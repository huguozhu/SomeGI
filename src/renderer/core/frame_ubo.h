#pragma once
#include <glm/glm.hpp>

// 统一帧常量 UBO 结构 —— CPU 端每帧填充，GPU 端所有 pass 的 set=0 binding=0 共享。
// 需与 shaders/common/shared_types.slang 中的 FrameUniforms 保持字段顺序和布局一致。
// 通过 "frame_ubo.h" 独立头文件解耦，不依赖 ForwardPass / GBufferPass。

namespace somegi {

struct FrameUBO {
    // ── 核心变换 ──
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
    glm::mat4 invViewProj;      // 延迟渲染：从 depth 重建世界坐标
    glm::mat4 prevViewProj;     // SSGI 时序累积：上一帧 viewProj 用于 reproject
    // ── 光照参数 ──
    glm::vec4 cameraPos;
    glm::vec4 sunDir;
    glm::vec4 sunColor_intensity;   // rgb=颜色, w=强度
    glm::vec4 ambient;
    // ── 全局计数 ──
    glm::ivec4 counts;              // x=materialCount, y=iblSpecularMips,
                                    // z=间接光启用 (0/1), w=rsmEnabled (0/1)
    // ── M6 LPV 网格 ──
    glm::ivec4 lpvCounts;           // x=gridResolution, y=lpvEnabled
    glm::vec4 lpvGridMinCell;       // xyz=gridMin (世界坐标), w=cellSize
    // ── M7 VXGI 网格 ──
    glm::ivec4 vxgiCounts;          // x=gridResolution, y=vxgiEnabled, z=mipLevels
    glm::vec4 vxgiGridMinCell;      // xyz=gridMin, w=cellSize
    // ── M8 PRT 参数 + SH 光照投影 ──
    glm::ivec4 prtCounts;           // x=gridResolution, y=prtEnabled
    glm::vec4 prtGridMinCell;
    glm::vec4 prtLightSH_R;         // SH4 系数（l=0,1 共 4 系数 / R 通道）
    glm::vec4 prtLightSH_G;
    glm::vec4 prtLightSH_B;
    glm::vec4 prtLightSH9_R0;       // SH9 扩展：l=2 共 5 系数 / 通道
    glm::vec4 prtLightSH9_R1;
    glm::vec4 prtLightSH9_G0;
    glm::vec4 prtLightSH9_G1;
    glm::vec4 prtLightSH9_B0;
    glm::vec4 prtLightSH9_B1;
    glm::vec4 prtLightSH16_R0;      // SH16 扩展：l=3 共 7 系数 / 通道
    glm::vec4 prtLightSH16_R1;
    glm::vec4 prtLightSH16_G0;
    glm::vec4 prtLightSH16_G1;
    glm::vec4 prtLightSH16_B0;
    glm::vec4 prtLightSH16_B1;
    // ── M11 DDGI probe 网格 ──
    glm::ivec4 ddgiCounts;          // probesX, probesY, probesZ, enabled (0/1/2)
    glm::vec4 ddgiOrigin;
    glm::vec4 ddgiSpacing;
    glm::ivec4 ddgiOctaSizes;       // octaIrr, octaDist
    // ── Lumen-lite ──
    glm::ivec4 lumenCounts;         // x=lumenEnabled (0/1)
};

} // namespace somegi
