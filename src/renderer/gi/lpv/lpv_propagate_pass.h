#pragma once
#include "core/vk_common.h"
#include "renderer/gi/lpv/lpv_grid.h"

// LpvPropagatePass —— M6.1：单次 6-邻居 SH 转移。算法见
// shaders/gi/lpv/lpv_propagate.slang。
//
// 用法：调用方 ping-pong 数次。每次 record(src, dst) 把 src 的 6 邻居
// 累加成 dst 当前 cell 的新 SH；调用方在两次 record 之间手动做 src→dst
// 的 layout / barrier 处理（src 转 SHADER_READ_ONLY，dst 转 GENERAL）。
//
// 一组 set 不够 ping-pong 用 —— 因为 src 和 dst 角色每次互换，描述符内
// 容也要换。本类持有两组 descriptor set：set[0] 绑 grid0→grid1，set[1]
// 绑 grid1→grid0。bindResources 时一次填好。

namespace somegi {
class Device;

class LpvPropagatePass {
public:
    void init(Device& d);
    void destroy();

    // 把两组 ping-pong grid 写入两组 descriptor set；之后 record 用 srcIdx
    // 选哪组（0 = 把 grids[0] 当 src，grids[1] 当 dst；1 = 反过来）。
    void bindResources(Device& d, const LpvGrid& grid0, const LpvGrid& grid1,
                       const Image& gv);

    // 录制一次 propagate dispatch。srcIdx ∈ {0,1}。occlusionAmplifier 是
    // 全局放大系数（UI 可调），1.0 = 标准能量守恒，>1 看起来更亮。
    // gvOcclusionStrength：B.8 GV 遮挡强度，0=禁用，1=标准。
    void record(VkCommandBuffer cmd, int srcIdx, uint32_t gridResolution,
                float occlusionAmp, float gvOcclusionStr);

    int   iterations = 8;             // ImGui 可调
    float occlusionAmplifier = 1.0f;  // 全局放大；>1 看着更亮，<1 衰减
    float gvOcclusionStrength = 1.0f; // B.8 GV 遮挡强度（UI 可调），0=关

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_sets[2]{VK_NULL_HANDLE, VK_NULL_HANDLE};
};

}
