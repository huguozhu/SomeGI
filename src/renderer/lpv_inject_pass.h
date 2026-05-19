#pragma once
#include "core/vk_common.h"
#include "core/image.h"
#include "lpv_grid.h"
#include <glm/glm.hpp>

// LpvInjectPass —— M6.0：把 RSM 当 VPL 数组，dispatch RSM 分辨率，
// 每 RSM texel 投到 LPV 网格对应 cell 的 SH 系数。算法见
// shaders/gi/lpv/lpv_inject.slang。
//
// 调用方负责：
//   - 进入 record 前，RSM 3 张图 in SHADER_READ_ONLY（M5 RsmGeometryPass
//     收尾已是这样）。
//   - record 末尾需要 SHADER_WRITE → SHADER_READ barrier 才能在 propagate
//     里读 grid。
// record 前会先 vkCmdClearColorImage 把 lpvR/G/B 抹 0（避免上帧残留）。

namespace somegi {
class Device;

class LpvInjectPass {
public:
    void init(Device& d, uint32_t rsmSize);
    void destroy();

    // grid 在 init 时不知道；scene 切换 / lpv resolution 改变都要重 bind。
    // 调用者按需调。RSM 3 张图也在这里写入描述符。
    void bindResources(Device& d,
                       const Image& rsmPos, const Image& rsmN, const Image& rsmFlux,
                       const LpvGrid& grid, const Image& gv);

    // 预条件：lpvR/G/B 在 GENERAL（用作 storage write）；RSM 3 张在
    // SHADER_READ_ONLY。grid 内部由本方法先 clearColorImage 再 dispatch。
    void record(VkCommandBuffer cmd, uint32_t gridResolution,
                const glm::vec3& gridMin, float cellSize);

    uint32_t rsmSize() const { return m_rsmSize; }

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    uint32_t m_rsmSize = 0;
    // 缓存当前 bind 的 grid image handle，供 record 内的 vkCmdClearColorImage
    // 使用。bindResources 时刷新。
    VkImage m_lpvR = VK_NULL_HANDLE;
    VkImage m_lpvG = VK_NULL_HANDLE;
    VkImage m_lpvB = VK_NULL_HANDLE;
    VkImage m_gv   = VK_NULL_HANDLE;   // B.8 GV image
};

}
