// src/renderer/fg/fg_resource_node.h
#pragma once
#include "fg_common.h"
#include "core/image.h"
#include "core/buffer.h"

namespace somegi {
namespace fg {

struct FGPassNode; // 前向声明

// ============================================================
// FGResourceNode: 图内部资源表示
// ============================================================
struct FGResourceNode {
    FGHandle handle;
    FGResourceDesc desc;
    bool isImported = false;  // true=外部导入, false=graph 托管

    // ---- 生命周期（编译后填充，值为 pass topological index） ----
    uint32_t firstWritePass = UINT32_MAX;
    uint32_t lastReadPass  = 0;

    // ---- 别名信息 ----
    uint32_t aliasedGroup = UINT32_MAX;  // 别名组 ID
    uint32_t aliasedOffset = 0;          // 组内偏移字节数

    // ---- 物理资源（execute 阶段由 FGExecutor 分配） ----
    Image*  physicalTexture = nullptr;
    Buffer* physicalBuffer = nullptr;
    VkImage importedImage = VK_NULL_HANDLE;  // 导入纹理的 VkImage（屏障发射用）

    // ---- Barrier 追踪状态（跨 pass 持续更新） ----
    struct BarrierState {
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkAccessFlags2 access = 0;
        FGPassNode* lastWriter = nullptr;
    };
    BarrierState state;
};

} // namespace fg
} // namespace somegi
