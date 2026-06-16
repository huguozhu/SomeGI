// src/renderer/fg/fg_pass_node.h
#pragma once
#include "fg_common.h"
#include <string>
#include <vector>
#include <functional>

namespace somegi {
namespace fg {

struct FGResourceNode;
class FGResources;

// ============================================================
// FGPassNode: 图内部 Pass 表示
// ============================================================
struct FGPassNode {
    std::string name;
    FGPassType passType = FGPassType::Compute;  // Pass 类型（决定 pipeline stage）
    bool enabled = true;
    bool culled = false;                        // 编译后被剔除

    // ---- 资源依赖边 ----
    struct ResourceRef {
        FGHandle handle;
        FGResourceNode* resource = nullptr;  // 编译后指向实际资源节点
        VkAccessFlags2 access = 0;
        VkPipelineStageFlags2 stages = 0;
        VkImageLayout requiredLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    };
    std::vector<ResourceRef> reads;   // 输入资源
    std::vector<ResourceRef> writes;  // 输出资源

    // ---- 执行回调 ----
    // execute 期调用：void(VkCommandBuffer cmd, const FGResources& resources)
    std::function<void(VkCommandBuffer, const FGResources&)> execute;

    // ---- Barrier 模式 ----
    // true: 此 pass 内部自己管理 barrier，FrameGraph 不为它插入 auto-barrier
    // false: 使用 FrameGraph 自动 barrier（默认）
    bool usesManualBarriers = false;

    // ---- 编译后填充 ----
    uint32_t topologicalIndex = 0;                     // 拓扑排序位置
    std::vector<FGPassNode*> predecessors;             // 直接前驱 pass
};

} // namespace fg
} // namespace somegi
