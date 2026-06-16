// src/renderer/fg/fg_compiler.h
#pragma once
#include "fg_common.h"
#include <vector>
#include <cstdint>

namespace somegi {
namespace fg {

struct FGPassNode;
struct FGResourceNode;

// ============================================================
// FGCompiler: 帧图编译器（声明）
//
// 输入：passes[] + resources[]（由 FrameGraph 收集的声明数据）
// 输出：CompiledGraph（执行计划）
// ============================================================
class FGCompiler {
public:
    struct AliasGroup {
        uint32_t sizeBytes = 0;                     // 该组需要的最大内存
        std::vector<FGResourceNode*> members;        // 成员资源
    };

    struct CompiledGraph {
        std::vector<FGPassNode*>     passOrder;      // 拓扑排序后的执行顺序
        std::vector<FGResourceNode*> resources;       // 所有资源节点
        std::vector<FGPassNode*>     culledPasses;    // 被剔除的 pass
        std::vector<AliasGroup>      aliasGroups;     // 别名分组
    };

    // 编译入口
    CompiledGraph compile(std::vector<FGPassNode*>& passes,
                          std::vector<FGResourceNode*>& resources);

private:
    void cullPasses(std::vector<FGPassNode*>& passes,
                    std::vector<FGResourceNode*>& resources);
    void buildEdges(std::vector<FGPassNode*>& passes,
                    std::vector<FGResourceNode*>& resources);
    void computeLifetimes(std::vector<FGPassNode*>& passes,
                         std::vector<FGResourceNode*>& resources);
    void computeAliasing(std::vector<FGResourceNode*>& resources,
                         std::vector<AliasGroup>& outGroups);
    void topologicalSort(std::vector<FGPassNode*>& passes);
};

} // namespace fg
} // namespace somegi
