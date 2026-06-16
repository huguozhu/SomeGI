// src/renderer/fg/fg_compiler.cpp
#include "fg_compiler.h"
#include "fg_pass_node.h"
#include "fg_resource_node.h"
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <cstdio>

namespace somegi {
namespace fg {

FGCompiler::CompiledGraph FGCompiler::compile(
    std::vector<FGPassNode*>& passes,
    std::vector<FGResourceNode*>& resources) {

    CompiledGraph result;
    result.resources = resources;

    cullPasses(passes, resources);
    buildEdges(passes, resources);
    topologicalSort(passes);
    computeLifetimes(passes, resources);
    computeAliasing(resources, result.aliasGroups);

    for (auto* p : passes) {
        if (!p->culled) result.passOrder.push_back(p);
        else result.culledPasses.push_back(p);
    }

    // 按拓扑序排列 passOrder，确保执行顺序与依赖顺序一致
    std::sort(result.passOrder.begin(), result.passOrder.end(),
        [](const FGPassNode* a, const FGPassNode* b) {
            return a->topologicalIndex < b->topologicalIndex;
        });

    return result;
}

// ============================================================
// 步骤 1: 剔除
// ============================================================

void FGCompiler::cullPasses(std::vector<FGPassNode*>& passes,
                             std::vector<FGResourceNode*>& resources) {
    (void)resources;
    // 第一遍：标记 enabled=false
    for (auto* p : passes) {
        if (p->culled) continue;
        if (!p->enabled) {
            p->culled = true;
        }
    }

    // 传播剔除：迭代直到稳定
    bool changed = true;
    while (changed) {
        changed = false;

        for (auto* p : passes) {
            if (p->culled) continue;

            // 手动屏障 pass 不参与 cull：它们管理自己的私有资源，
            // FrameGraph 不追踪这些资源的消费者关系
            if (p->usesManualBarriers) continue;

            bool anyConsumer = false;
            for (auto& w : p->writes) {
                if (!w.resource || w.resource->isImported) {
                    anyConsumer = true;
                    break;
                }
                for (auto* other : passes) {
                    if (other == p || other->culled) continue;
                    for (auto& r : other->reads) {
                        if (r.resource == w.resource) {
                            anyConsumer = true;
                            break;
                        }
                    }
                    if (anyConsumer) break;
                }
                if (anyConsumer) break;
            }

            if (!anyConsumer) {
                p->culled = true;
                changed = true;
                std::printf("[FGCompiler] culled: %s (no consumers)\n", p->name.c_str());
            }
        }
    }
}

// ============================================================
// 步骤 2: 构建依赖边
// ============================================================

void FGCompiler::buildEdges(std::vector<FGPassNode*>& passes,
                             std::vector<FGResourceNode*>& resources) {
    (void)resources;
    std::unordered_map<FGResourceNode*, FGPassNode*> lastWriter;

    for (auto* p : passes) {
        if (p->culled) continue;

        for (auto& ref : p->reads) {
            auto* res = ref.resource;
            if (!res) continue;

            auto it = lastWriter.find(res);
            if (it != lastWriter.end() && it->second != p) {
                if (std::find(p->predecessors.begin(), p->predecessors.end(), it->second)
                    == p->predecessors.end()) {
                    p->predecessors.push_back(it->second);
                }
            }
        }

        for (auto& ref : p->writes) {
            auto* res = ref.resource;
            if (!res) continue;

            auto it = lastWriter.find(res);
            if (it != lastWriter.end() && it->second != p) {
                if (std::find(p->predecessors.begin(), p->predecessors.end(), it->second)
                    == p->predecessors.end()) {
                    p->predecessors.push_back(it->second);
                }
            }

            lastWriter[res] = p;
        }
    }
}

// ============================================================
// 步骤 4: 计算资源寿命
// ============================================================

void FGCompiler::computeLifetimes(std::vector<FGPassNode*>& passes,
                                   std::vector<FGResourceNode*>& resources) {
    for (auto* res : resources) {
        if (!res) continue;
        res->firstWritePass = UINT32_MAX;
        res->lastReadPass = 0;
    }

    for (auto* pass : passes) {
        if (pass->culled) continue;
        uint32_t idx = pass->topologicalIndex;

        for (auto& ref : pass->writes) {
            if (!ref.resource) continue;
            if (idx < ref.resource->firstWritePass)
                ref.resource->firstWritePass = idx;
            if (idx > ref.resource->lastReadPass)
                ref.resource->lastReadPass = idx;
        }

        for (auto& ref : pass->reads) {
            if (!ref.resource) continue;
            if (idx > ref.resource->lastReadPass)
                ref.resource->lastReadPass = idx;
        }
    }
}

// ============================================================
// 步骤 5: 别名分析
// ============================================================

void FGCompiler::computeAliasing(std::vector<FGResourceNode*>& resources,
                                  std::vector<AliasGroup>& outGroups) {
    struct ManagedRes {
        FGResourceNode* resource;
        uint32_t sizeBytes;
    };
    std::vector<ManagedRes> managed;

    for (auto* res : resources) {
        if (!res || res->isImported) continue;

        uint32_t size = 4 * 1024 * 1024;
        if (res->desc.type == FGResourceType::Texture) {
            auto& t = res->desc.texture;
            size = t.extent.width * t.extent.height * t.extent.depth * 4;
        } else {
            size = (uint32_t)res->desc.buffer.size;
        }

        managed.push_back({res, size});
    }

    // 按大小降序排列
    std::sort(managed.begin(), managed.end(),
        [](const ManagedRes& a, const ManagedRes& b) {
            return a.sizeBytes > b.sizeBytes;
        });

    // 贪心别名分配
    for (auto& mr : managed) {
        bool placed = false;

        for (auto& group : outGroups) {
            bool overlap = false;
            for (auto* member : group.members) {
                if (!(mr.resource->lastReadPass < member->firstWritePass ||
                      mr.resource->firstWritePass > member->lastReadPass)) {
                    overlap = true;
                    break;
                }
            }

            if (!overlap) {
                group.members.push_back(mr.resource);
                if (mr.sizeBytes > group.sizeBytes) {
                    group.sizeBytes = mr.sizeBytes;
                }
                mr.resource->aliasedGroup = (uint32_t)(&group - outGroups.data());
                placed = true;
                break;
            }
        }

        if (!placed) {
            AliasGroup newGroup;
            newGroup.sizeBytes = mr.sizeBytes;
            newGroup.members.push_back(mr.resource);
            mr.resource->aliasedGroup = (uint32_t)outGroups.size();
            outGroups.push_back(std::move(newGroup));
        }
    }

    if (!outGroups.empty()) {
        uint32_t totalSaved = 0;
        for (auto& g : outGroups) {
            for (size_t i = 1; i < g.members.size(); ++i) {
                totalSaved += g.sizeBytes;
            }
        }
        std::printf("[FGCompiler] alias groups: %zu, estimated memory saved: %u bytes\n",
                    outGroups.size(), totalSaved);
    }
}

// ============================================================
// 步骤 3: 拓扑排序 (Kahn BFS)
// ============================================================

void FGCompiler::topologicalSort(std::vector<FGPassNode*>& passes) {
    std::unordered_map<FGPassNode*, uint32_t> inDegree;
    for (auto* p : passes) {
        if (p->culled) continue;
        inDegree[p] = 0;
    }
    for (auto* p : passes) {
        if (p->culled) continue;
        for (auto* pred : p->predecessors) {
            if (!pred->culled) {
                inDegree[p]++;
            }
        }
    }

    std::queue<FGPassNode*> queue;
    for (auto& [node, deg] : inDegree) {
        if (deg == 0) queue.push(node);
    }

    uint32_t order = 0;
    while (!queue.empty()) {
        auto* node = queue.front();
        queue.pop();
        node->topologicalIndex = order++;

        for (auto* other : passes) {
            if (other->culled) continue;
            for (auto* pred : other->predecessors) {
                if (pred == node) {
                    inDegree[other]--;
                    if (inDegree[other] == 0) {
                        queue.push(other);
                    }
                }
            }
        }
    }

    uint32_t activeCount = 0;
    for (auto* p : passes) {
        if (!p->culled) activeCount++;
    }
    if (order < activeCount) {
        std::fprintf(stderr, "[FGCompiler] ERROR: cycle detected in pass graph! "
                             "Sorted %u/%u passes.\n", order, activeCount);
    }
}

} // namespace fg
} // namespace somegi
