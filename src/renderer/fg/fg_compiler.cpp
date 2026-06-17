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

    // 第一遍：标记 enabled=false 的 pass
    for (auto* p : passes) {
        if (p->culled) continue;
        if (!p->enabled) {
            p->culled = true;
        }
    }

    // ---- 构建资源↔Pass 关系图（O(n+e) 单次遍历） ----
    // writersOf[r] = 写入资源 r 的所有 pass
    std::unordered_map<FGResourceNode*, std::vector<FGPassNode*>> writersOf;
    // consumerCount[r] = 资源 r 的外部消费者数
    //   （不包含 write 该资源的 pass 自身的 read，即 readWrite 不算自消费）
    std::unordered_map<FGResourceNode*, int> consumerCount;

    for (auto* p : passes) {
        if (p->culled) continue;
        for (auto& w : p->writes) {
            if (w.resource) writersOf[w.resource].push_back(p);
        }
        for (auto& r : p->reads) {
            if (!r.resource) continue;
            // 若同一 pass 也 write 此资源（readWrite），不视为外部消费者
            bool isSelfWrite = false;
            for (auto& w : p->writes) {
                if (w.resource == r.resource) { isSelfWrite = true; break; }
            }
            if (!isSelfWrite) {
                consumerCount[r.resource]++;  // operator[] 默认构造 0 再自增
            }
        }
    }

    // ---- 判断 pass 是否有外部消费者 ----
    auto hasAnyConsumer = [&](FGPassNode* p) -> bool {
        for (auto& w : p->writes) {
            if (!w.resource) continue;
            if (w.resource->isImported) return true;
            auto it = consumerCount.find(w.resource);
            if (it != consumerCount.end() && it->second > 0) return true;
        }
        return false;
    };

    // ---- 初始候选队列：所有无消费者的非 manual pass ----
    std::queue<FGPassNode*> workQueue;
    for (auto* p : passes) {
        if (p->culled || p->usesManualBarriers) continue;
        if (!hasAnyConsumer(p)) {
            workQueue.push(p);
        }
    }

    // ---- 传播剔除（work-queue 驱动，每 pass 最多处理一次） ----
    while (!workQueue.empty()) {
        auto* p = workQueue.front();
        workQueue.pop();
        if (p->culled) continue;  // 可能被多个资源触发重复入队

        p->culled = true;
        std::printf("[FGCompiler] culled: %s (no consumers)\n", p->name.c_str());

        // 此 pass 读取的资源失去一个外部消费者
        for (auto& r : p->reads) {
            if (!r.resource) continue;
            // readWrite 的同资源读不算外部消费者，已在计数时排除
            bool isSelfWrite = false;
            for (auto& w : p->writes) {
                if (w.resource == r.resource) { isSelfWrite = true; break; }
            }
            if (isSelfWrite) continue;

            auto it = consumerCount.find(r.resource);
            if (it == consumerCount.end()) continue;
            if (it->second > 0) it->second--;

            // 若此资源不再有外部消费者，重新检查其写入者
            if (it->second == 0 && !r.resource->isImported) {
                auto wit = writersOf.find(r.resource);
                if (wit != writersOf.end()) {
                    for (auto* writer : wit->second) {
                        if (writer->culled || writer->usesManualBarriers) continue;
                        if (!hasAnyConsumer(writer)) {
                            workQueue.push(writer);
                        }
                    }
                }
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
                    it->second->successors.push_back(p);  // 反向建边
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
                    it->second->successors.push_back(p);  // 反向建边
                }
            }

            lastWriter[res] = p;
        }
    }

    // 清理 successors 中的重复项（可能因同一资源被多次 read+write 而产生）
    for (auto* p : passes) {
        if (p->culled) continue;
        std::sort(p->successors.begin(), p->successors.end());
        p->successors.erase(std::unique(p->successors.begin(), p->successors.end()), p->successors.end());
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

    // 贪心别名分配 — 两阶段：
    //   阶段1: 分配资源到组，暂存组号到局部数组
    //   阶段2: 回写 aliasedGroup（避免 outGroups push_back 扩容导致指针/索引失效）
    std::vector<uint32_t> assignedGroup(managed.size(), UINT32_MAX);

    for (size_t i = 0; i < managed.size(); ++i) {
        auto& mr = managed[i];
        bool placed = false;

        for (size_t g = 0; g < outGroups.size(); ++g) {
            auto& group = outGroups[g];
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
                assignedGroup[i] = (uint32_t)g;
                placed = true;
                break;
            }
        }

        if (!placed) {
            AliasGroup newGroup;
            newGroup.sizeBytes = mr.sizeBytes;
            newGroup.members.push_back(mr.resource);
            assignedGroup[i] = (uint32_t)outGroups.size();
            outGroups.push_back(std::move(newGroup));
        }
    }

    // 阶段2: 回写 aliasedGroup（outGroups 不再变化，索引安全）
    for (size_t i = 0; i < managed.size(); ++i) {
        managed[i].resource->aliasedGroup = assignedGroup[i];
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
    // 用 predecessors 数量作为入度（已去重且在 buildEdges 中维护）
    std::queue<FGPassNode*> queue;
    uint32_t activeCount = 0;

    for (auto* p : passes) {
        if (p->culled) continue;
        activeCount++;
        uint32_t deg = 0;
        for (auto* pred : p->predecessors)
            if (!pred->culled) deg++;
        if (deg == 0) {
            queue.push(p);
        } else {
            // 临时用 topologicalIndex 存放入度（排序时会覆盖）
            p->topologicalIndex = deg;
        }
    }

    uint32_t order = 0;
    while (!queue.empty()) {
        auto* node = queue.front();
        queue.pop();
        node->topologicalIndex = order++;

        // O(1) 遍历后继，而非 O(n) 扫描全部 pass
        for (auto* succ : node->successors) {
            if (succ->culled) continue;
            uint32_t& deg = succ->topologicalIndex;  // 入度暂存于此
            if (--deg == 0) {
                queue.push(succ);
            }
        }
    }

    if (order < activeCount) {
        std::fprintf(stderr, "[FGCompiler] ERROR: cycle detected in pass graph! "
                             "Sorted %u/%u passes.\n", order, activeCount);
    }
}

} // namespace fg
} // namespace somegi
