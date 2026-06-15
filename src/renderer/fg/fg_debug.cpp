// src/renderer/fg/fg_debug.cpp
#include "fg_debug.h"
#include "fg_compiler.h"
#include "fg_pass_node.h"
#include "fg_resource_node.h"
#include <cstdio>

namespace somegi {
namespace fg {

static uint32_t estimateSizeBytes(const FGResourceDesc& desc) {
    if (desc.type == FGResourceType::Texture) {
        const auto& t = desc.texture;
        uint32_t pixelSize = 4;  // 默认 RGBA8
        if (t.format == VK_FORMAT_R16G16B16A16_SFLOAT) pixelSize = 8;
        else if (t.format == VK_FORMAT_R32_SFLOAT) pixelSize = 4;
        else if (t.format == VK_FORMAT_D32_SFLOAT) pixelSize = 4;
        return t.extent.width * t.extent.height * t.extent.depth * pixelSize;
    } else {
        return (uint32_t)desc.buffer.size;
    }
}

void FGDebug::populate(const FGCompiler::CompiledGraph& compiled,
                        const std::vector<FGPassNode>& passes,
                        const std::vector<FGResourceNode>& resources) {
    // 清空上帧数据
    this->passes.clear();
    this->resources.clear();
    this->aliasGroups.clear();

    // Pass 列表
    for (auto* p : compiled.passOrder) {
        if (!p) continue;
        PassDebugInfo pi;
        pi.name = p->name;
        pi.passType = p->passType;
        pi.culled = p->culled;
        pi.execOrder = p->topologicalIndex;

        for (auto& r : p->reads) {
            if (r.resource && r.resource->desc.debugName)
                pi.reads.push_back(r.resource->desc.debugName);
        }
        for (auto& w : p->writes) {
            if (w.resource && w.resource->desc.debugName)
                pi.writes.push_back(w.resource->desc.debugName);
        }
        for (auto* pred : p->predecessors)
            pi.deps.push_back(pred->name);

        this->passes.push_back(std::move(pi));
    }

    // 资源列表
    for (auto* r : compiled.resources) {
        if (!r) continue;
        ResourceDebugInfo ri;
        ri.name = r->desc.debugName ? r->desc.debugName : "?";
        ri.type = r->desc.type;
        ri.isImported = r->isImported;
        ri.firstWritePass = r->firstWritePass;
        ri.lastReadPass = r->lastReadPass;
        ri.aliasGroup = r->aliasedGroup;
        ri.sizeBytes = estimateSizeBytes(r->desc);
        if (r->desc.type == FGResourceType::Texture)
            ri.extent = r->desc.texture.extent;
        this->resources.push_back(std::move(ri));
    }

    // 别名组列表
    for (auto& ag : compiled.aliasGroups) {
        AliasGroupDebug agd;
        agd.id = this->aliasGroups.size();
        agd.totalBytes = ag.sizeBytes;
        uint32_t maxSize = 0;
        for (auto* m : ag.members) {
            if (m && m->desc.debugName)
                agd.members.push_back(m->desc.debugName);
            uint32_t s = m ? estimateSizeBytes(m->desc) : 0;
            if (s > maxSize) maxSize = s;
        }
        agd.wastedBytes = ag.sizeBytes - maxSize;
        this->aliasGroups.push_back(std::move(agd));
    }
}

} // namespace fg
} // namespace somegi
