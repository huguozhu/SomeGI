// src/renderer/fg/fg_debug.h
#pragma once
#include "fg_common.h"
#include "fg_compiler.h"
#include <string>
#include <vector>
#include <cstdint>

namespace somegi {
namespace fg {

// ============================================================
// FGDebug: FrameGraph 调试/可视化数据
//
// 由 FrameGraph::compile() 填充，ImGui 面板读取展示。
// ============================================================
struct FGDebug {
    // ---- Pass 信息 ----
    struct PassDebugInfo {
        std::string name;
        FGPassType passType;
        bool culled = false;
        uint32_t execOrder = 0;
        std::vector<std::string> reads;   // 输入资源名
        std::vector<std::string> writes;  // 输出资源名
        std::vector<std::string> deps;    // 前驱 pass 名
        float gpuMs = 0.0f;
    };
    std::vector<PassDebugInfo> passes;

    // ---- 资源信息 ----
    struct ResourceDebugInfo {
        std::string name;
        FGResourceType type;
        bool isImported = false;
        uint32_t firstWritePass = 0;
        uint32_t lastReadPass = 0;
        uint32_t aliasGroup = UINT32_MAX;
        uint32_t sizeBytes = 0;
        VkExtent3D extent{};
    };
    std::vector<ResourceDebugInfo> resources;

    // ---- 别名组信息 ----
    struct AliasGroupDebug {
        uint32_t id = 0;
        uint32_t totalBytes = 0;
        uint32_t wastedBytes = 0;
        std::vector<std::string> members;
    };
    std::vector<AliasGroupDebug> aliasGroups;

    // ---- 开关 ----
    bool showPassList    = true;
    bool showResources   = false;
    bool showAliasGroups = false;
    bool showBarrierLog  = false;
    bool showTimeline    = false;  // 资源布局时间线

    // ---- 资源布局时间线 ----
    // 每个资源的每个 pass 边界处的布局状态
    struct LayoutSnapshot {
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkAccessFlags2 access = 0;
        bool barrierEmitted = false;   // 此 pass 前是否发射了 barrier
    };
    struct ResourceTimeline {
        std::string resourceName;
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::vector<LayoutSnapshot> snapshots;  // [passIndex] = 进入 pass 前的布局
    };
    std::vector<ResourceTimeline> timelines;
    std::vector<std::string> timelinePassNames;  // 列标题

    // 在 FGExecutor::execute() 期间填充
    void recordLayout(FGHandle handle, const char* name, VkFormat fmt,
                      uint32_t passIndex, VkImageLayout layout,
                      VkAccessFlags2 access, bool barrier);
    void finishTimeline(uint32_t passCount);

    // 编译后填充
    void populate(const FGCompiler::CompiledGraph& compiled,
                  const std::vector<struct FGPassNode>& allPasses,
                  const std::vector<struct FGResourceNode>& allResources);

    // 应用上帧 GPU timestamp 数据到 pass 列表
    void applyTimestamps(const std::vector<float>& gpuMs);
};

} // namespace fg
} // namespace somegi
