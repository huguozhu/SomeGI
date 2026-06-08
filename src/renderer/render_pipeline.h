#pragma once
#include "core/vk_common.h"
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

namespace somegi {

// 单个渲染步骤的完整描述
struct RenderStep {
    std::string name;                                    // e.g. "GBuffer", "SSAO", "Lighting"
    std::string phase;                                   // 阶段分组: "PrePass", "AO", "GI", "Shading", "PostProcess", "Overlay"
    bool enabled = true;                                 // 运行时开关，build() 时过滤
    uint32_t timestampSlot = UINT32_MAX;                 // GPU timestamp 槽位 (可选)
    std::function<void(VkCommandBuffer)> record;         // 录制函数

    // 调试/可视化用
    float lastMs = 0.0f;                                 // 上一帧 GPU 耗时
};

// 渲染管线表管理器
// 用法:
//   1. addStep() 按执行顺序注册所有步骤
//   2. setEnabled() 根据运行状态动态开关
//   3. build() 构建执行表 (过滤 disabled)
//   4. execute(cmd) 遍历执行表依次调用 record
class RenderPipeline {
public:
    // 按注册顺序添加步骤
    void addStep(RenderStep step);

    // 根据 enabled 状态构建执行表 (只含启用的步骤)
    void build();

    // 遍历执行表，依次调用每个步骤的 record
    void execute(VkCommandBuffer cmd);

    // 运行时动态开关某个步骤 (按 name 查找)
    void setEnabled(const std::string& name, bool enabled);

    // 获取建好的执行表 (用于 UI 展示/调试)
    const std::vector<RenderStep*>& table() const { return m_execTable; }

    // 清空所有注册 (用于重建)
    void clear();

    // 已注册的步骤总数
    size_t size() const { return m_registry.size(); }

    // 执行表中的步骤数 (build 之后有效)
    size_t execSize() const { return m_execTable.size(); }

private:
    std::vector<RenderStep> m_registry;                  // 所有注册的步骤 (固定顺序)
    std::vector<RenderStep*> m_execTable;                 // build() 后的执行表 (只含启用的)
    std::unordered_map<std::string, size_t> m_index;     // name → registry index
};

} // namespace somegi
