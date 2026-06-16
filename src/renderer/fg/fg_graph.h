// src/renderer/fg/fg_graph.h
#pragma once
#include "fg_common.h"
#include "fg_pass_node.h"
#include "fg_resource_node.h"
#include "fg_builder.h"
#include "fg_resources.h"
#include "fg_compiler.h"
#include "fg_executor.h"
#include "fg_debug.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>

namespace somegi {

class Device;

namespace fg {

// ============================================================
// FrameGraph: 顶层 API
//
// 使用流程：
//   1. importTexture() / createTexture() — 声明资源
//   2. addPass() — 声明 pass 及其资源依赖
//   3. compile() — 构建 DAG / 剔除 / 别名分析 / 拓扑排序
//   4. execute(cmd) — 分配物理资源 / 插入 barrier / 执行 pass
//   5. reset() — 每帧结束时清理图数据
// ============================================================
class FrameGraph {
public:
    FrameGraph();
    ~FrameGraph();

    // ---- 初始化 ----
    void init(Device& device);

    // ---- 资源声明 ----
    // 导入外部已分配的资源（跨帧持久，不参与 aliasing）
    FGHandle importTexture(const char* name,
                           VkImage image,
                           const FGTextureDesc& desc,
                           VkImageLayout initialLayout);

    // 声明托管临时纹理（帧内生命周期，参与 aliasing）
    FGHandle createTexture(const char* name, const FGTextureDesc& desc);

    // 声明托管临时 Buffer（帧内生命周期，参与 aliasing）
    FGHandle createBuffer(const char* name, const FGBufferDesc& desc);

    // ---- Pass 声明 ----
    // setup: void(FGBuilder&) — 声明资源依赖和执行回调
    void addPass(const char* name, std::function<void(FGBuilder&)> setup);

    // ---- 编译 ----
    // 调用 FGCompiler::compile()，结果存入 m_compiled
    void compile();

    // ---- 执行 ----
    // 调用 FGExecutor::execute()，遍历编译后的 pass 并执行
    void execute(VkCommandBuffer cmd);

    // ---- 查询（execute 期使用） ----
    VkImageView getTextureView(FGHandle handle,
                               uint32_t mip = 0,
                               uint32_t layer = 0) const;

    VkBuffer getBuffer(FGHandle handle,
                       VkDeviceSize* outOffset = nullptr) const;

    // ---- 帧管理 ----
    // 每帧开始时调用：清理 pass/resource 节点和视图缓存
    void reset();

    // ---- Barrier 控制 ----
    // 启用/禁用自动 Barrier（默认关闭，pass 内部管理 barrier）
    void setAutoBarriers(bool enabled) { m_executor.setAutoBarriers(enabled); }
    bool autoBarriers() const { return m_executor.autoBarriers(); }

    // ---- 调试 ----
    const FGCompiler::CompiledGraph& compiledGraph() const { return m_compiled; }
    FGDebug& debug() { return m_debug; }

private:
    friend class FGBuilder;

    Device* m_device = nullptr;
    uint64_t m_frameIndex = 0;

    // 用户声明的节点
    std::vector<FGPassNode>     m_passes;
    std::vector<FGResourceNode> m_resources;
    std::unordered_map<std::string, uint32_t> m_resourceNameMap;  // name → resource index

    // 编译/执行
    FGCompiler m_compiler;
    FGExecutor m_executor;
    FGDebug    m_debug;
    FGCompiler::CompiledGraph m_compiled;
    bool m_compiledThisFrame = false;

    // execute 期资源视图缓存
    FGResources m_viewCache;

    // ---- 内部方法 ----
    // 添加托管资源（由 FGBuilder::createTexture/createBuffer 调用）
    FGHandle addManagedResource(const FGResourceDesc& desc);

    // 查找资源节点
    FGResourceNode* findResource(FGHandle handle);
    const FGResourceNode* findResource(FGHandle handle) const;

    // 填充 FGResources 视图缓存
    void populateViewCache();

    // 获取 pipeline stage 对应的 VkPipelineStageFlags2
    static VkPipelineStageFlags2 stageForPassType(FGPassType type);
    static VkPipelineStageFlags2 readStageForPassType(FGPassType type);
};

} // namespace fg
} // namespace somegi
