// src/renderer/fg/fg_builder.h
#pragma once
#include "fg_common.h"
#include <functional>

namespace somegi {
namespace fg {

class FrameGraph;
struct FGPassNode;
struct FGResourceNode;
struct FGResources;

// ============================================================
// FGBuilder: 在 addPass() 的 setup lambda 中使用
//
// 生命周期：仅在 setup lambda 执行期间有效。
// 内部持有指向当前 FGPassNode 的非拥有指针。
// ============================================================
class FGBuilder {
public:
    // 构造函数由 FrameGraph::addPass() 调用，用户不直接创建
    FGBuilder(FrameGraph& graph, FGPassNode& passNode);

    // ---- Pass 类型 ----
    // 显式指定 pass 类型，覆盖自动推导
    FGBuilder& setPassType(FGPassType type);

    // ---- 资源依赖声明 ----
    // 声明读依赖：pass 执行前资源处于 SHADER_READ_ONLY_OPTIMAL
    FGHandle read(FGHandle handle);

    // 声明写依赖：pass 执行前资源处于合适的写 layout
    FGHandle write(FGHandle handle);

    // 声明读写依赖：in-place update（先读后写同一资源）
    FGHandle readWrite(FGHandle handle);

    // ---- 托管资源创建 ----
    FGHandle createTexture(const char* name, const FGTextureDesc& desc);
    FGHandle createBuffer(const char* name, const FGBufferDesc& desc);

    // ---- 执行回调 ----
    // fn 签名: void(VkCommandBuffer cmd, const FGResources& resources)
    template<typename F>
    void setExecute(F&& fn) {
        m_passNode->execute = std::forward<F>(fn);
    }

private:
    FrameGraph&   m_graph;
    FGPassNode*   m_passNode = nullptr;
};

} // namespace fg
} // namespace somegi
