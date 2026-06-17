// rhi/base/command_buffer.h
#pragma once
#include "common.h"
#include <memory>

namespace somegi {
namespace rhi {

class RHIPipelineState;
class RHIDescriptorSet;
class RHIBuffer;
class RHITexture;
class RHITextureView;
class RHIFence;
class RHISemaphore;
class RHIQueryPool;

// ════════════════════════════════════════════════════════════════
// RHICommandPool
// ════════════════════════════════════════════════════════════════
class RHICommandBuffer; // 前向声明

class RHICommandPool {
public:
    virtual ~RHICommandPool() = default;
    virtual RHICommandBuffer* allocateRaw() = 0;
    virtual void reset() = 0;
};

// ════════════════════════════════════════════════════════════════
// RHICommandBuffer
// ════════════════════════════════════════════════════════════════
class RHICommandBuffer {
public:
    virtual ~RHICommandBuffer() = default;

    virtual void begin() = 0;
    virtual void end() = 0;
    virtual void reset() = 0;

    // ── 动态状态（视口 / 裁剪） ──
    virtual void setViewport(float x, float y, float width, float height,
                             float minDepth = 0.0f, float maxDepth = 1.0f) = 0;
    virtual void setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) = 0;

    // ── PSO 绑定 ──
    virtual void bindPipelineState(const RHIPipelineState& pso) = 0;
    virtual void bindDescriptorSet(uint32_t slot, const RHIDescriptorSet& set) = 0;
    // 批量绑定多个描述符集（firstSlot 为起始槽位，count 为数量）
    virtual void bindDescriptorSets(uint32_t firstSlot, uint32_t count,
                                    const RHIDescriptorSet* const* sets) = 0;
    virtual void pushConstants(ShaderStage stage, const void* data, uint32_t size, uint32_t offset = 0) = 0;

    // ── 顶点 / 索引 ──
    virtual void bindVertexBuffer(const RHIBuffer& buffer, uint64_t offset = 0) = 0;
    virtual void bindIndexBuffer(const RHIBuffer& buffer, uint64_t offset = 0, bool uint16 = true) = 0;

    // ── Draw ──
    virtual void draw(uint32_t vertexCount, uint32_t firstVertex = 0, uint32_t firstInstance = 0) = 0;
    virtual void drawIndexed(uint32_t indexCount, uint32_t firstIndex = 0, int32_t vertexOffset = 0) = 0;
    virtual void drawIndirect(const RHIBuffer& buffer, uint64_t offset, uint32_t drawCount, uint32_t stride) = 0;
    virtual void drawIndexedIndirect(const RHIBuffer& buffer, uint64_t offset, uint32_t drawCount, uint32_t stride) = 0;
    virtual void drawMeshTasks(uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1) = 0;
    virtual void drawMeshTasksIndirect(const RHIBuffer& buffer, uint64_t offset,
                                       uint32_t drawCount, uint32_t stride) = 0;

    // ── Dispatch ──
    virtual void dispatch(uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1) = 0;
    virtual void dispatchIndirect(const RHIBuffer& buffer, uint64_t offset) = 0;

    // ── 复制 / 清除 ──
    virtual void copyBuffer(const RHIBuffer& src, const RHIBuffer& dst, uint64_t size,
                            uint64_t srcOffset = 0, uint64_t dstOffset = 0) = 0;
    virtual void copyTexture(const RHITexture& src, const RHITexture& dst) = 0;
    virtual void clearColor(const RHITextureView& view, float r, float g, float b, float a) = 0;
    virtual void clearDepth(const RHITextureView& view, float depth, uint32_t stencil = 0) = 0;

    // ── Barrier ──
    // 纹理屏障（自动推断阶段和访问掩码）
    virtual void textureBarrier(const RHITexture& tex, TextureLayout oldLayout, TextureLayout newLayout) = 0;
    // Buffer 屏障（需要显式指定源/目标阶段和访问类型）
    virtual void bufferBarrier(const RHIBuffer& buf,
                               PipelineStage srcStage, PipelineStage dstStage,
                               BufferAccess srcAccess, BufferAccess dstAccess) = 0;
    // 全局管线屏障（最重的同步，覆盖所有资源）
    virtual void globalBarrier() = 0;

    // ── 渲染通道（Metal 显式 render pass 模型） ──
    // loadOnly=true: LOAD_OP_LOAD（保留已有内容）；false: LOAD_OP_CLEAR（清除）
    virtual void beginRendering(const RHITextureView* colorViews, uint32_t colorCount,
                                const RHITextureView* depthView,
                                uint32_t width, uint32_t height,
                                bool loadOnly = false) = 0;
    virtual void endRendering() = 0;

    // ── 时间戳 ──
    virtual void writeTimestamp(const RHIQueryPool& pool, uint32_t index) = 0;
    virtual void resetQueryPool(const RHIQueryPool& pool, uint32_t first, uint32_t count) = 0;

    // ── 原生句柄（兼容迁移期） ──
    virtual void* nativeHandle() const = 0;
};

// ════════════════════════════════════════════════════════════════
// RHIQueryPool
// ════════════════════════════════════════════════════════════════
class RHIQueryPool {
public:
    virtual ~RHIQueryPool() = default;
    virtual void getResults(uint32_t first, uint32_t count, uint64_t* data) = 0;
    virtual void* nativeHandle() const = 0;
};

// ════════════════════════════════════════════════════════════════
// 提交描述符
// ════════════════════════════════════════════════════════════════
struct SubmitDesc {
    const RHICommandBuffer* commandBuffer;
    const RHISemaphore* waitSemaphore = nullptr;
    const RHISemaphore* signalSemaphore = nullptr;
    const RHIFence* signalFence = nullptr;
};

} // namespace rhi
} // namespace somegi
