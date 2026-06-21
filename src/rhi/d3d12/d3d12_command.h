// rhi/d3d12/d3d12_command.h — D3D12 命令缓冲和命令池
#pragma once
#include "../base/command_buffer.h"
#include "../base/fence.h"
#include "d3d12_fence.h"
#include <d3d12.h>
#include <stdexcept>
#include <vector>

namespace somegi {
namespace rhi {

class D3D12RHIDevice;
class D3D12RHIPipelineState;

// ════════════════════════════════════════════════════════════════
// D3D12RHICommandPool
// ════════════════════════════════════════════════════════════════
class D3D12RHICommandPool : public RHICommandPool {
public:
    D3D12RHICommandPool(D3D12RHIDevice& device);
    ~D3D12RHICommandPool() override;

    RHICommandBuffer* allocateRaw() override;
    void reset() override;

private:
    friend class D3D12RHICommandBuffer;
    D3D12RHIDevice& m_device;
    ID3D12CommandAllocator* m_allocator = nullptr;
};

// ════════════════════════════════════════════════════════════════
// D3D12RHICommandBuffer
// ════════════════════════════════════════════════════════════════
class D3D12RHICommandBuffer : public RHICommandBuffer {
public:
    D3D12RHICommandBuffer(D3D12RHIDevice& device, D3D12RHICommandPool& pool);
    ~D3D12RHICommandBuffer() override;

    // ── 生命周期 ──
    void begin() override;
    void end() override;
    void reset() override;

    // ── 动态状态 ──
    void setViewport(float x, float y, float w, float h, float minD, float maxD) override;
    void setScissor(int32_t x, int32_t y, uint32_t w, uint32_t h) override;

    // ── PSO / Descriptor / PushConstants ──
    void bindPipelineState(const RHIPipelineState& pso) override;
    void bindDescriptorSet(uint32_t slot, const RHIDescriptorSet& set) override;
    void bindDescriptorSets(uint32_t firstSlot, uint32_t count,
                            const RHIDescriptorSet* const* sets) override;
    void pushConstants(ShaderStage stage, const void* data,
                       uint32_t size, uint32_t offset) override;

    // ── 顶点/索引 ──
    void bindVertexBuffer(uint32_t binding, const RHIBuffer& buffer,
                          uint64_t offset, uint64_t stride) override;
    void bindIndexBuffer(const RHIBuffer& buffer, uint64_t offset,
                         bool uint16) override;

    // ── Draw ──
    void draw(uint32_t vc, uint32_t fv, uint32_t fi) override;
    void drawIndexed(uint32_t ic, uint32_t fi, int32_t vo) override;
    void drawIndirect(const RHIBuffer&, uint64_t, uint32_t, uint32_t) override;
    void drawIndexedIndirect(const RHIBuffer&, uint64_t, uint32_t, uint32_t) override;
    void drawIndexedIndirectCount(const RHIBuffer&, uint64_t,
                                   const RHIBuffer&, uint64_t,
                                   uint32_t, uint32_t) override;
    void drawMeshTasks(uint32_t, uint32_t, uint32_t) override;
    void drawMeshTasksIndirect(const RHIBuffer&, uint64_t, uint32_t, uint32_t) override;

    // ── Dispatch ──
    void dispatch(uint32_t, uint32_t, uint32_t) override;
    void dispatchIndirect(const RHIBuffer&, uint64_t) override;

    // ── 复制/清除 ──
    void copyBuffer(const RHIBuffer&, const RHIBuffer&, uint64_t,
                    uint64_t, uint64_t) override;
    void copyTexture(const RHITexture&, const RHITexture&) override;
    void fillBuffer(const RHIBuffer&, uint64_t, uint64_t, uint32_t) override;
    void clearColor(const RHITexture&, float, float, float, float) override;
    void clearDepth(const RHITexture&, float, uint32_t) override;

    // ── Barrier ──
    void textureBarrier(const RHITexture&, TextureLayout, TextureLayout) override;
    void bufferBarrier(const RHIBuffer&, PipelineStage, PipelineStage,
                       BufferAccess, BufferAccess) override;
    void globalBarrier() override;

    // ── 渲染通道 ──
    void beginRendering(const RenderingAttachmentInfo*, uint32_t,
                        const RenderingAttachmentInfo*,
                        uint32_t, uint32_t) override;
    void endRendering() override;

    // ── 时间戳 ──
    void writeTimestamp(const RHIQueryPool&, uint32_t) override;
    void resetQueryPool(const RHIQueryPool&, uint32_t, uint32_t) override;

    void* nativeHandle() const override { return (void*)m_cmdList; }

    // ── D3D12 特定 ──
    ID3D12GraphicsCommandList* cmdList() { return m_cmdList; }

    // 获取或创建 Indirect Draw 命令签名
    ID3D12CommandSignature* getDrawIndexedSignature();
    ID3D12CommandSignature* getDrawIndexedCountSignature();

private:
    D3D12RHIDevice& m_device;
    D3D12RHICommandPool& m_pool;
    ID3D12GraphicsCommandList* m_cmdList = nullptr;
    bool m_recording = false;
    const D3D12RHIPipelineState* m_boundPSO = nullptr; // 当前绑定的 PSO

    // 缓存的命令签名 + upload buffer
    ID3D12CommandSignature* m_drawIndexedSig = nullptr;
    ID3D12CommandSignature* m_drawIndexedCountSig = nullptr;
    ID3D12Resource* m_fillUploadBuf = nullptr; // fillBuffer 持久化 upload buffer
    uint64_t m_fillUploadSize = 0;
};


} // namespace rhi
} // namespace somegi
