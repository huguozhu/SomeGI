// rhi/d3d12/d3d12_command.cpp — D3D12 命令缓冲/池/栅栏实现
#include "d3d12_command.h"
#include "d3d12_device.h"
#include "d3d12_texture.h"
#include "d3d12_buffer.h"
#include "d3d12_pso.h"
#include "d3d12_swapchain.h"
#include <stdexcept>
#include <cstdio>

namespace somegi {
namespace rhi {

// ════════════════════════════════════════════════════════════════
// D3D12RHICommandPool
// ════════════════════════════════════════════════════════════════

D3D12RHICommandPool::D3D12RHICommandPool(D3D12RHIDevice& device)
    : m_device(device) {
    if (FAILED(device.device()->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_allocator)))) {
        throw std::runtime_error("[d3d12] CreateCommandAllocator failed");
    }
}

D3D12RHICommandPool::~D3D12RHICommandPool() {
    if (m_allocator) m_allocator->Release();
}

RHICommandBuffer* D3D12RHICommandPool::allocateRaw() {
    return new D3D12RHICommandBuffer(m_device, *this);
}

void D3D12RHICommandPool::reset() {
    m_allocator->Reset();
}

// ════════════════════════════════════════════════════════════════
// D3D12RHICommandBuffer
// ════════════════════════════════════════════════════════════════

D3D12RHICommandBuffer::D3D12RHICommandBuffer(D3D12RHIDevice& device,
                                               D3D12RHICommandPool& pool)
    : m_device(device), m_pool(pool) {
    if (FAILED(device.device()->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT, pool.m_allocator,
            nullptr, IID_PPV_ARGS(&m_cmdList)))) {
        throw std::runtime_error("[d3d12] CreateCommandList failed");
    }
    // 创建后立即关闭（D3D12 命令列表创建时处于 recording 状态）
    m_cmdList->Close();
}

D3D12RHICommandBuffer::~D3D12RHICommandBuffer() {
    if (m_drawIndexedSig) m_drawIndexedSig->Release();
    if (m_drawIndexedCountSig) m_drawIndexedCountSig->Release();
    if (m_cmdList) m_cmdList->Release();
}

void D3D12RHICommandBuffer::begin() {
    m_pool.m_allocator->Reset();
    m_cmdList->Reset(m_pool.m_allocator, nullptr);
    m_recording = true;
}

void D3D12RHICommandBuffer::end() {
    m_cmdList->Close();
    m_recording = false;
}

void D3D12RHICommandBuffer::reset() {
    if (m_recording) end();
    begin();
}

void D3D12RHICommandBuffer::setViewport(float x, float y, float w, float h,
                                         float minD, float maxD) {
    D3D12_VIEWPORT vp{x, y, w, h, minD, maxD};
    m_cmdList->RSSetViewports(1, &vp);
}

void D3D12RHICommandBuffer::setScissor(int32_t x, int32_t y,
                                        uint32_t w, uint32_t h) {
    D3D12_RECT rc{LONG(x), LONG(y), LONG(x + w), LONG(y + h)};
    m_cmdList->RSSetScissorRects(1, &rc);
}

// ════════════════════════════════════════════════════════════════
// PSO / Descriptor / PushConstants
// ════════════════════════════════════════════════════════════════

void D3D12RHICommandBuffer::bindPipelineState(const RHIPipelineState& pso) {
    auto& d3dPso = static_cast<const D3D12RHIPipelineState&>(pso);
    m_cmdList->SetPipelineState(d3dPso.pipeline());
    if (d3dPso.isCompute())
        m_cmdList->SetComputeRootSignature(d3dPso.rootSignature());
    else
        m_cmdList->SetGraphicsRootSignature(d3dPso.rootSignature());
}

void D3D12RHICommandBuffer::bindDescriptorSet(uint32_t slot,
                                               const RHIDescriptorSet& set) {
    auto& d3dSet = static_cast<const D3D12RHIDescriptorSet&>(set);
    m_cmdList->SetComputeRootDescriptorTable(slot, d3dSet.gpuHandle());
}

void D3D12RHICommandBuffer::bindDescriptorSets(uint32_t firstSlot, uint32_t count,
                                                const RHIDescriptorSet* const* sets) {
    for (uint32_t i = 0; i < count; ++i) {
        bindDescriptorSet(firstSlot + i, *sets[i]);
    }
}

void D3D12RHICommandBuffer::pushConstants(ShaderStage, const void* data,
                                           uint32_t size, uint32_t offset) {
    // 简化：仅支持 compute root constants（参数索引固定为 descriptor set 之后的槽位）
    m_cmdList->SetComputeRoot32BitConstants(0, size / 4, data, offset / 4);
}

// ════════════════════════════════════════════════════════════════
// 顶点/索引
// ════════════════════════════════════════════════════════════════

void D3D12RHICommandBuffer::bindVertexBuffer(uint32_t binding,
                                              const RHIBuffer& buffer,
                                              uint64_t offset, uint64_t stride) {
    auto& d3dBuf = static_cast<const D3D12RHIBuffer&>(buffer);
    D3D12_VERTEX_BUFFER_VIEW vbv{};
    vbv.BufferLocation = d3dBuf.gpuAddress() + offset;
    vbv.SizeInBytes = (UINT)(d3dBuf.size() - offset);
    vbv.StrideInBytes = (UINT)(stride ? stride : 0); // stride=0 表示从绑定描述获取
    m_cmdList->IASetVertexBuffers(binding, 1, &vbv);
}

void D3D12RHICommandBuffer::bindIndexBuffer(const RHIBuffer& buffer,
                                             uint64_t offset, bool uint16) {
    auto& d3dBuf = static_cast<const D3D12RHIBuffer&>(buffer);
    D3D12_INDEX_BUFFER_VIEW ibv{};
    ibv.BufferLocation = d3dBuf.gpuAddress() + offset;
    ibv.SizeInBytes = (UINT)(d3dBuf.size() - offset);
    ibv.Format = uint16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
    m_cmdList->IASetIndexBuffer(&ibv);
}

// ════════════════════════════════════════════════════════════════
// Draw
// ════════════════════════════════════════════════════════════════

void D3D12RHICommandBuffer::draw(uint32_t vc, uint32_t fv, uint32_t fi) {
    m_cmdList->DrawInstanced(vc, 1, fv, fi);
}
void D3D12RHICommandBuffer::drawIndexed(uint32_t ic, uint32_t fi, int32_t vo) {
    m_cmdList->DrawIndexedInstanced(ic, 1, fi, vo, 0);
}
void D3D12RHICommandBuffer::drawIndirect(const RHIBuffer&, uint64_t, uint32_t, uint32_t) {}
void D3D12RHICommandBuffer::drawIndexedIndirect(const RHIBuffer& buf, uint64_t offset,
                                                  uint32_t drawCount, uint32_t stride) {
    auto& d3dBuf = static_cast<const D3D12RHIBuffer&>(buf);
    auto* sig = getDrawIndexedSignature();
    m_cmdList->ExecuteIndirect(sig, drawCount, d3dBuf.resource(), offset, nullptr, 0);
}
void D3D12RHICommandBuffer::drawIndexedIndirectCount(const RHIBuffer& drawBuf,
                                                       uint64_t drawOffset,
                                                       const RHIBuffer& countBuf,
                                                       uint64_t countOffset,
                                                       uint32_t maxDrawCount,
                                                       uint32_t stride) {
    auto& d3dDraw = static_cast<const D3D12RHIBuffer&>(drawBuf);
    auto& d3dCount = static_cast<const D3D12RHIBuffer&>(countBuf);
    auto* sig = getDrawIndexedCountSignature();
    m_cmdList->ExecuteIndirect(sig, maxDrawCount, d3dDraw.resource(), drawOffset,
                                d3dCount.resource(), countOffset);
}
void D3D12RHICommandBuffer::drawMeshTasks(uint32_t, uint32_t, uint32_t) {}
void D3D12RHICommandBuffer::drawMeshTasksIndirect(const RHIBuffer&, uint64_t,
                                                    uint32_t, uint32_t) {}

// ════════════════════════════════════════════════════════════════
// Dispatch
// ════════════════════════════════════════════════════════════════

void D3D12RHICommandBuffer::dispatch(uint32_t gx, uint32_t gy, uint32_t gz) {
    m_cmdList->Dispatch(gx, gy, gz);
}
void D3D12RHICommandBuffer::dispatchIndirect(const RHIBuffer&, uint64_t) {}

// ════════════════════════════════════════════════════════════════
// 复制/清除
// ════════════════════════════════════════════════════════════════

void D3D12RHICommandBuffer::copyBuffer(const RHIBuffer&, const RHIBuffer&,
                                        uint64_t, uint64_t, uint64_t) {}
void D3D12RHICommandBuffer::copyTexture(const RHITexture&, const RHITexture&) {}
void D3D12RHICommandBuffer::fillBuffer(const RHIBuffer&, uint64_t, uint64_t, uint32_t) {}
void D3D12RHICommandBuffer::clearColor(const RHITexture& tex,
                                        float r, float g, float b, float a) {
    // 简化：仅支持创建了 RTV descriptor 的纹理
    // 完整实现需在 D3D12RHITexture 上缓存 RTV handle
    auto& d3dTex = static_cast<const D3D12RHITexture&>(tex);
    const float clear[4] = { r, g, b, a };
    // 使用临时 descriptor heap 创建 RTV 并清除
    // Phase 5 完整实现 descriptor heap 管理后改进
    (void)d3dTex; (void)clear;
}
void D3D12RHICommandBuffer::clearDepth(const RHITexture&, float, uint32_t) {}

// ════════════════════════════════════════════════════════════════
// Indirect Draw 命令签名
// ════════════════════════════════════════════════════════════════

ID3D12CommandSignature* D3D12RHICommandBuffer::getDrawIndexedSignature() {
    if (m_drawIndexedSig) return m_drawIndexedSig;

    D3D12_INDIRECT_ARGUMENT_DESC arg{};
    arg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;

    D3D12_COMMAND_SIGNATURE_DESC sd{};
    sd.ByteStride = sizeof(D3D12_DRAW_INDEXED_ARGUMENTS); // 3 * uint32_t
    sd.NumArgumentDescs = 1;
    sd.pArgumentDescs = &arg;
    sd.NodeMask = 0;

    m_device.device()->CreateCommandSignature(&sd, nullptr,
        IID_PPV_ARGS(&m_drawIndexedSig));
    return m_drawIndexedSig;
}

ID3D12CommandSignature* D3D12RHICommandBuffer::getDrawIndexedCountSignature() {
    if (m_drawIndexedCountSig) return m_drawIndexedCountSig;

    D3D12_INDIRECT_ARGUMENT_DESC args[2]{};
    args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
    args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
    args[1].Constant.RootParameterIndex = 0; // draw count
    args[1].Constant.DestOffsetIn32BitValues = 0;
    args[1].Constant.Num32BitValuesToSet = 1;

    D3D12_COMMAND_SIGNATURE_DESC sd{};
    sd.ByteStride = sizeof(D3D12_DRAW_INDEXED_ARGUMENTS); // 3 * uint32_t (indexCountPerInstance, instanceCount, startIndexLocation, baseVertexLocation, startInstanceLocation)
    // Wait, D3D12_DRAW_INDEXED_ARGUMENTS is actually larger. Let me use the correct size.
    sd.ByteStride = 20; // indexCount(4) + instanceCount(4) + startIndex(4) + baseVertex(4) + startInstance(4) = 20
    sd.NumArgumentDescs = 2;
    sd.pArgumentDescs = args;
    sd.NodeMask = 0;

    m_device.device()->CreateCommandSignature(&sd, nullptr,
        IID_PPV_ARGS(&m_drawIndexedCountSig));
    return m_drawIndexedCountSig;
}

// ════════════════════════════════════════════════════════════════
// Barrier
// ════════════════════════════════════════════════════════════════

void D3D12RHICommandBuffer::textureBarrier(const RHITexture& tex,
                                            TextureLayout oldLayout,
                                            TextureLayout newLayout) {
    auto& d3dTex = static_cast<const D3D12RHITexture&>(tex);
    // 简化映射（Phase 4 完整实现 ResourceStateTracker）
    D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES after  = D3D12_RESOURCE_STATE_COMMON;
    switch (newLayout) {
        case TextureLayout::ColorAttachment: after = D3D12_RESOURCE_STATE_RENDER_TARGET; break;
        case TextureLayout::DepthAttachment: after = D3D12_RESOURCE_STATE_DEPTH_WRITE; break;
        case TextureLayout::ShaderReadOnly:  after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; break;
        case TextureLayout::TransferDst:     after = D3D12_RESOURCE_STATE_COPY_DEST; break;
        case TextureLayout::TransferSrc:     after = D3D12_RESOURCE_STATE_COPY_SOURCE; break;
        case TextureLayout::Present:         after = D3D12_RESOURCE_STATE_PRESENT; break;
        default: break;
    }
    D3D12_RESOURCE_BARRIER rb{};
    rb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    rb.Transition.pResource = d3dTex.resource();
    rb.Transition.StateBefore = before;
    rb.Transition.StateAfter = after;
    rb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_cmdList->ResourceBarrier(1, &rb);
}

void D3D12RHICommandBuffer::bufferBarrier(const RHIBuffer&,
                                           PipelineStage, PipelineStage,
                                           BufferAccess, BufferAccess) {}
void D3D12RHICommandBuffer::globalBarrier() {
    D3D12_RESOURCE_BARRIER rb{};
    rb.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    rb.UAV.pResource = nullptr;
    m_cmdList->ResourceBarrier(1, &rb);
}

// ════════════════════════════════════════════════════════════════
// 渲染通道
// ════════════════════════════════════════════════════════════════

void D3D12RHICommandBuffer::beginRendering(const RenderingAttachmentInfo* colors,
                                            uint32_t colorCount,
                                            const RenderingAttachmentInfo* depth,
                                            uint32_t width, uint32_t height) {
    (void)width; (void)height;

    D3D12_CPU_DESCRIPTOR_HANDLE rtvs[8]{};
    for (uint32_t i = 0; i < colorCount && i < 8; ++i) {
        if (!colors[i].view) continue;
        auto* view = static_cast<const D3D12RHITextureView*>(colors[i].view);
        rtvs[i] = view->srvCpuHandle();
    }

    // 清除 RTV（如果 loadOp 为 Clear）
    for (uint32_t i = 0; i < colorCount && i < 8; ++i) {
        if (colors[i].loadOp == AttachmentLoadOp::Clear && rtvs[i].ptr) {
            m_cmdList->ClearRenderTargetView(rtvs[i], colors[i].clearColor, 0, nullptr);
        }
    }

    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle{};
    const D3D12_CPU_DESCRIPTOR_HANDLE* dsv = nullptr;
    if (depth && depth->view) {
        auto* dView = static_cast<const D3D12RHITextureView*>(depth->view);
        dsvHandle = dView->srvCpuHandle();
        dsv = &dsvHandle;
        if (depth->loadOp == AttachmentLoadOp::Clear) {
            m_cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH,
                                              depth->clearDepth, 0, 0, nullptr);
        }
    }
    m_cmdList->OMSetRenderTargets(colorCount, rtvs, FALSE, dsv);
}

void D3D12RHICommandBuffer::endRendering() {
    // D3D12 无需显式结束渲染通道
}

// ════════════════════════════════════════════════════════════════
// 时间戳
// ════════════════════════════════════════════════════════════════

void D3D12RHICommandBuffer::writeTimestamp(const RHIQueryPool&, uint32_t) {}
void D3D12RHICommandBuffer::resetQueryPool(const RHIQueryPool&, uint32_t, uint32_t) {}

// ════════════════════════════════════════════════════════════════
// D3D12RHIFence
// ════════════════════════════════════════════════════════════════

D3D12RHIFence::D3D12RHIFence(D3D12RHIDevice& device, bool signaled) {
    if (FAILED(device.device()->CreateFence(
            signaled ? 1 : 0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) {
        throw std::runtime_error("[d3d12] CreateFence failed");
    }
    m_event = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    m_value = signaled ? 1 : 0;
}

D3D12RHIFence::~D3D12RHIFence() {
    if (m_fence) m_fence->Release();
    if (m_event) CloseHandle(m_event);
}

void D3D12RHIFence::wait(uint64_t timeoutNs) {
    if (m_fence->GetCompletedValue() < m_value) {
        m_fence->SetEventOnCompletion(m_value, m_event);
        DWORD ms = (timeoutNs == UINT64_MAX) ? INFINITE
                   : static_cast<DWORD>(timeoutNs / 1'000'000);
        WaitForSingleObject(m_event, ms);
    }
}

void D3D12RHIFence::reset() {
    m_value++;
    m_fence->Signal(m_value);
}

// ════════════════════════════════════════════════════════════════
// D3D12RHISemaphore（桩，D3D12 用 fence 代替 semaphore）
// ════════════════════════════════════════════════════════════════

D3D12RHISemaphore::D3D12RHISemaphore() = default;
D3D12RHISemaphore::~D3D12RHISemaphore() = default;

} // namespace rhi
} // namespace somegi
