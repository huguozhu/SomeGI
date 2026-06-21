// rhi/d3d12/d3d12_command.cpp — D3D12 命令缓冲/池/栅栏实现
#include "d3d12_command.h"
#include "d3d12_device.h"
#include "d3d12_texture.h"
#include "d3d12_buffer.h"
#include "d3d12_pso.h"
#include "d3d12_swapchain.h"
#include "d3d12_query_pool.h"
#include <stdexcept>
#include <cstdio>
#include <cstring>

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
    if (m_fillUploadBuf) m_fillUploadBuf->Release();
    if (m_cmdList) m_cmdList->Release();
}

void D3D12RHICommandBuffer::begin() {
    m_pool.m_allocator->Reset();
    m_cmdList->Reset(m_pool.m_allocator, nullptr);

    // 绑定 GPU 可见描述符堆（CBV_SRV_UAV + Sampler）
    {
        ID3D12DescriptorHeap* heaps[] = {
            m_device.gpuDescriptorHeap(),
            m_device.gpuSamplerHeap()
        };
        m_cmdList->SetDescriptorHeaps(2, heaps);
    }

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
    m_boundPSO = &d3dPso; // 记录用于 descriptor set 根参数映射
}

void D3D12RHICommandBuffer::bindDescriptorSet(uint32_t slot,
                                               const RHIDescriptorSet& set) {
    auto& d3dSet = static_cast<const D3D12RHIDescriptorSet&>(set);
    if (m_boundPSO) {
        uint32_t resParam = m_boundPSO->getResourceParamForSet(slot);
        uint32_t smpParam = m_boundPSO->getSamplerParamForSet(slot);
        if (resParam != ~0u) {
            if (m_boundPSO->isCompute())
                m_cmdList->SetComputeRootDescriptorTable(resParam, d3dSet.gpuHandle());
            else
                m_cmdList->SetGraphicsRootDescriptorTable(resParam, d3dSet.gpuHandle());
        }
        if (smpParam != ~0u && d3dSet.samplerGpuHandle().ptr != 0) {
            if (m_boundPSO->isCompute())
                m_cmdList->SetComputeRootDescriptorTable(smpParam,
                    d3dSet.samplerGpuHandle());
            else
                m_cmdList->SetGraphicsRootDescriptorTable(smpParam,
                    d3dSet.samplerGpuHandle());
        }
    } else {
        m_cmdList->SetComputeRootDescriptorTable(slot, d3dSet.gpuHandle());
    }
}

void D3D12RHICommandBuffer::bindDescriptorSets(uint32_t firstSlot, uint32_t count,
                                                const RHIDescriptorSet* const* sets) {
    for (uint32_t i = 0; i < count; ++i) {
        bindDescriptorSet(firstSlot + i, *sets[i]);
    }
}

void D3D12RHICommandBuffer::pushConstants(ShaderStage stage, const void* data,
                                           uint32_t size, uint32_t offset) {
    // 简化：root constants 参数索引固定为 0（Phase 5 从 PSO 获取正确索引）
    if (stage == ShaderStage::Compute || (uint32_t)stage & (uint32_t)ShaderStage::Compute)
        m_cmdList->SetComputeRoot32BitConstants(0, size / 4, data, offset / 4);
    else
        m_cmdList->SetGraphicsRoot32BitConstants(0, size / 4, data, offset / 4);
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
void D3D12RHICommandBuffer::drawIndirect(const RHIBuffer& buf, uint64_t offset,
                                          uint32_t drawCount, uint32_t stride) {
    auto& d3dBuf = static_cast<const D3D12RHIBuffer&>(buf);
    // 复用 drawIndexed 命令签名（D3D12 通过 stride 区分 indexed vs non-indexed）
    auto* sig = getDrawIndexedSignature();
    m_cmdList->ExecuteIndirect(sig, drawCount, d3dBuf.resource(), offset, nullptr, 0);
}
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
void D3D12RHICommandBuffer::dispatchIndirect(const RHIBuffer& buf, uint64_t offset) {
    auto& d3dBuf = static_cast<const D3D12RHIBuffer&>(buf);
    // 创建 Dispatch 命令签名（懒初始化）
    static ID3D12CommandSignature* s_dispatchSig = nullptr;
    if (!s_dispatchSig) {
        D3D12_INDIRECT_ARGUMENT_DESC arg{};
        arg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        D3D12_COMMAND_SIGNATURE_DESC sd{};
        sd.ByteStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
        sd.NumArgumentDescs = 1;
        sd.pArgumentDescs = &arg;
        m_device.device()->CreateCommandSignature(&sd, nullptr, IID_PPV_ARGS(&s_dispatchSig));
    }
    m_cmdList->ExecuteIndirect(s_dispatchSig, 1, d3dBuf.resource(), offset, nullptr, 0);
}

// ════════════════════════════════════════════════════════════════
// 复制/清除
// ════════════════════════════════════════════════════════════════

void D3D12RHICommandBuffer::copyBuffer(const RHIBuffer& src, const RHIBuffer& dst,
                                        uint64_t size, uint64_t srcOff, uint64_t dstOff) {
    auto& d3dSrc = static_cast<const D3D12RHIBuffer&>(src);
    auto& d3dDst = static_cast<const D3D12RHIBuffer&>(dst);
    m_cmdList->CopyBufferRegion(d3dDst.resource(), dstOff,
                                 d3dSrc.resource(), srcOff, size);
}
void D3D12RHICommandBuffer::copyTexture(const RHITexture& src, const RHITexture& dst) {
    auto& d3dSrc = static_cast<const D3D12RHITexture&>(src);
    auto& d3dDst = static_cast<const D3D12RHITexture&>(dst);
    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = d3dSrc.resource();
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = d3dDst.resource();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = 0;
    m_cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
}
void D3D12RHICommandBuffer::fillBuffer(const RHIBuffer& dst, uint64_t offset,
                                        uint64_t size, uint32_t data) {
    auto& d3dBuf = static_cast<const D3D12RHIBuffer&>(dst);
    // 使用持久化 upload buffer（延迟分配，按需扩展）
    if (!m_fillUploadBuf || m_fillUploadSize < size) {
        if (m_fillUploadBuf) m_fillUploadBuf->Release();
        m_fillUploadSize = (size + 255) & ~255ull; // 256 对齐
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_UPLOAD,
            D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
        D3D12_RESOURCE_DESC rd{D3D12_RESOURCE_DIMENSION_BUFFER, 0, m_fillUploadSize,
            1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        m_device.device()->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_fillUploadBuf));
    }
    // 填充 upload buffer
    uint8_t* mapped = nullptr;
    m_fillUploadBuf->Map(0, nullptr, (void**)&mapped);
    for (uint64_t i = 0; i < size; i += 4)
        std::memcpy(mapped + i, &data, size - i < 4 ? size - i : 4);
    m_fillUploadBuf->Unmap(0, nullptr);
    m_cmdList->CopyBufferRegion(d3dBuf.resource(), offset, m_fillUploadBuf, 0, size);
}
void D3D12RHICommandBuffer::copyBufferToTexture(const RHIBuffer& src, const RHITexture& dst,
                                                  const BufferTextureCopyRegion& region) {
    auto& d3dSrc = static_cast<const D3D12RHIBuffer&>(src);
    auto& d3dDst = static_cast<const D3D12RHITexture&>(dst);

    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = d3dDst.resource();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = region.texMipLevel;

    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = d3dSrc.resource();
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;

    // 查询目标纹理在指定 mip level 的布局
    auto* device = m_device.device();
    D3D12_RESOURCE_DESC dstDesc = d3dDst.resource()->GetDesc();

    UINT numRows = 0;
    UINT64 rowSizeBytes = 0;
    UINT64 totalBytes = 0;
    device->GetCopyableFootprints(&dstDesc, region.texMipLevel, 1,
                                    region.bufferOffset,
                                    &srcLoc.PlacedFootprint, &numRows, &rowSizeBytes, &totalBytes);

    srcLoc.PlacedFootprint.Offset = region.bufferOffset;
    srcLoc.PlacedFootprint.Footprint.Width = region.extentWidth;
    srcLoc.PlacedFootprint.Footprint.Height = region.extentHeight;
    srcLoc.PlacedFootprint.Footprint.Depth = region.extentDepth;
    srcLoc.PlacedFootprint.Footprint.RowPitch = region.bufferRowLength > 0
        ? region.bufferRowLength : rowSizeBytes;

    m_cmdList->CopyTextureRegion(&dstLoc, region.texOffsetX, region.texOffsetY, 0,
                                   &srcLoc, nullptr);
}

void D3D12RHICommandBuffer::blitTexture(const RHITexture& src, const RHITexture& dst,
                                          const TextureBlitRegion& region) {
    // D3D12 没有直接的 blit API。使用 CopyTextureRegion 进行整个子资源的复制。
    // 带 linear filtering 的 mip 生成无法在此完成 —— 调用者应使用 compute shader。
    auto& d3dSrc = static_cast<const D3D12RHITexture&>(src);
    auto& d3dDst = static_cast<const D3D12RHITexture&>(dst);

    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = d3dSrc.resource();
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex = region.srcMipLevel;

    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = d3dDst.resource();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = region.dstMipLevel;

    D3D12_BOX srcBox{};
    srcBox.left = region.srcOffsetX;
    srcBox.top = region.srcOffsetY;
    srcBox.front = region.srcOffsetZ;
    srcBox.right = region.srcOffsetX + region.extentWidth;
    srcBox.bottom = region.srcOffsetY + region.extentHeight;
    srcBox.back = region.srcOffsetZ + region.extentDepth;

    m_cmdList->CopyTextureRegion(&dstLoc, region.dstOffsetX, region.dstOffsetY, 0,
                                   &srcLoc, &srcBox);
}

void D3D12RHICommandBuffer::clearColor(const RHITexture& tex,
                                        float r, float g, float b, float a) {
    auto& d3dTex = static_cast<const D3D12RHITexture&>(tex);
    // 在 CPU RTV heap 上创建临时 RTV descriptor 并清除
    if (m_device.cpuRtvHeap()) {
        static uint32_t tmpRtvIdx = 256; // 使用 heap 后半段避免与持久 RTV 冲突
        D3D12_CPU_DESCRIPTOR_HANDLE h = m_device.cpuRtvHeap()->GetCPUDescriptorHandleForHeapStart();
        h.ptr += tmpRtvIdx++ * m_device.cpuRtvIncrement();
        if (tmpRtvIdx > 384) tmpRtvIdx = 256; // 循环使用

        D3D12_RENDER_TARGET_VIEW_DESC rtvd{};
        rtvd.Format = d3dTex.dxgiFormat();
        rtvd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        m_device.device()->CreateRenderTargetView(d3dTex.resource(), &rtvd, h);

        const float clear[4] = { r, g, b, a };
        m_cmdList->ClearRenderTargetView(h, clear, 0, nullptr);
    }
}
void D3D12RHICommandBuffer::clearDepth(const RHITexture& tex,
                                        float depth, uint32_t stencil) {
    auto& d3dTex = static_cast<const D3D12RHITexture&>(tex);
    if (m_device.cpuDsvHeap()) {
        static uint32_t tmpDsvIdx = 32;
        D3D12_CPU_DESCRIPTOR_HANDLE h = m_device.cpuDsvHeap()->GetCPUDescriptorHandleForHeapStart();
        h.ptr += tmpDsvIdx++ * m_device.cpuDsvIncrement();
        if (tmpDsvIdx > 48) tmpDsvIdx = 32;

        D3D12_DEPTH_STENCIL_VIEW_DESC dsvd{};
        dsvd.Format = d3dTex.dxgiFormat();
        dsvd.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        m_device.device()->CreateDepthStencilView(d3dTex.resource(), &dsvd, h);

        D3D12_CLEAR_FLAGS flags = D3D12_CLEAR_FLAG_DEPTH;
        if (stencil) flags |= D3D12_CLEAR_FLAG_STENCIL;
        m_cmdList->ClearDepthStencilView(h, flags, depth, (UINT8)stencil, 0, nullptr);
    }
}

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

    // 从状态追踪器获取当前状态（而非始终 COMMON）
    D3D12_RESOURCE_STATES before = m_device.getResourceState(d3dTex.resource());
    D3D12_RESOURCE_STATES after  = before;

    switch (newLayout) {
        case TextureLayout::ColorAttachment: after = D3D12_RESOURCE_STATE_RENDER_TARGET; break;
        case TextureLayout::DepthAttachment: after = D3D12_RESOURCE_STATE_DEPTH_WRITE; break;
        case TextureLayout::ShaderReadOnly:  after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; break;
        case TextureLayout::TransferDst:     after = D3D12_RESOURCE_STATE_COPY_DEST; break;
        case TextureLayout::TransferSrc:     after = D3D12_RESOURCE_STATE_COPY_SOURCE; break;
        case TextureLayout::Present:         after = D3D12_RESOURCE_STATE_PRESENT; break;
        case TextureLayout::General:         after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; break;
        default: break;
    }

    // 状态未变化则跳过
    if (before == after) return;

    D3D12_RESOURCE_BARRIER rb{};
    rb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    rb.Transition.pResource = d3dTex.resource();
    rb.Transition.StateBefore = before;
    rb.Transition.StateAfter = after;
    rb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_cmdList->ResourceBarrier(1, &rb);

    // 更新追踪状态
    m_device.trackResourceState(d3dTex.resource(), after);
}

// ════════════════════════════════════════════════════════════════
// BufferAccess → D3D12_RESOURCE_STATES 映射（用于 bufferBarrier）
// ════════════════════════════════════════════════════════════════
static D3D12_RESOURCE_STATES toD3D12BufferState(BufferAccess access) {
    if (access == BufferAccess::None) return D3D12_RESOURCE_STATE_COMMON;

    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;

    if ((uint32_t)access & (uint32_t)BufferAccess::UniformRead)
        state |= D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
               | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if ((uint32_t)access & (uint32_t)BufferAccess::StorageRead)
        state |= D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if ((uint32_t)access & (uint32_t)BufferAccess::StorageWrite)
        state |= D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if ((uint32_t)access & (uint32_t)BufferAccess::IndexRead)
        state |= D3D12_RESOURCE_STATE_INDEX_BUFFER;
    if ((uint32_t)access & (uint32_t)BufferAccess::VertexRead)
        state |= D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    if ((uint32_t)access & (uint32_t)BufferAccess::IndirectRead)
        state |= D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    if ((uint32_t)access & (uint32_t)BufferAccess::TransferRead)
        state |= D3D12_RESOURCE_STATE_COPY_SOURCE;
    if ((uint32_t)access & (uint32_t)BufferAccess::TransferWrite)
        state |= D3D12_RESOURCE_STATE_COPY_DEST;

    return state;
}

void D3D12RHICommandBuffer::bufferBarrier(const RHIBuffer& buf,
                                           PipelineStage /*srcStage*/,
                                           PipelineStage /*dstStage*/,
                                           BufferAccess srcAccess,
                                           BufferAccess dstAccess) {
    auto& d3dBuf = static_cast<const D3D12RHIBuffer&>(buf);

    D3D12_RESOURCE_STATES before = m_device.getResourceState(d3dBuf.resource());
    D3D12_RESOURCE_STATES after  = toD3D12BufferState(dstAccess);

    if (before == after) {
        // UAV → UAV 特殊处理：TRANSITION barrier 不允许 before == after，
        // fallback 为 UAV barrier 确保 UAV 写入对其他 pass 可见
        if (before & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12_RESOURCE_BARRIER rb{};
            rb.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            rb.UAV.pResource = d3dBuf.resource();
            m_cmdList->ResourceBarrier(1, &rb);
        }
        // 其他相同状态：跳过（无需 barrier）
        return;
    }

    D3D12_RESOURCE_BARRIER rb{};
    rb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    rb.Transition.pResource   = d3dBuf.resource();
    rb.Transition.StateBefore = before;
    rb.Transition.StateAfter  = after;
    rb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_cmdList->ResourceBarrier(1, &rb);

    m_device.trackResourceState(d3dBuf.resource(), after);
}
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
        rtvs[i] = view->rtvCpuHandle();
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
        dsvHandle = dView->dsvCpuHandle();
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

void D3D12RHICommandBuffer::writeTimestamp(const RHIQueryPool& pool, uint32_t index) {
    auto& d3dPool = static_cast<const D3D12RHIQueryPool&>(pool);
    m_cmdList->EndQuery(d3dPool.heap(), D3D12_QUERY_TYPE_TIMESTAMP, index);
}
void D3D12RHICommandBuffer::resetQueryPool(const RHIQueryPool& pool, uint32_t first,
                                            uint32_t count) {
    // D3D12 query heap 不需要显式重置；时间戳通过 resolve 回读
    (void)pool; (void)first; (void)count;
}

// ════════════════════════════════════════════════════════════════

} // namespace rhi
} // namespace somegi
