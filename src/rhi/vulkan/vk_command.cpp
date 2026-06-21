// rhi/vulkan/vk_command.cpp
#include "vk_command.h"
#include "vk_pso.h"
#include "vk_descriptor.h"
#include "vk_buffer.h"
#include "vk_texture.h"
#include "vk_query_pool.h"
#include <core/vk_common.h>  // VK_CHECK

namespace somegi {
namespace rhi {

static VkImageLayout toVkLayout(TextureLayout l) {
    switch (l) {
        case TextureLayout::Undefined:         return VK_IMAGE_LAYOUT_UNDEFINED;
        case TextureLayout::General:           return VK_IMAGE_LAYOUT_GENERAL;
        case TextureLayout::ShaderReadOnly:    return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        case TextureLayout::ColorAttachment:   return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        case TextureLayout::DepthAttachment:   return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        case TextureLayout::TransferSrc:       return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        case TextureLayout::TransferDst:       return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        case TextureLayout::Present:           return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        default: return VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

static VkPipelineStageFlags2 toVkStage(PipelineStage s) {
    VkPipelineStageFlags2 f = 0;
    if ((uint32_t)s & (uint32_t)PipelineStage::VertexShader)     f |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    if ((uint32_t)s & (uint32_t)PipelineStage::FragmentShader)   f |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    if ((uint32_t)s & (uint32_t)PipelineStage::ComputeShader)    f |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    if ((uint32_t)s & (uint32_t)PipelineStage::MeshShader)       f |= VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    if ((uint32_t)s & (uint32_t)PipelineStage::TaskShader)       f |= VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
    if ((uint32_t)s & (uint32_t)PipelineStage::RayTracingShader) f |= VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    if ((uint32_t)s & (uint32_t)PipelineStage::Transfer)         f |= VK_PIPELINE_STAGE_2_COPY_BIT;
    if (s == PipelineStage::AllCommands)                         f = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    return f;
}

static VkAccessFlags2 toVkAccess(BufferAccess a) {
    VkAccessFlags2 f = 0;
    if ((uint32_t)a & (uint32_t)BufferAccess::UniformRead)   f |= VK_ACCESS_2_UNIFORM_READ_BIT;
    if ((uint32_t)a & (uint32_t)BufferAccess::StorageRead)   f |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    if ((uint32_t)a & (uint32_t)BufferAccess::StorageWrite)  f |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    if ((uint32_t)a & (uint32_t)BufferAccess::IndexRead)     f |= VK_ACCESS_2_INDEX_READ_BIT;
    if ((uint32_t)a & (uint32_t)BufferAccess::VertexRead)    f |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
    if ((uint32_t)a & (uint32_t)BufferAccess::IndirectRead)  f |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    if ((uint32_t)a & (uint32_t)BufferAccess::TransferRead)  f |= VK_ACCESS_2_TRANSFER_READ_BIT;
    if ((uint32_t)a & (uint32_t)BufferAccess::TransferWrite) f |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
    return f;
}

// ════════════════════════════════════════════════════════════════
// Command Pool
// ════════════════════════════════════════════════════════════════
std::unique_ptr<RHICommandPool> VkRHICommandPool::create(VkRHIDevice& device) {
    auto p = new VkRHICommandPool(device);
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.queueFamilyIndex = device.queueFamily();
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device.vkDevice(), &ci, nullptr, &p->m_pool));
    return std::unique_ptr<RHICommandPool>(p);
}
VkRHICommandPool::~VkRHICommandPool() { if (m_pool) vkDestroyCommandPool(m_device.vkDevice(), m_pool, nullptr); }

RHICommandBuffer* VkRHICommandPool::allocateRaw() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = m_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(m_device.vkDevice(), &ai, &cmd));
    return new VkRHICommandBuffer(m_device, cmd);
}
void VkRHICommandPool::reset() { vkResetCommandPool(m_device.vkDevice(), m_pool, 0); }

// ════════════════════════════════════════════════════════════════
// Command Buffer
// ════════════════════════════════════════════════════════════════
void VkRHICommandBuffer::begin() {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(m_cmd, &bi));
}
void VkRHICommandBuffer::end() { VK_CHECK(vkEndCommandBuffer(m_cmd)); }
void VkRHICommandBuffer::reset() { VK_CHECK(vkResetCommandBuffer(m_cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)); }

// ════════════════════════════════════════════════════════════════
// 动态状态
// ════════════════════════════════════════════════════════════════
void VkRHICommandBuffer::setViewport(float x, float y, float w, float h, float minD, float maxD) {
    VkViewport vp{x, y, w, h, minD, maxD};
    vkCmdSetViewport(m_cmd, 0, 1, &vp);
}
void VkRHICommandBuffer::setScissor(int32_t x, int32_t y, uint32_t w, uint32_t h) {
    VkRect2D sc{{x, y}, {w, h}};
    vkCmdSetScissor(m_cmd, 0, 1, &sc);
}

void VkRHICommandBuffer::bindPipelineState(const RHIPipelineState& pso) {
    auto* vkpso = static_cast<const VkRHIPipelineState*>(&pso);
    m_bindPoint = vkpso->bindPoint();
    m_boundLayout = vkpso->layout();
    vkCmdBindPipeline(m_cmd, m_bindPoint, (VkPipeline)(uintptr_t)pso.nativeHandle());
}
void VkRHICommandBuffer::bindDescriptorSet(uint32_t slot, const RHIDescriptorSet& set) {
    VkDescriptorSet ds = (VkDescriptorSet)(uintptr_t)set.nativeHandle();
    vkCmdBindDescriptorSets(m_cmd, m_bindPoint, m_boundLayout, slot, 1, &ds, 0, nullptr);
}
void VkRHICommandBuffer::bindDescriptorSets(uint32_t firstSlot, uint32_t count,
                                             const RHIDescriptorSet* const* sets) {
    std::vector<VkDescriptorSet> vkSets(count);
    for (uint32_t i = 0; i < count; ++i)
        vkSets[i] = (VkDescriptorSet)(uintptr_t)sets[i]->nativeHandle();
    vkCmdBindDescriptorSets(m_cmd, m_bindPoint, m_boundLayout,
                            firstSlot, count, vkSets.data(), 0, nullptr);
}
void VkRHICommandBuffer::pushConstants(ShaderStage stage, const void* data, uint32_t size, uint32_t offset) {
    // 映射 RHI ShaderStage → Vulkan stage flags（必须匹配 pipeline layout 的 push constant range）
    VkShaderStageFlags vkStage = 0;
    if ((uint32_t)stage & (uint32_t)ShaderStage::Vertex)   vkStage |= VK_SHADER_STAGE_VERTEX_BIT;
    if ((uint32_t)stage & (uint32_t)ShaderStage::Fragment) vkStage |= VK_SHADER_STAGE_FRAGMENT_BIT;
    if ((uint32_t)stage & (uint32_t)ShaderStage::Compute)  vkStage |= VK_SHADER_STAGE_COMPUTE_BIT;
    if ((uint32_t)stage & (uint32_t)ShaderStage::Mesh)     vkStage |= VK_SHADER_STAGE_MESH_BIT_EXT;
    if ((uint32_t)stage & (uint32_t)ShaderStage::Task)     vkStage |= VK_SHADER_STAGE_TASK_BIT_EXT;
    if (!vkStage) vkStage = VK_SHADER_STAGE_COMPUTE_BIT; // fallback
    vkCmdPushConstants(m_cmd, m_boundLayout, vkStage, offset, size, data);
}

void VkRHICommandBuffer::bindVertexBuffer(uint32_t binding, const RHIBuffer& buffer,
                                          uint64_t offset, uint64_t stride) {
    (void)stride;
    VkBuffer b = (VkBuffer)(uintptr_t)buffer.nativeHandle();
    VkDeviceSize o = (VkDeviceSize)offset;
    vkCmdBindVertexBuffers(m_cmd, binding, 1, &b, &o);
}
void VkRHICommandBuffer::bindIndexBuffer(const RHIBuffer& buffer, uint64_t offset, bool uint16) {
    vkCmdBindIndexBuffer(m_cmd, (VkBuffer)(uintptr_t)buffer.nativeHandle(), offset, uint16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
}

void VkRHICommandBuffer::draw(uint32_t vc, uint32_t fv, uint32_t fi) { vkCmdDraw(m_cmd, vc, 1, fv, fi); }
void VkRHICommandBuffer::drawIndexed(uint32_t ic, uint32_t fi, int32_t vo) { vkCmdDrawIndexed(m_cmd, ic, 1, fi, vo, 0); }
void VkRHICommandBuffer::drawIndirect(const RHIBuffer& buf, uint64_t off, uint32_t dc, uint32_t stride) {
    vkCmdDrawIndirect(m_cmd, (VkBuffer)(uintptr_t)buf.nativeHandle(), off, dc, stride);
}
void VkRHICommandBuffer::drawIndexedIndirect(const RHIBuffer& buf, uint64_t off, uint32_t dc, uint32_t stride) {
    vkCmdDrawIndexedIndirect(m_cmd, (VkBuffer)(uintptr_t)buf.nativeHandle(), off, dc, stride);
}
void VkRHICommandBuffer::drawIndexedIndirectCount(const RHIBuffer& buf, uint64_t off,
                                                   const RHIBuffer& countBuf, uint64_t countOff,
                                                   uint32_t maxDraws, uint32_t stride) {
    vkCmdDrawIndexedIndirectCount(m_cmd, (VkBuffer)(uintptr_t)buf.nativeHandle(), off,
                                  (VkBuffer)(uintptr_t)countBuf.nativeHandle(), countOff,
                                  maxDraws, stride);
}
void VkRHICommandBuffer::drawMeshTasks(uint32_t gx, uint32_t gy, uint32_t gz) {
    m_device.dispatch().cmdDrawMeshTasksEXT(m_cmd, gx, gy, gz);
}
void VkRHICommandBuffer::drawMeshTasksIndirect(const RHIBuffer& buf, uint64_t off, uint32_t dc, uint32_t stride) {
    m_device.dispatch().cmdDrawMeshTasksIndirectEXT(m_cmd, (VkBuffer)(uintptr_t)buf.nativeHandle(), off, dc, stride);
}

void VkRHICommandBuffer::dispatch(uint32_t gx, uint32_t gy, uint32_t gz) { vkCmdDispatch(m_cmd, gx, gy, gz); }
void VkRHICommandBuffer::dispatchIndirect(const RHIBuffer& buf, uint64_t off) {
    vkCmdDispatchIndirect(m_cmd, (VkBuffer)(uintptr_t)buf.nativeHandle(), off);
}

void VkRHICommandBuffer::copyBuffer(const RHIBuffer& src, const RHIBuffer& dst, uint64_t size, uint64_t srcOff, uint64_t dstOff) {
    VkBufferCopy region{srcOff, dstOff, size};
    vkCmdCopyBuffer(m_cmd, (VkBuffer)(uintptr_t)src.nativeHandle(), (VkBuffer)(uintptr_t)dst.nativeHandle(), 1, &region);
}
void VkRHICommandBuffer::copyTexture(const RHITexture& src, const RHITexture& dst) {
    VkImageCopy region{};
    auto aspect = toVkAspect(src.format());
    region.srcSubresource = {aspect, 0, 0, 1};
    region.dstSubresource = {aspect, 0, 0, 1};
    region.extent = {src.width(), src.height(), 1};
    vkCmdCopyImage(m_cmd, (VkImage)(uintptr_t)src.nativeHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   (VkImage)(uintptr_t)dst.nativeHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}
void VkRHICommandBuffer::fillBuffer(const RHIBuffer& dst, uint64_t offset, uint64_t size, uint32_t data) {
    vkCmdFillBuffer(m_cmd, (VkBuffer)(uintptr_t)dst.nativeHandle(), offset, size, data);
}
void VkRHICommandBuffer::copyBufferToTexture(const RHIBuffer& src, const RHITexture& dst,
                                               const BufferTextureCopyRegion& region) {
    auto& vkSrc = static_cast<const VkRHIBuffer&>(src);
    auto& vkDst = static_cast<const VkRHITexture&>(dst);

    VkBufferImageCopy2 copy{VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2};
    copy.bufferOffset = region.bufferOffset;
    copy.bufferRowLength = region.bufferRowLength;
    copy.bufferImageHeight = region.bufferImageHeight;
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.mipLevel = region.texMipLevel;
    copy.imageSubresource.baseArrayLayer = region.texArrayLayer;
    copy.imageSubresource.layerCount = 1;
    copy.imageOffset = {region.texOffsetX, region.texOffsetY, region.texOffsetZ};
    copy.imageExtent = {region.extentWidth, region.extentHeight, region.extentDepth};

    VkCopyBufferToImageInfo2 info{VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2};
    info.srcBuffer = (VkBuffer)(uintptr_t)vkSrc.nativeHandle();
    info.dstImage = (VkImage)(uintptr_t)vkDst.nativeHandle();
    info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    info.regionCount = 1;
    info.pRegions = &copy;

    vkCmdCopyBufferToImage2(m_cmd, &info);
}

void VkRHICommandBuffer::blitTexture(const RHITexture& src, const RHITexture& dst,
                                       const TextureBlitRegion& region) {
    auto& vkSrc = static_cast<const VkRHITexture&>(src);
    auto& vkDst = static_cast<const VkRHITexture&>(dst);

    VkImageBlit2 blit{VK_STRUCTURE_TYPE_IMAGE_BLIT_2};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel = region.srcMipLevel;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = region.layerCount;
    blit.srcOffsets[0] = {region.srcOffsetX, region.srcOffsetY, region.srcOffsetZ};
    blit.srcOffsets[1] = {region.srcOffsetX + (int32_t)region.srcExtentWidth,
                          region.srcOffsetY + (int32_t)region.srcExtentHeight,
                          region.srcOffsetZ + (int32_t)region.srcExtentDepth};
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.mipLevel = region.dstMipLevel;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = region.layerCount;
    blit.dstOffsets[0] = {region.dstOffsetX, region.dstOffsetY, region.dstOffsetZ};
    blit.dstOffsets[1] = {region.dstOffsetX + (int32_t)region.dstExtentWidth,
                          region.dstOffsetY + (int32_t)region.dstExtentHeight,
                          region.dstOffsetZ + (int32_t)region.dstExtentDepth};

    VkBlitImageInfo2 info{VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2};
    info.srcImage = (VkImage)(uintptr_t)vkSrc.nativeHandle();
    info.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    info.dstImage = (VkImage)(uintptr_t)vkDst.nativeHandle();
    info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    info.regionCount = 1;
    info.pRegions = &blit;
    info.filter = region.linearFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;

    vkCmdBlitImage2(m_cmd, &info);
}

void VkRHICommandBuffer::clearColor(const RHITexture& tex, float r, float g, float b, float a) {
    VkClearColorValue cv{r, g, b, a};
    VkImageSubresourceRange range{toVkAspect(tex.format()), 0, tex.mipLevels(), 0, 1};
    vkCmdClearColorImage(m_cmd, (VkImage)(uintptr_t)tex.nativeHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &range);
}
void VkRHICommandBuffer::clearDepth(const RHITexture& tex, float depth, uint32_t stencil) {
    VkClearDepthStencilValue cv{depth, stencil};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
    vkCmdClearDepthStencilImage(m_cmd, (VkImage)(uintptr_t)tex.nativeHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &range);
}

void VkRHICommandBuffer::textureBarrier(const RHITexture& tex, TextureLayout oldLayout, TextureLayout newLayout) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = toVkLayout(oldLayout);
    b.newLayout = toVkLayout(newLayout);
    b.image = (VkImage)(uintptr_t)tex.nativeHandle();
    b.subresourceRange = {toVkAspect(tex.format()), 0, tex.mipLevels(), 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(m_cmd, &di);
}
void VkRHICommandBuffer::globalBarrier() {
    VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1; di.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(m_cmd, &di);
}
void VkRHICommandBuffer::bufferBarrier(const RHIBuffer& buf,
                                       PipelineStage srcStage, PipelineStage dstStage,
                                       BufferAccess srcAccess, BufferAccess dstAccess) {
    VkBufferMemoryBarrier2 b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    b.srcStageMask = toVkStage(srcStage);
    b.srcAccessMask = toVkAccess(srcAccess);
    b.dstStageMask = toVkStage(dstStage);
    b.dstAccessMask = toVkAccess(dstAccess);
    b.buffer = (VkBuffer)(uintptr_t)buf.nativeHandle();
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.bufferMemoryBarrierCount = 1;
    di.pBufferMemoryBarriers = &b;
    vkCmdPipelineBarrier2(m_cmd, &di);
}

static VkAttachmentLoadOp toVkLoadOp(AttachmentLoadOp op) {
    switch (op) {
        case AttachmentLoadOp::Clear:    return VK_ATTACHMENT_LOAD_OP_CLEAR;
        case AttachmentLoadOp::Load:     return VK_ATTACHMENT_LOAD_OP_LOAD;
        case AttachmentLoadOp::DontCare: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        default: return VK_ATTACHMENT_LOAD_OP_CLEAR;
    }
}
static VkAttachmentStoreOp toVkStoreOp(AttachmentStoreOp op) {
    switch (op) {
        case AttachmentStoreOp::Store:    return VK_ATTACHMENT_STORE_OP_STORE;
        case AttachmentStoreOp::DontCare: return VK_ATTACHMENT_STORE_OP_DONT_CARE;
        default: return VK_ATTACHMENT_STORE_OP_STORE;
    }
}
static VkResolveModeFlagBits toVkResolveMode(ResolveMode m) {
    switch (m) {
        case ResolveMode::Average: return VK_RESOLVE_MODE_AVERAGE_BIT;
        case ResolveMode::Min:     return VK_RESOLVE_MODE_MIN_BIT;
        case ResolveMode::Max:     return VK_RESOLVE_MODE_MAX_BIT;
        default: return VK_RESOLVE_MODE_AVERAGE_BIT;
    }
}

void VkRHICommandBuffer::beginRendering(const RenderingAttachmentInfo* colorAttachments, uint32_t colorCount,
                                         const RenderingAttachmentInfo* depthAttachment, uint32_t width, uint32_t height) {
    // ── Color attachments ──
    std::vector<VkRenderingAttachmentInfo> colors(colorCount);
    for (uint32_t i = 0; i < colorCount; ++i) {
        auto& a = colorAttachments[i];
        colors[i] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        colors[i].imageView   = (VkImageView)(uintptr_t)a.view->nativeHandle();
        colors[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colors[i].loadOp      = toVkLoadOp(a.loadOp);
        colors[i].storeOp     = toVkStoreOp(a.storeOp);
        std::memcpy(colors[i].clearValue.color.float32, a.clearColor, sizeof(a.clearColor));
        // MSAA resolve
        if (a.resolveView) {
            colors[i].resolveImageView   = (VkImageView)(uintptr_t)a.resolveView->nativeHandle();
            colors[i].resolveMode        = toVkResolveMode(a.resolveMode);
            colors[i].resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
    }

    // ── Depth attachment ──
    VkRenderingAttachmentInfo depth{};
    VkRenderingAttachmentInfo* pDepth = nullptr;
    if (depthAttachment && depthAttachment->view) {
        depth = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView   = (VkImageView)(uintptr_t)depthAttachment->view->nativeHandle();
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp      = toVkLoadOp(depthAttachment->loadOp);
        depth.storeOp     = toVkStoreOp(depthAttachment->storeOp);
        depth.clearValue.depthStencil = {depthAttachment->clearDepth, depthAttachment->clearStencil};
        // MSAA resolve
        if (depthAttachment->resolveView) {
            depth.resolveImageView   = (VkImageView)(uintptr_t)depthAttachment->resolveView->nativeHandle();
            depth.resolveMode        = toVkResolveMode(depthAttachment->resolveMode);
            depth.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        }
        pDepth = &depth;
    }

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {width, height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = colorCount;
    ri.pColorAttachments    = colors.data();
    ri.pDepthAttachment     = pDepth;
    vkCmdBeginRendering(m_cmd, &ri);
}
void VkRHICommandBuffer::endRendering() { vkCmdEndRendering(m_cmd); }

void VkRHICommandBuffer::writeTimestamp(const RHIQueryPool& pool, uint32_t index) {
    vkCmdWriteTimestamp2(m_cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, (VkQueryPool)(uintptr_t)pool.nativeHandle(), index);
}
void VkRHICommandBuffer::resetQueryPool(const RHIQueryPool& pool, uint32_t first, uint32_t count) {
    vkCmdResetQueryPool(m_cmd, (VkQueryPool)(uintptr_t)pool.nativeHandle(), first, count);
}

} // namespace rhi
} // namespace somegi
