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

void VkRHICommandBuffer::bindPipelineState(const RHIPipelineState& pso) {
    auto* vkpso = static_cast<const VkRHIPipelineState*>(&pso);
    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)(uintptr_t)pso.nativeHandle());
    (void)vkpso; // 后续通过 pso 缓存 pipeline layout
}
void VkRHICommandBuffer::bindDescriptorSet(uint32_t slot, const RHIDescriptorSet& set) {
    VkDescriptorSet ds = (VkDescriptorSet)(uintptr_t)set.nativeHandle();
    vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VK_NULL_HANDLE, slot, 1, &ds, 0, nullptr);
}
void VkRHICommandBuffer::pushConstants(ShaderStage stage, const void* data, uint32_t size, uint32_t offset) {
    (void)stage;
    vkCmdPushConstants(m_cmd, VK_NULL_HANDLE, VK_SHADER_STAGE_ALL, offset, size, data);
}

void VkRHICommandBuffer::bindVertexBuffer(const RHIBuffer& buffer, uint64_t offset) {
    VkBuffer b = (VkBuffer)(uintptr_t)buffer.nativeHandle();
    VkDeviceSize o = (VkDeviceSize)offset;
    vkCmdBindVertexBuffers(m_cmd, 0, 1, &b, &o);
}
void VkRHICommandBuffer::bindIndexBuffer(const RHIBuffer& buffer, uint64_t offset, bool uint16) {
    vkCmdBindIndexBuffer(m_cmd, (VkBuffer)(uintptr_t)buffer.nativeHandle(), offset, uint16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
}

void VkRHICommandBuffer::draw(uint32_t vc, uint32_t fv, uint32_t fi) { vkCmdDraw(m_cmd, vc, 1, fv, fi); }
void VkRHICommandBuffer::drawIndexed(uint32_t ic, uint32_t fi, int32_t vo) { vkCmdDrawIndexed(m_cmd, ic, 1, fi, vo, 0); }
void VkRHICommandBuffer::drawIndexedIndirect(const RHIBuffer& buf, uint64_t off, uint32_t dc, uint32_t stride) {
    vkCmdDrawIndexedIndirect(m_cmd, (VkBuffer)(uintptr_t)buf.nativeHandle(), off, dc, stride);
}
void VkRHICommandBuffer::drawMeshTasks(uint32_t gx, uint32_t gy, uint32_t gz) {
    vkCmdDrawMeshTasksEXT(m_cmd, gx, gy, gz);
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
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {src.width(), src.height(), 1};
    vkCmdCopyImage(m_cmd, (VkImage)(uintptr_t)src.nativeHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   (VkImage)(uintptr_t)dst.nativeHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}
void VkRHICommandBuffer::clearColor(const RHITextureView& view, float r, float g, float b, float a) {
    VkClearColorValue cv{r, g, b, a};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(m_cmd, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &range);
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
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, tex.mipLevels(), 0, 1};
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

void VkRHICommandBuffer::beginRendering(const RHITextureView* colorViews, uint32_t colorCount,
                                         const RHITextureView* depthView, uint32_t width, uint32_t height) {
    std::vector<VkRenderingAttachmentInfo> colors(colorCount);
    for (uint32_t i = 0; i < colorCount; ++i) {
        colors[i] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        colors[i].imageView = (VkImageView)(uintptr_t)colorViews[i].nativeHandle();
        colors[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colors[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colors[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    VkRenderingAttachmentInfo depth{};
    if (depthView) {
        depth = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView = (VkImageView)(uintptr_t)depthView->nativeHandle();
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {width, height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = colorCount;
    ri.pColorAttachments = colors.data();
    ri.pDepthAttachment = depthView ? &depth : nullptr;
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
