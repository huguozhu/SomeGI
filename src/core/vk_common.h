#pragma once

#include <vulkan/vulkan.h>
#include "rhi/base/common.h"    // PipelineStage, BufferAccess, TextureLayout
#include <stdexcept>
#include <string>

#define VK_CHECK(expr)                                                     \
    do {                                                                   \
        VkResult _r = (expr);                                              \
        if (_r != VK_SUCCESS) {                                            \
            throw std::runtime_error(std::string(#expr " failed: ") +      \
                                     std::to_string(static_cast<int>(_r)));\
        }                                                                  \
    } while (0)

namespace somegi {

constexpr uint32_t kFramesInFlight = 2;

// ════════════════════════════════════════════════════════════════
// Vk→RHI 反向映射（用于桥接遗留 transitionImage 到 RHI barrier）
// ════════════════════════════════════════════════════════════════
inline rhi::TextureLayout fromVkLayout(VkImageLayout l) {
    switch (l) {
        case VK_IMAGE_LAYOUT_UNDEFINED:           return rhi::TextureLayout::Undefined;
        case VK_IMAGE_LAYOUT_GENERAL:             return rhi::TextureLayout::General;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: return rhi::TextureLayout::ColorAttachment;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL: return rhi::TextureLayout::DepthAttachment;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return rhi::TextureLayout::ShaderReadOnly;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: return rhi::TextureLayout::TransferSrc;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return rhi::TextureLayout::TransferDst;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:     return rhi::TextureLayout::Present;
        default: return rhi::TextureLayout::Undefined;
    }
}

inline rhi::PipelineStage fromVkStage(VkPipelineStageFlags2 s) {
    if (s == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)     return rhi::PipelineStage::AllCommands;
    if (s == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT)      return rhi::PipelineStage::TopOfPipe;
    if (s == VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT)   return rhi::PipelineStage::BottomOfPipe;
    uint32_t r = 0;
    if (s & VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT)     r |= (uint32_t)rhi::PipelineStage::VertexShader;
    if (s & VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)   r |= (uint32_t)rhi::PipelineStage::FragmentShader;
    if (s & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)    r |= (uint32_t)rhi::PipelineStage::ComputeShader;
    if (s & VK_PIPELINE_STAGE_2_COPY_BIT)              r |= (uint32_t)rhi::PipelineStage::Transfer;
    if (s & VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT)   r |= (uint32_t)rhi::PipelineStage::MeshShader;
    if (s & VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT)   r |= (uint32_t)rhi::PipelineStage::TaskShader;
    if (s & VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR) r |= (uint32_t)rhi::PipelineStage::RayTracingShader;
    return (rhi::PipelineStage)r;
}

inline rhi::BufferAccess fromVkAccess(VkAccessFlags2 a) {
    uint32_t r = 0;
    if (a & VK_ACCESS_2_UNIFORM_READ_BIT)              r |= (uint32_t)rhi::BufferAccess::UniformRead;
    if (a & VK_ACCESS_2_SHADER_STORAGE_READ_BIT)       r |= (uint32_t)rhi::BufferAccess::StorageRead;
    if (a & VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)      r |= (uint32_t)rhi::BufferAccess::StorageWrite;
    if (a & VK_ACCESS_2_INDEX_READ_BIT)                r |= (uint32_t)rhi::BufferAccess::IndexRead;
    if (a & VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT)     r |= (uint32_t)rhi::BufferAccess::VertexRead;
    if (a & VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT)     r |= (uint32_t)rhi::BufferAccess::IndirectRead;
    if (a & VK_ACCESS_2_TRANSFER_READ_BIT)             r |= (uint32_t)rhi::BufferAccess::TransferRead;
    if (a & VK_ACCESS_2_TRANSFER_WRITE_BIT)            r |= (uint32_t)rhi::BufferAccess::TransferWrite;
    if (a & VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT)     r |= (uint32_t)rhi::BufferAccess::ColorAttachmentRead;
    if (a & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)    r |= (uint32_t)rhi::BufferAccess::ColorAttachmentWrite;
    if (a & VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT)  r |= (uint32_t)rhi::BufferAccess::DepthStencilRead;
    if (a & VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) r |= (uint32_t)rhi::BufferAccess::DepthStencilWrite;
    if (a & VK_ACCESS_2_MEMORY_READ_BIT)               r |= (uint32_t)rhi::BufferAccess::MemoryRead;
    if (a & VK_ACCESS_2_MEMORY_WRITE_BIT)              r |= (uint32_t)rhi::BufferAccess::MemoryWrite;
    return (rhi::BufferAccess)r;
}

// Vulkan 图像布局转换辅助函数 —— 内部委托到 RHI barrier API。
// (注：rhiCmd 必须由调用方通过 VkRHICommandBuffer(vkDev, cmd) 构造并提供)
namespace rhi { class VkRHICommandBuffer; class VkRHIDevice; class VkRHITexture; }

// 旧接口：仅 VkCommandBuffer（保留兼容，内部直接调用 vkCmdPipelineBarrier2）
inline void transitionImage(VkCommandBuffer cmd, VkImage image,
                            VkImageAspectFlags aspect,
                            VkImageLayout oldLayout, VkImageLayout newLayout,
                            VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                            VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask  = srcStage;  b.srcAccessMask = srcAccess;
    b.dstStageMask  = dstStage;  b.dstAccessMask = dstAccess;
    b.oldLayout = oldLayout;     b.newLayout = newLayout;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};

    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}

}
