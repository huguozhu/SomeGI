// rhi/vulkan/vk_common.h — Vulkan RHI 内部工具
#pragma once
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>
#include "../base/common.h"

#define VK_CHECK(expr)                                                     \
    do {                                                                   \
        VkResult _r = (expr);                                              \
        if (_r != VK_SUCCESS) {                                            \
            throw std::runtime_error(std::string(#expr " failed: ") +      \
                                     std::to_string(static_cast<int>(_r)));\
        }                                                                  \
    } while (0)

namespace somegi {
namespace rhi {

// ── Vk→RHI 映射函数 ──

inline TextureLayout fromVkLayout(VkImageLayout l) {
    switch (l) {
        case VK_IMAGE_LAYOUT_UNDEFINED:                   return TextureLayout::Undefined;
        case VK_IMAGE_LAYOUT_GENERAL:                     return TextureLayout::General;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:    return TextureLayout::ColorAttachment;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:    return TextureLayout::DepthAttachment;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:    return TextureLayout::ShaderReadOnly;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:        return TextureLayout::TransferSrc;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:        return TextureLayout::TransferDst;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:             return TextureLayout::Present;
        default: return TextureLayout::Undefined;
    }
}

inline PipelineStage fromVkStage(VkPipelineStageFlags2 s) {
    if (s == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)     return PipelineStage::AllCommands;
    if (s == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT)      return PipelineStage::TopOfPipe;
    if (s == VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT)   return PipelineStage::BottomOfPipe;
    uint32_t r = 0;
    if (s & VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT)          r |= (uint32_t)PipelineStage::VertexShader;
    if (s & VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)        r |= (uint32_t)PipelineStage::FragmentShader;
    if (s & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)         r |= (uint32_t)PipelineStage::ComputeShader;
    if (s & VK_PIPELINE_STAGE_2_COPY_BIT)                   r |= (uint32_t)PipelineStage::Transfer;
    if (s & VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT)        r |= (uint32_t)PipelineStage::MeshShader;
    if (s & VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT)        r |= (uint32_t)PipelineStage::TaskShader;
    if (s & VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR) r |= (uint32_t)PipelineStage::RayTracingShader;
    return (PipelineStage)r;
}

inline BufferAccess fromVkAccess(VkAccessFlags2 a) {
    uint32_t r = 0;
    if (a & VK_ACCESS_2_UNIFORM_READ_BIT)                 r |= (uint32_t)BufferAccess::UniformRead;
    if (a & VK_ACCESS_2_SHADER_STORAGE_READ_BIT)          r |= (uint32_t)BufferAccess::StorageRead;
    if (a & VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)         r |= (uint32_t)BufferAccess::StorageWrite;
    if (a & VK_ACCESS_2_INDEX_READ_BIT)                   r |= (uint32_t)BufferAccess::IndexRead;
    if (a & VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT)        r |= (uint32_t)BufferAccess::VertexRead;
    if (a & VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT)        r |= (uint32_t)BufferAccess::IndirectRead;
    if (a & VK_ACCESS_2_TRANSFER_READ_BIT)                r |= (uint32_t)BufferAccess::TransferRead;
    if (a & VK_ACCESS_2_TRANSFER_WRITE_BIT)               r |= (uint32_t)BufferAccess::TransferWrite;
    if (a & VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT)        r |= (uint32_t)BufferAccess::ColorAttachmentRead;
    if (a & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)       r |= (uint32_t)BufferAccess::ColorAttachmentWrite;
    if (a & VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT)  r |= (uint32_t)BufferAccess::DepthStencilRead;
    if (a & VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) r |= (uint32_t)BufferAccess::DepthStencilWrite;
    if (a & VK_ACCESS_2_MEMORY_READ_BIT)                  r |= (uint32_t)BufferAccess::MemoryRead;
    if (a & VK_ACCESS_2_MEMORY_WRITE_BIT)                 r |= (uint32_t)BufferAccess::MemoryWrite;
    return (BufferAccess)r;
}

} // namespace rhi
} // namespace somegi
