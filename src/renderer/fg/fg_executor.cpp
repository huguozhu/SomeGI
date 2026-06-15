// src/renderer/fg/fg_executor.cpp
#include "fg_executor.h"
#include "fg_compiler.h"
#include "fg_pass_node.h"
#include "fg_resource_node.h"
#include "fg_resources.h"
#include "core/device.h"
#include "core/image.h"
#include "core/buffer.h"
#include <cstdio>

namespace somegi {
namespace fg {

void FGExecutor::init(Device& device) {
    m_device = &device;
}

void FGExecutor::destroy() {
    m_texturePool.clear();
    m_bufferPool.clear();
}

void FGExecutor::execute(VkCommandBuffer cmd,
                          FGCompiler::CompiledGraph& compiled,
                          const FGResources& viewCache) {
    // 1. 分配别名组
    for (auto& group : compiled.aliasGroups) {
        allocateAliasGroup(group, compiled.resources);
    }

    // 2. 遍历执行 pass
    for (auto* pass : compiled.passOrder) {
        if (!pass || pass->culled) continue;

        // 2a. 插入 barrier（仅在 autoBarriers 模式）
        if (m_autoBarriers) {
            emitBarriers(cmd, *pass, compiled.resources, viewCache);
        }

        // 2b. 执行 pass
        if (pass->execute) {
            pass->execute(cmd, viewCache);
        }

        // 2c. 更新资源状态
        updateResourceStates(*pass, compiled.resources);
    }

    // 3. 回收
    recycleUnused(kRecycleFrames);

    ++m_currentFrame;
}

// ---- 别名组分配 ----

void FGExecutor::allocateAliasGroup(const FGCompiler::AliasGroup& group,
                                     std::vector<FGResourceNode*>& resources) {
    (void)resources;

    Image* sharedTexture = nullptr;
    Buffer* sharedBuffer = nullptr;

    for (auto* member : group.members) {
        if (!member) continue;

        if (member->desc.type == FGResourceType::Texture) {
            if (!sharedTexture) {
                sharedTexture = allocateTexture(member->desc);
            }
            member->physicalTexture = sharedTexture;
            member->physicalBuffer = nullptr;
        } else {
            if (!sharedBuffer) {
                sharedBuffer = allocateBuffer(member->desc);
            }
            member->physicalBuffer = sharedBuffer;
            member->physicalTexture = nullptr;
        }
    }
}

// ---- 纹理分配 ----

Image* FGExecutor::allocateTexture(const FGResourceDesc& desc) {
    // 先在池中查找匹配的闲置纹理
    for (auto& pt : m_texturePool) {
        if (!pt.inUse &&
            pt.format == desc.texture.format &&
            pt.extent.width >= desc.texture.extent.width &&
            pt.extent.height >= desc.texture.extent.height &&
            pt.extent.depth >= desc.texture.extent.depth) {
            pt.inUse = true;
            pt.lastUsedFrame = m_currentFrame;
            return &pt.image;
        }
    }

    // 未命中 → 创建新纹理
    ImageDesc imgDesc;
    imgDesc.format = desc.texture.format;
    imgDesc.extent = desc.texture.extent;
    imgDesc.mipLevels = desc.texture.mipLevels;
    imgDesc.arrayLayers = desc.texture.arrayLayers;
    imgDesc.usage = desc.texture.usage;
    imgDesc.aspect = (desc.texture.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
                     ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    imgDesc.samples = desc.texture.samples;
    if (desc.texture.isCubemap) {
        imgDesc.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }

    PooledTexture pt;
    pt.image = Image(*m_device, imgDesc);
    pt.format = desc.texture.format;
    pt.extent = desc.texture.extent;
    pt.lastUsedFrame = m_currentFrame;
    pt.inUse = true;

    m_texturePool.push_back(std::move(pt));
    return &m_texturePool.back().image;
}

// ---- Buffer 分配 ----

Buffer* FGExecutor::allocateBuffer(const FGResourceDesc& desc) {
    // 先在池中查找匹配的闲置 Buffer
    for (auto& pb : m_bufferPool) {
        if (!pb.inUse && pb.size >= desc.buffer.size) {
            pb.inUse = true;
            pb.lastUsedFrame = m_currentFrame;
            return &pb.buffer;
        }
    }

    // 未命中 → 创建新 Buffer
    VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    PooledBuffer pb;
    pb.buffer = Buffer(*m_device, desc.buffer.size, desc.buffer.usage, memProps);
    pb.size = desc.buffer.size;
    pb.lastUsedFrame = m_currentFrame;
    pb.inUse = true;

    m_bufferPool.push_back(std::move(pb));
    return &m_bufferPool.back().buffer;
}

// ---- Barrier 插入 ----

void FGExecutor::emitBarriers(VkCommandBuffer cmd,
                               const FGPassNode& pass,
                               std::vector<FGResourceNode*>& resources,
                               const FGResources& viewCache) {
    (void)resources;
    (void)viewCache;

    std::vector<VkImageMemoryBarrier2> imgBarriers;
    std::vector<VkBufferMemoryBarrier2> bufBarriers;

    auto addBarrier = [&](const FGPassNode::ResourceRef& ref) {
        auto* res = ref.resource;
        if (!res) return;

        if (res->desc.type == FGResourceType::Texture && res->physicalTexture) {
            VkImageLayout currentLayout = res->state.layout;
            VkImageLayout targetLayout = ref.requiredLayout;

            if (res->isImported && currentLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
                return;  // 导入资源首次使用，信任 initialLayout
            }

            if (currentLayout == targetLayout &&
                res->state.access == ref.access) {
                return;  // 无需 barrier
            }

            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask  = res->state.stage;
            b.srcAccessMask = res->state.access;
            b.dstStageMask  = ref.stages;
            b.dstAccessMask = ref.access;
            b.oldLayout = currentLayout;
            b.newLayout = targetLayout;
            b.image = res->physicalTexture->image();
            b.subresourceRange = {
                static_cast<VkImageAspectFlags>(
                    (res->desc.texture.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
                        ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
                0, res->desc.texture.mipLevels, 0, res->desc.texture.arrayLayers
            };
            imgBarriers.push_back(b);
        }

        if (res->desc.type == FGResourceType::Buffer && res->physicalBuffer) {
            if (res->state.access == ref.access) return;

            VkBufferMemoryBarrier2 b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
            b.srcStageMask  = res->state.stage;
            b.srcAccessMask = res->state.access;
            b.dstStageMask  = ref.stages;
            b.dstAccessMask = ref.access;
            b.buffer = res->physicalBuffer->handle();
            b.offset = 0;
            b.size = VK_WHOLE_SIZE;
            bufBarriers.push_back(b);
        }
    };

    for (auto& ref : pass.reads)  addBarrier(ref);
    for (auto& ref : pass.writes) addBarrier(ref);

    if (!imgBarriers.empty() || !bufBarriers.empty()) {
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = (uint32_t)imgBarriers.size();
        di.pImageMemoryBarriers = imgBarriers.data();
        di.bufferMemoryBarrierCount = (uint32_t)bufBarriers.size();
        di.pBufferMemoryBarriers = bufBarriers.data();
        vkCmdPipelineBarrier2(cmd, &di);
    }
}

// ---- 状态更新 ----

void FGExecutor::updateResourceStates(const FGPassNode& pass,
                                       std::vector<FGResourceNode*>& resources) {
    (void)resources;

    for (auto& ref : pass.reads) {
        if (!ref.resource) continue;
        if (ref.resource->state.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            ref.resource->state.layout = ref.requiredLayout;
        }
        ref.resource->state.access = ref.access;
        ref.resource->state.stage = ref.stages;
    }

    for (auto& ref : pass.writes) {
        if (!ref.resource) continue;
        ref.resource->state.layout = ref.requiredLayout;
        ref.resource->state.access = ref.access;
        ref.resource->state.stage = ref.stages;
        ref.resource->state.lastWriter = const_cast<FGPassNode*>(&pass);
    }
}

// ---- 池回收 ----

void FGExecutor::recycleUnused(uint64_t threshold) {
    (void)threshold;

    for (auto& pt : m_texturePool) {
        if (pt.inUse) {
            pt.inUse = false;
        }
    }
    for (auto& pb : m_bufferPool) {
        if (pb.inUse) {
            pb.inUse = false;
        }
    }
}

// ============================================================
// 静态 Layout/Access/Stage 推导
// ============================================================

VkImageLayout FGExecutor::derivedLayout(FGPassType passType,
                                         VkImageUsageFlags usage,
                                         bool isWrite) {
    (void)passType;

    if (!isWrite) return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    return VK_IMAGE_LAYOUT_GENERAL;
}

VkAccessFlags2 FGExecutor::derivedAccess(FGPassType passType,
                                          VkImageUsageFlags usage,
                                          bool isWrite,
                                          bool isReadWrite) {
    (void)passType;

    if (isReadWrite) {
        return VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    }

    if (!isWrite) {
        if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
            return VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
            return VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        if (usage & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR)
            return VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        return VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    }

    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        return VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
        return VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    if (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        return VK_ACCESS_2_TRANSFER_WRITE_BIT;

    return VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
}

VkPipelineStageFlags2 FGExecutor::derivedStage(FGPassType passType,
                                                VkImageUsageFlags usage,
                                                bool isWrite) {
    switch (passType) {
        case FGPassType::Compute:
            return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

        case FGPassType::Graphics:
            if (isWrite && (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
                return VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            if (isWrite && (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
                return VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
            return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

        case FGPassType::MeshShading:
            return VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

        case FGPassType::RayTracing:
            return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    }

    return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
}

} // namespace fg
} // namespace somegi
