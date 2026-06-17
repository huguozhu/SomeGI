// src/renderer/fg/fg_executor.cpp
#include "fg_executor.h"
#include "fg_compiler.h"
#include "fg_pass_node.h"
#include "fg_resource_node.h"
#include "fg_resources.h"
#include <algorithm>
#include <unordered_set>
#include "core/device.h"
#include "core/image.h"
#include "core/buffer.h"
#include "rhi/base/command_buffer.h"  // RHICommandBuffer::nativeHandle
#include <cstdio>

namespace somegi {
namespace fg {

void FGExecutor::init(Device& device) {
    m_device = &device;
}

void FGExecutor::destroy() {
    m_texturePool.clear();
    m_bufferPool.clear();
    m_persistentState.clear();
    if (m_timestampPool && m_device) {
        vkDestroyQueryPool(m_device->device(), m_timestampPool, nullptr);
        m_timestampPool = VK_NULL_HANDLE;
    }
}

void FGExecutor::initTimestamps(Device& d, uint32_t maxPasses) {
    m_maxTsPasses = maxPasses;
    m_passGpuMs.resize(maxPasses, 0.0f);
    VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = maxPasses * 2;
    VK_CHECK(vkCreateQueryPool(d.device(), &ci, nullptr, &m_timestampPool));
}

void FGExecutor::execute(VkCommandBuffer cmd,
                          FGCompiler::CompiledGraph& compiled,
                          const FGResources& viewCache) {
    // 0. 恢复上帧持久化的 barrier 状态 + 读回上帧 GPU timestamp
    restoreResourceStates(compiled.resources);
    readbackTimestamps();

    // 0b. 重置本帧 timestamp 池
    if (m_timestampPool) {
        vkCmdResetQueryPool(cmd, m_timestampPool, 0, m_maxTsPasses * 2);
    }

    // 1. 分配别名组
    for (auto& group : compiled.aliasGroups) {
        allocateAliasGroup(group, compiled.resources);
    }

    // 2. 遍历执行 pass
    uint32_t tsIdx = 0;
    for (auto* pass : compiled.passOrder) {
        if (!pass || pass->culled) continue;

        if (m_timestampPool && tsIdx < m_maxTsPasses) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                 m_timestampPool, tsIdx * 2);
        }

        // 2a. 插入 barrier：
        if (m_autoBarriers && !pass->usesManualBarriers) {
            emitBarriers(cmd, *pass, compiled.resources, viewCache);
        }

        // 2b. 执行 pass
        if (pass->execute) {
            pass->execute(cmd, viewCache);
        }

        if (m_timestampPool && tsIdx < m_maxTsPasses) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                 m_timestampPool, tsIdx * 2 + 1);
            tsIdx++;
        }

        // 2c. 更新资源状态
        if (!pass->usesManualBarriers) {
            updateResourceStates(*pass, compiled.resources);
        } else {
            // 手动 pass：记录 lastWriter + stage/access。
            // 若声明了 exitLayout，同步更新 tracked layout（避免后续 auto-barrier
            // 被迫使用 UNDEFINED oldLayout 丢弃数据）。
            for (auto& ref : pass->reads) {
                if (ref.resource) {
                    ref.resource->state.lastWriter = const_cast<FGPassNode*>(pass);
                    ref.resource->state.stage  = ref.stages;
                    ref.resource->state.access = ref.access;
                    if (ref.exitLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
                        ref.resource->state.layout = ref.exitLayout;
                    }
                }
            }
            for (auto& ref : pass->writes) {
                if (ref.resource) {
                    ref.resource->state.lastWriter = const_cast<FGPassNode*>(pass);
                    ref.resource->state.stage  = ref.stages;
                    ref.resource->state.access = ref.access;
                    if (ref.exitLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
                        ref.resource->state.layout = ref.exitLayout;
                    }
                }
            }
        }

        // 记录本 pass 执行后的资源布局（用于调试可视化）
        if (m_debug) {
            uint32_t pi = pass->topologicalIndex;
            for (auto* res : compiled.resources) {
                if (!res || res->desc.type != FGResourceType::Texture) continue;
                bool hasBarrier = (m_autoBarriers && !pass->usesManualBarriers);
                m_debug->recordLayout(res->handle,
                    res->desc.debugName ? res->desc.debugName : "?",
                    res->desc.texture.format,
                    pi, res->state.layout, res->state.access, hasBarrier);
            }
        }
    }

    m_tsCount = tsIdx;

    // 3. 保存状态供下帧恢复
    saveResourceStates(compiled.resources);

    // 4. 回收
    recycleUnused(kRecycleFrames);

    ++m_currentFrame;
}

// ════════════════════════════════════════════════════════════════
// RHI 执行路径（通过 nativeHandle() 桥接到现有 Vk 路径）
// ════════════════════════════════════════════════════════════════
void FGExecutor::executeRHI(rhi::RHICommandBuffer& rhiCmd,
                            FGCompiler::CompiledGraph& compiled,
                            const FGResources& viewCache) {
    // 从 RHI 命令缓冲区提取原生 VkCommandBuffer，委托给现有 execute()
    VkCommandBuffer vkCmd = (VkCommandBuffer)(uintptr_t)rhiCmd.nativeHandle();
    execute(vkCmd, compiled, viewCache);
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

        // 获取屏障用的 VkImage：托管资源用 physicalTexture，导入资源用 importedImage
        VkImage barrierImage = VK_NULL_HANDLE;
        if (res->desc.type == FGResourceType::Texture) {
            if (res->physicalTexture) {
                barrierImage = res->physicalTexture->image();
            } else if (res->importedImage) {
                barrierImage = res->importedImage;
            }
        }

        if (barrierImage) {
            VkImageLayout currentLayout = res->state.layout;
            VkImageLayout targetLayout = ref.requiredLayout;

            // 若上一 writer 是手动 pass，tracked state 不可信 → 使用 UNDEFINED
            bool prevManual = res->state.lastWriter &&
                              res->state.lastWriter->usesManualBarriers;

            if (currentLayout == targetLayout &&
                res->state.access == ref.access &&
                !prevManual) {
                return;  // 无需 barrier
            }

            // oldLayout 决策：
            // - 非 manual→auto：使用 tracked currentLayout（精确，数据保留）
            // - manual→auto + exitLayout 已知：使用 tracked layout（manual pass 声明了退出布局）
            // - manual→auto + exitLayout 未知：回退 UNDEFINED（布局安全，但可能丢数据）
            VkImageLayout oldLayout;
            if (!prevManual) {
                oldLayout = currentLayout;        // 精确 oldLayout，数据保留
            } else if (currentLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
                oldLayout = currentLayout;        // exitLayout 已声明，数据保留
            } else {
                oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;  // 回退：布局合法，数据可能丢失
            }
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask  = res->state.stage;
            b.srcAccessMask = res->state.access;
            b.dstStageMask  = ref.stages;
            b.dstAccessMask = ref.access;
            b.oldLayout = oldLayout;
            b.newLayout = targetLayout;
            b.image = barrierImage;
            b.subresourceRange = {
                static_cast<VkImageAspectFlags>(
                    (res->desc.texture.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
                        ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
                0, res->desc.texture.mipLevels, 0, res->desc.texture.arrayLayers
            };
            imgBarriers.push_back(b);
        }

        if (res->desc.type == FGResourceType::Buffer && res->physicalBuffer) {
            bool prevManual = res->state.lastWriter &&
                              res->state.lastWriter->usesManualBarriers;
            if (res->state.access == ref.access && !prevManual) return;

            VkBufferMemoryBarrier2 b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
            // manual→auto 交接：若手动 pass 未声明退出状态，回退到安全默认值
            if (prevManual && res->state.stage == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) {
                b.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                b.srcAccessMask = 0;
            } else {
                b.srcStageMask  = res->state.stage;
                b.srcAccessMask = res->state.access;
            }
            b.dstStageMask  = ref.stages;
            b.dstAccessMask = ref.access;
            b.buffer = res->physicalBuffer->handle();
            b.offset = 0;
            b.size = VK_WHOLE_SIZE;
            bufBarriers.push_back(b);
        }
    };

    // 收集同一资源同时出现在 reads 和 writes 中的情况（readWrite 模式），
    // 合并为单个 barrier 避免重复
    std::unordered_set<FGResourceNode*> merged;
    for (auto& wref : pass.writes) {
        if (!wref.resource) continue;
        for (auto& rref : pass.reads) {
            if (rref.resource == wref.resource) {
                merged.insert(wref.resource);

                // 为合并的 read+write 发射单个 barrier：GENERAL + R/W access
                VkImageLayout targetLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkAccessFlags2 access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                VkPipelineStageFlags2 stages = wref.stages;

                auto* res = wref.resource;
                VkImageLayout currentLayout = res->state.layout;
                bool prevManual = res->state.lastWriter && res->state.lastWriter->usesManualBarriers;
                if (currentLayout != targetLayout || res->state.access != access || prevManual) {
                    VkImageLayout oldLayout = (!prevManual || currentLayout != VK_IMAGE_LAYOUT_UNDEFINED)
                        ? currentLayout : VK_IMAGE_LAYOUT_UNDEFINED;
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = res->state.stage; b.srcAccessMask = res->state.access;
                    b.dstStageMask = stages; b.dstAccessMask = access;
                    b.oldLayout = oldLayout; b.newLayout = targetLayout;
                    b.image = res->physicalTexture ? res->physicalTexture->image() : res->importedImage;
                    b.subresourceRange = {static_cast<VkImageAspectFlags>(
                        (res->desc.texture.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
                            ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
                        0, res->desc.texture.mipLevels, 0, res->desc.texture.arrayLayers};
                    imgBarriers.push_back(b);
                }
                break;
            }
        }
    }

    for (auto& ref : pass.reads) {
        if (merged.count(ref.resource)) continue;
        addBarrier(ref);
    }
    for (auto& ref : pass.writes) {
        if (merged.count(ref.resource)) continue;
        addBarrier(ref);
    }

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
        // barrier 已将 layout 过渡到 target，必须跟踪
        ref.resource->state.layout = ref.requiredLayout;
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
    // 回收长时间未用的纹理池资源
    // 注意：PooledTexture/PooledBuffer 持有 Image/Buffer 值对象，
    // erase 时自动调用析构函数销毁 GPU 资源
    m_texturePool.erase(
        std::remove_if(m_texturePool.begin(), m_texturePool.end(),
            [&](PooledTexture& pt) {
                if (!pt.inUse && (m_currentFrame - pt.lastUsedFrame) > threshold) {
                    // Image 析构函数随 PooledTexture 销毁自动释放 VkImage/VkImageView/VMA
                    return true;
                }
                pt.inUse = false;  // 本帧使用的标记复位，下帧复用
                return false;
            }),
        m_texturePool.end());

    // 回收长时间未用的 Buffer 池资源
    m_bufferPool.erase(
        std::remove_if(m_bufferPool.begin(), m_bufferPool.end(),
            [&](PooledBuffer& pb) {
                if (!pb.inUse && (m_currentFrame - pb.lastUsedFrame) > threshold) {
                    // Buffer 析构函数随 PooledBuffer 销毁自动释放 VkBuffer/VMA
                    return true;
                }
                pb.inUse = false;
                return false;
            }),
        m_bufferPool.end());
}

// ---- 跨帧状态持久化 ----

void FGExecutor::saveResourceStates(const std::vector<FGResourceNode*>& resources) {
    for (auto* res : resources) {
        if (!res || !res->desc.debugName) continue;
        if (res->state.layout == VK_IMAGE_LAYOUT_UNDEFINED) continue;
        m_persistentState[res->desc.debugName] = res->state;
    }
}

void FGExecutor::readbackTimestamps() {
    if (!m_timestampPool || m_tsCount == 0) return;

    uint32_t count = m_tsCount * 2;
    std::vector<uint64_t> buf(count * 2);  // value + availability pairs
    VkResult r = vkGetQueryPoolResults(m_device->device(), m_timestampPool,
        0, count, buf.size() * sizeof(uint64_t), buf.data(), 2 * sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);

    if (r != VK_SUCCESS && r != VK_NOT_READY) return;

    float period = m_device->timestampPeriod() * 1e-6f;  // ns → ms
    for (uint32_t i = 0; i < m_tsCount; ++i) {
        uint64_t startVal   = buf[i * 4];
        uint64_t startAvail = buf[i * 4 + 1];
        uint64_t endVal     = buf[i * 4 + 2];
        uint64_t endAvail   = buf[i * 4 + 3];
        if (startAvail && endAvail && endVal > startVal) {
            m_passGpuMs[i] = float(endVal - startVal) * period;
        } else {
            m_passGpuMs[i] = 0.0f;  // 未就绪
        }
    }
}

void FGExecutor::restoreResourceStates(std::vector<FGResourceNode*>& resources) {
    if (m_persistentState.empty()) return;
    for (auto* res : resources) {
        if (!res || !res->desc.debugName) continue;
        auto it = m_persistentState.find(res->desc.debugName);
        if (it != m_persistentState.end()) {
            res->state = it->second;
        }
    }
}

// ============================================================
// 静态 Layout/Access/Stage 推导
// ============================================================

VkImageLayout FGExecutor::derivedLayout(FGPassType passType,
                                         VkImageUsageFlags usage,
                                         bool isWrite) {
    if (!isWrite) return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Compute/RayTracing 写不使用 attachment layout，统一用 GENERAL
    bool isShaderWrite = (passType == FGPassType::Compute ||
                          passType == FGPassType::RayTracing ||
                          passType == FGPassType::MeshShading);

    if (!isShaderWrite) {
        if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
            return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
            return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    }
    return VK_IMAGE_LAYOUT_GENERAL;
}

VkAccessFlags2 FGExecutor::derivedAccess(FGPassType passType,
                                          VkImageUsageFlags usage,
                                          bool isWrite,
                                          bool isReadWrite) {

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

    // 写操作：Compute/RT/Mesh 不使用 attachment access
    bool isShaderWrite = (passType == FGPassType::Compute ||
                          passType == FGPassType::RayTracing ||
                          passType == FGPassType::MeshShading);

    if (!isShaderWrite) {
        if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
            return VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
            return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
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
