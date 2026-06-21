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
#include "rhi/base/command_buffer.h"
#include "rhi/base/device.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include <cstdio>

namespace somegi {
namespace fg {

// ════════════════════════════════════════════════════════════════
// Vk → RHI 映射辅助函数（局部作用域）
// ════════════════════════════════════════════════════════════════

static rhi::TextureLayout toRhiLayout(VkImageLayout l) {
    switch (l) {
        case VK_IMAGE_LAYOUT_UNDEFINED:                   return rhi::TextureLayout::Undefined;
        case VK_IMAGE_LAYOUT_GENERAL:                     return rhi::TextureLayout::General;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:    return rhi::TextureLayout::ShaderReadOnly;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:    return rhi::TextureLayout::ColorAttachment;
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: return rhi::TextureLayout::DepthAttachment;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:        return rhi::TextureLayout::TransferSrc;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:        return rhi::TextureLayout::TransferDst;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:             return rhi::TextureLayout::Present;
        // Depth-stencil read-only — 保守映射为 ShaderReadOnly（可被采样读取）
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
        case VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL:
            return rhi::TextureLayout::ShaderReadOnly;
        default: return rhi::TextureLayout::Undefined;
    }
}

static rhi::PipelineStage toRhiStage(VkPipelineStageFlags2 s) {
    if (s == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
        return rhi::PipelineStage::AllCommands;
    if (s == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT)
        return rhi::PipelineStage::None;  // TOP_OF_PIPE → 无需等待任何阶段

    uint32_t r = 0;
    if (s & VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT)               r |= (uint32_t)rhi::PipelineStage::VertexShader;
    if (s & VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)             r |= (uint32_t)rhi::PipelineStage::FragmentShader;
    if (s & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)              r |= (uint32_t)rhi::PipelineStage::ComputeShader;
    if (s & VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT)             r |= (uint32_t)rhi::PipelineStage::MeshShader;
    if (s & VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT)             r |= (uint32_t)rhi::PipelineStage::TaskShader;
    if (s & VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR)      r |= (uint32_t)rhi::PipelineStage::RayTracingShader;
    if (s & VK_PIPELINE_STAGE_2_COPY_BIT)                        r |= (uint32_t)rhi::PipelineStage::Transfer;
    // Color attachment / depth tests → FragmentShader（最近似阶段）
    if (s & VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)     r |= (uint32_t)rhi::PipelineStage::FragmentShader;
    if (s & (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
             VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT))      r |= (uint32_t)rhi::PipelineStage::FragmentShader;
    if (s & VK_PIPELINE_STAGE_2_TRANSFER_BIT)                    r |= (uint32_t)rhi::PipelineStage::Transfer;
    if (r == 0) r = (uint32_t)rhi::PipelineStage::AllCommands;
    return (rhi::PipelineStage)r;
}

static rhi::BufferAccess toRhiBufferAccess(VkAccessFlags2 a) {
    if (a == 0) return rhi::BufferAccess::None;

    uint32_t r = 0;
    if (a & VK_ACCESS_2_UNIFORM_READ_BIT)           r |= (uint32_t)rhi::BufferAccess::UniformRead;
    if (a & VK_ACCESS_2_SHADER_STORAGE_READ_BIT)    r |= (uint32_t)rhi::BufferAccess::StorageRead;
    if (a & VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)   r |= (uint32_t)rhi::BufferAccess::StorageWrite;
    if (a & VK_ACCESS_2_INDEX_READ_BIT)             r |= (uint32_t)rhi::BufferAccess::IndexRead;
    if (a & VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT)  r |= (uint32_t)rhi::BufferAccess::VertexRead;
    if (a & VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT)  r |= (uint32_t)rhi::BufferAccess::IndirectRead;
    if (a & VK_ACCESS_2_TRANSFER_READ_BIT)          r |= (uint32_t)rhi::BufferAccess::TransferRead;
    if (a & VK_ACCESS_2_TRANSFER_WRITE_BIT)         r |= (uint32_t)rhi::BufferAccess::TransferWrite;
    // 图像专用 access 无 RHI BufferAccess 对等项 → 保守 fallback
    if (r == 0) r = (uint32_t)rhi::BufferAccess::StorageRead | (uint32_t)rhi::BufferAccess::StorageWrite;
    return (rhi::BufferAccess)r;
}

// ════════════════════════════════════════════════════════════════
// Lifecycle
// ════════════════════════════════════════════════════════════════

FGExecutor::FGExecutor() = default;
FGExecutor::~FGExecutor() = default;

void FGExecutor::init(Device& device) {
    m_device = &device;
}

void FGExecutor::destroy() {
    m_texturePool.clear();
    m_bufferPool.clear();
    m_persistentState.clear();
    m_timestampPool.reset();  // unique_ptr 自动析构销毁 GPU 资源
}

void FGExecutor::initTimestamps(rhi::RHIDevice& d, uint32_t maxPasses) {
    m_rhiDevice = &d;
    m_maxTsPasses = maxPasses;
    m_tsSlotCount = maxPasses * 2;
    m_passGpuMs.resize(maxPasses, 0.0f);
    // 双缓冲：kFramesInFlight=2，每 slot 一段独立 query 范围
    m_timestampPool = d.createQueryPool(m_tsSlotCount * 2);
}

// ════════════════════════════════════════════════════════════════
// execute — RHI 主路径
// ════════════════════════════════════════════════════════════════

void FGExecutor::execute(rhi::RHICommandBuffer& cmd,
                          FGCompiler::CompiledGraph& compiled,
                          const FGResources& viewCache) {
    // 0. 恢复上帧持久化的 barrier 状态 + 读回上帧 GPU timestamp
    restoreResourceStates(compiled.resources);
    readbackTimestamps();

    // 0b. 选择本帧写入的 slot 并只重置本帧段
    m_tsSlot = m_currentFrame % 2;
    uint32_t tsBase = m_tsSlot * m_tsSlotCount;
    if (m_timestampPool) {
        cmd.resetQueryPool(*m_timestampPool, tsBase, m_tsSlotCount);
    }

    // 1. 分配别名组
    for (auto& group : compiled.aliasGroups) {
        allocateAliasGroup(group, compiled.resources);
    }

    // 2. 遍历执行 pass
    //    提取原生 VkCommandBuffer 用于 pass 回调（尚未迁移到 RHI 的遗留 pass）
    VkCommandBuffer vkCmd = (VkCommandBuffer)(uintptr_t)cmd.nativeHandle();

    uint32_t tsIdx = 0;
    for (auto* pass : compiled.passOrder) {
        if (!pass || pass->culled) continue;

        if (m_timestampPool && tsIdx < m_maxTsPasses) {
            cmd.writeTimestamp(*m_timestampPool, tsBase + tsIdx * 2);
        }

        // 2a. 插入 barrier：
        if (m_autoBarriers && !pass->usesManualBarriers) {
            emitBarriers(cmd, *pass, compiled.resources, viewCache);
        }

        // 2b. 执行 pass（回退到 VkCommandBuffer 桥接）
        if (pass->execute) {
            pass->execute(vkCmd, viewCache);
        }

        if (m_timestampPool && tsIdx < m_maxTsPasses) {
            cmd.writeTimestamp(*m_timestampPool, tsBase + tsIdx * 2 + 1);
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

    m_tsCount[m_tsSlot] = tsIdx;

    // 3. 保存状态供下帧恢复
    saveResourceStates(compiled.resources);

    // 4. 回收
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

// ════════════════════════════════════════════════════════════════
// Barrier 插入（RHI 路径）
// ════════════════════════════════════════════════════════════════

void FGExecutor::emitBarriers(rhi::RHICommandBuffer& cmd,
                               const FGPassNode& pass,
                               std::vector<FGResourceNode*>& resources,
                               const FGResources& viewCache) {
    (void)resources;
    (void)viewCache;

    // 收集需要合并的 read+write 资源（同一 handle 同时出现在 reads 和 writes）
    std::unordered_set<FGResourceNode*> merged;
    for (auto& wref : pass.writes) {
        if (!wref.resource) continue;
        for (auto& rref : pass.reads) {
            if (rref.resource == wref.resource) {
                merged.insert(wref.resource);

                auto* res = wref.resource;
                VkImageLayout currentLayout = res->state.layout;

                // Read+write 合并：过渡到 GENERAL + 完整存储读写
                rhi::TextureLayout targetLayout = rhi::TextureLayout::General;
                bool prevManual = res->state.lastWriter && res->state.lastWriter->usesManualBarriers;
                if (currentLayout != VK_IMAGE_LAYOUT_GENERAL || res->state.access !=
                    (VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) || prevManual) {

                    // 为合并的 read+write 发射单个 barrier
                    if (res->desc.type == FGResourceType::Texture) {
                        // 获取屏障用的 VkImage
                        VkImage img = res->physicalTexture
                            ? res->physicalTexture->image()
                            : res->importedImage;
                        if (img && m_rhiDevice) {
                            auto* vkDev = static_cast<rhi::VkRHIDevice*>(m_rhiDevice);
                            auto tex = rhi::VkRHITexture::createNonOwning(
                                *vkDev, img,
                                rhi::toRhiFormat(res->desc.texture.format),
                                res->desc.texture.extent.width,
                                res->desc.texture.extent.height,
                                res->desc.texture.mipLevels);
                            // oldLayout 使用 UNDEFINED（与原有行为一致，不假设旧布局）
                            cmd.textureBarrier(*tex,
                                rhi::TextureLayout::Undefined, targetLayout);
                        }
                    } else if (res->desc.type == FGResourceType::Buffer && res->physicalBuffer) {
                        VkBuffer buf = res->physicalBuffer->handle();
                        if (buf && m_rhiDevice) {
                            auto* vkDev = static_cast<rhi::VkRHIDevice*>(m_rhiDevice);
                            auto rhiBuf = rhi::VkRHIBuffer::createNonOwning(
                                *vkDev, buf,
                                res->physicalBuffer->size());

                            // manual→auto 交接安全回退
                            uint32_t srcStageRaw = (uint32_t)rhi::PipelineStage::ComputeShader;
                            uint32_t srcAccessRaw = 0;
                            if (prevManual && res->state.stage == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) {
                                srcStageRaw = (uint32_t)rhi::PipelineStage::AllCommands;
                                srcAccessRaw = 0;
                            } else {
                                srcStageRaw = (uint32_t)toRhiStage(res->state.stage);
                                srcAccessRaw = (uint32_t)toRhiBufferAccess(res->state.access);
                            }

                            cmd.bufferBarrier(*rhiBuf,
                                (rhi::PipelineStage)srcStageRaw,
                                rhi::PipelineStage::ComputeShader,  // readWrite → compute shader stage
                                (rhi::BufferAccess)srcAccessRaw,
                                (rhi::BufferAccess)((uint32_t)rhi::BufferAccess::StorageRead | (uint32_t)rhi::BufferAccess::StorageWrite));
                        }
                    }
                }
                break;
            }
        }
    }

    // 辅助 lambda：为单个 ResourceRef 发射 barrier
    auto emitRefBarrier = [&](const FGPassNode::ResourceRef& ref) {
        auto* res = ref.resource;
        if (!res) return;

        if (res->desc.type == FGResourceType::Texture) {
            VkImage img = res->physicalTexture
                ? res->physicalTexture->image()
                : res->importedImage;
            if (!img || !m_rhiDevice) return;

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

            // 注意：手动 pass 可能未声明 exitLayout，导致 tracked state 不可信。
            // 统一使用 UNDEFINED 以避免 oldLayout=currentLayout 的验证错误。
            // （Undefined oldLayout 在 Vulkan 中无条件执行过渡并丢弃内容，总是合法。）
            (void)currentLayout;
            (void)prevManual;
            rhi::TextureLayout oldRhiLayout = rhi::TextureLayout::Undefined;

            auto* vkDev = static_cast<rhi::VkRHIDevice*>(m_rhiDevice);
            auto tex = rhi::VkRHITexture::createNonOwning(
                *vkDev, img,
                rhi::toRhiFormat(res->desc.texture.format),
                res->desc.texture.extent.width,
                res->desc.texture.extent.height,
                res->desc.texture.mipLevels);

            cmd.textureBarrier(*tex, oldRhiLayout, toRhiLayout(targetLayout));
        }

        if (res->desc.type == FGResourceType::Buffer && res->physicalBuffer) {
            VkBuffer buf = res->physicalBuffer->handle();
            if (!buf || !m_rhiDevice) return;

            bool prevManual = res->state.lastWriter &&
                              res->state.lastWriter->usesManualBarriers;
            if (res->state.access == ref.access && !prevManual) return;

            uint32_t srcStageRaw, srcAccessRaw;
            if (prevManual && res->state.stage == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) {
                srcStageRaw  = (uint32_t)rhi::PipelineStage::AllCommands;
                srcAccessRaw = 0;
            } else {
                srcStageRaw  = (uint32_t)toRhiStage(res->state.stage);
                srcAccessRaw = (uint32_t)toRhiBufferAccess(res->state.access);
            }

            auto* vkDev = static_cast<rhi::VkRHIDevice*>(m_rhiDevice);
            auto rhiBuf = rhi::VkRHIBuffer::createNonOwning(
                *vkDev, buf,
                res->physicalBuffer->size());

            cmd.bufferBarrier(*rhiBuf,
                (rhi::PipelineStage)srcStageRaw,
                toRhiStage(ref.stages),
                (rhi::BufferAccess)srcAccessRaw,
                toRhiBufferAccess(ref.access));
        }
    };

    // 遍历 reads 和 writes，跳过已合并的
    for (auto& ref : pass.reads) {
        if (merged.count(ref.resource)) continue;
        emitRefBarrier(ref);
    }
    for (auto& ref : pass.writes) {
        if (merged.count(ref.resource)) continue;
        emitRefBarrier(ref);
    }

    // 注意：RHI textBarrier / bufferBarrier 各自在内部调用 vkCmdPipelineBarrier2，
    // 不再是批量发射。若性能关键，可后续扩展 RHI 批量 barrier 接口。
}

// ---- 状态更新 ----

void FGExecutor::updateResourceStates(const FGPassNode& pass,
                                       std::vector<FGResourceNode*>& resources) {
    (void)resources;

    for (auto& ref : pass.reads) {
        if (!ref.resource) continue;
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
    m_texturePool.erase(
        std::remove_if(m_texturePool.begin(), m_texturePool.end(),
            [&](PooledTexture& pt) {
                if (!pt.inUse && (m_currentFrame - pt.lastUsedFrame) > threshold) {
                    return true;
                }
                pt.inUse = false;
                return false;
            }),
        m_texturePool.end());

    // 回收长时间未用的 Buffer 池资源
    m_bufferPool.erase(
        std::remove_if(m_bufferPool.begin(), m_bufferPool.end(),
            [&](PooledBuffer& pb) {
                if (!pb.inUse && (m_currentFrame - pb.lastUsedFrame) > threshold) {
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

// ════════════════════════════════════════════════════════════════
// Timestamp 读回（RHI 路径）
// ════════════════════════════════════════════════════════════════

void FGExecutor::readbackTimestamps() {
    // 读取上一帧写入的 slot（当前写入 slot 的对侧），其 GPU fence 已由主循环等待
    uint32_t readSlot = 1 - m_tsSlot;
    uint32_t readBase = readSlot * m_tsSlotCount;
    uint32_t tsCount = m_tsCount[readSlot];

    if (!m_timestampPool || tsCount == 0) return;

    uint32_t count = tsCount * 2;
    // 主循环每帧结束时已 vkWaitForFences，query 结果保证就绪
    std::vector<uint64_t> buf(count);
    m_timestampPool->getResults(readBase, count, buf.data());

    // timestampPeriod 来自 RHI DeviceLimits（nanoseconds per tick）
    float period = (m_rhiDevice ? m_rhiDevice->limits().timestampPeriod : 1.0f) * 1e-6f;
    for (uint32_t i = 0; i < tsCount; ++i) {
        uint64_t startVal = buf[i * 2];
        uint64_t endVal   = buf[i * 2 + 1];
        if (endVal > startVal) {
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
