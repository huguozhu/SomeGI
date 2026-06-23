// src/renderer/fg/fg_executor.h
#pragma once
#include "fg_common.h"
#include "fg_compiler.h"
#include "fg_resource_node.h"
#include "fg_debug.h"
#include "core/image.h"
#include "core/buffer.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <memory>

namespace somegi {

class Device;

namespace rhi {
class RHICommandBuffer;
class RHIQueryPool;
class RHIDevice;
}

namespace fg {

struct FGPassNode;
struct FGResourceNode;
class FGResources;

// ============================================================
// FGExecutor: 帧图执行器
//
// 职责：
//   1. 为每个 AliasGroup 分配/复用物理 Image/Buffer
//   2. 遍历 passOrder，为每个 pass 自动插入 barrier
//   3. 调用 pass.execute() 录制用户命令
//   4. 更新资源 Barrier 状态
//   5. 回收长期未用的池资源
// ============================================================
class FGExecutor {
public:
    FGExecutor();
    ~FGExecutor();
    void init(Device& device);
    void destroy();

    // 设置 RHI 设备（供非拥有型包装和 query pool 创建使用）
    void setRHIDevice(rhi::RHIDevice& rhiDevice) { m_rhiDevice = &rhiDevice; }

    // 绑定 FGDebug 用于资源布局时间线记录
    void setDebug(struct FGDebug* debug) { m_debug = debug; }

    // GPU timestamp 池（可选，调用 initTimestamps 后启用）
    void initTimestamps(rhi::RHIDevice& d, uint32_t maxPasses);
    const std::vector<float>& passGpuMs() const { return m_passGpuMs; }

    // 执行编译后的图（RHI 路径）
    void execute(rhi::RHICommandBuffer& cmd,
                 FGCompiler::CompiledGraph& compiled,
                 const FGResources& viewCache);

    // 控制是否自动插入 barrier（默认 false，pass 内部管理 barrier）
    void setAutoBarriers(bool enabled) { m_autoBarriers = enabled; }
    bool autoBarriers() const { return m_autoBarriers; }

    // 静态方法：根据 pass 类型和资源 usage 推导 layout/access/stage
    static VkImageLayout derivedLayout(FGPassType passType,
                                        VkImageUsageFlags usage,
                                        bool isWrite);
    static VkAccessFlags2 derivedAccess(FGPassType passType,
                                         VkImageUsageFlags usage,
                                         bool isWrite,
                                         bool isReadWrite);
    static VkPipelineStageFlags2 derivedStage(FGPassType passType,
                                               VkImageUsageFlags usage,
                                               bool isWrite);

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    uint64_t m_currentFrame = 0;
    bool m_autoBarriers = false;  // 默认关闭，pass 内部管理 barrier

    // ---- 纹理资源池 ----
    struct PooledTexture {
        Image image;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent3D extent{};
        uint64_t lastUsedFrame = 0;
        bool inUse = false;
    };
    std::vector<PooledTexture> m_texturePool;

    // ---- Buffer 资源池 ----
    struct PooledBuffer {
        Buffer buffer;
        VkDeviceSize size = 0;
        uint64_t lastUsedFrame = 0;
        bool inUse = false;
    };
    std::vector<PooledBuffer> m_bufferPool;

    // ---- 内部方法 ----

    // 为别名组分配物理资源
    void allocateAliasGroup(const FGCompiler::AliasGroup& group,
                            std::vector<FGResourceNode*>& resources);

    // 分配单个托管纹理（池中取或新建）
    Image* allocateTexture(const FGResourceDesc& desc);

    // 分配单个托管 Buffer（池中取或新建）
    Buffer* allocateBuffer(const FGResourceDesc& desc);

    // 为 pass 插入前置 barrier（RHI 路径）
    void emitBarriers(rhi::RHICommandBuffer& cmd,
                      const FGPassNode& pass,
                      std::vector<FGResourceNode*>& resources,
                      const FGResources& viewCache);

    // 更新 pass 执行后的资源状态
    void updateResourceStates(const FGPassNode& pass,
                              std::vector<FGResourceNode*>& resources);

    // 回收超过 recycleFrameThreshold 帧未用的池资源
    void recycleUnused(uint64_t threshold);
    static constexpr uint32_t kRecycleFrames = 30;  // 30 帧不用则回收

    // 跨帧持久化 barrier 状态（reset() 会清空资源节点，但 GPU 布局不变）
    void saveResourceStates(const std::vector<FGResourceNode*>& resources);
    void restoreResourceStates(std::vector<FGResourceNode*>& resources);

    std::unordered_map<std::string, FGResourceNode::BarrierState> m_persistentState;

    // ---- 资源布局时间线记录 ----
    struct FGDebug* m_debug = nullptr;

    // ---- GPU Timestamp (双缓冲，每 frameInFlight 独立 query 段) ----
    std::unique_ptr<rhi::RHIQueryPool> m_timestampPool;
    uint32_t m_maxTsPasses = 0;
    uint32_t m_tsSlotCount = 0;       // 每 slot 的 query 数 (maxTsPasses * 2)
    uint32_t m_tsCount[2] = {};       // 每 slot 上次写入的 pass 数
    uint32_t m_tsSlot = 0;            // 本帧写入的 slot 索引 (0/1)
    std::vector<float> m_passGpuMs;   // 上帧每 pass 的 GPU 耗时 (ms)

    void readbackTimestamps();
};

} // namespace fg
} // namespace somegi
