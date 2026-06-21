// rhi/base/context.h — RHI 命令录制上下文抽象
#pragma once
#include "common.h"
#include <cstdint>

namespace somegi {
namespace rhi {

class RHICommandBuffer;
class RHIFence;
class RHISemaphore;
class RHIDevice;

// ════════════════════════════════════════════════════════════════
// RHIContext: 管理命令池、命令缓冲区、帧同步
//
// 每个 Context 持有 kFramesInFlight 组资源（cmd buffer + fence），
// 调用方通过 beginFrame(n) 获取当帧的就绪命令缓冲区。
//
// 典型用法：
//   auto ctx = device->createContext(kFramesInFlight);
//   for each frame:
//     auto& cmd = ctx->beginFrame(frameIndex);  // 自动等待 fence + 重置
//     // … record commands …
//     ctx->submit(cmd, waitSem, signalSem);
//     ctx->endFrame(frameIndex);                // signal fence
// ════════════════════════════════════════════════════════════════
class RHIContext {
public:
    virtual ~RHIContext() = default;

    // 获取第 n 帧的命令缓冲区（自动等待上一轮 fence 并 begin）
    virtual RHICommandBuffer& beginFrame(uint32_t frameIndex) = 0;

    // 结束命令录制，提交到队列，并触发 fence
    virtual void endFrame(uint32_t frameIndex,
                          const RHISemaphore* waitSemaphore,
                          const RHISemaphore* signalSemaphore) = 0;

    // 直接获取已录制的命令缓冲区（用于额外命令，如 ImGui）
    virtual RHICommandBuffer& commandBuffer(uint32_t frameIndex) = 0;

    // 等待设备空闲
    virtual void waitIdle() = 0;

    // 获取底层设备
    virtual RHIDevice& device() = 0;
};

} // namespace rhi
} // namespace somegi
