// rhi/vulkan/vk_command.h
#pragma once
#include "../base/command_buffer.h"
#include "vk_device.h"
#include <vulkan/vulkan.h>

namespace somegi {
namespace rhi {

class VkRHIDevice;

class VkRHICommandPool : public RHICommandPool {
public:
    static std::unique_ptr<RHICommandPool> create(VkRHIDevice& device);
    ~VkRHICommandPool() override;
    RHICommandBuffer* allocateRaw() override;
    void reset() override;
private:
    VkRHIDevice& m_device;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    VkRHICommandPool(VkRHIDevice& d) : m_device(d) {}
};

class VkRHICommandBuffer : public RHICommandBuffer {
public:
    VkRHICommandBuffer(VkRHIDevice& device, VkCommandBuffer cmd) : m_device(device), m_cmd(cmd) {}
    ~VkRHICommandBuffer() override = default; // pool manages lifecycle

    void begin() override;
    void end() override;
    void reset() override;

    // 动态状态
    void setViewport(float x, float y, float width, float height,
                     float minDepth = 0.0f, float maxDepth = 1.0f) override;
    void setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) override;

    // PSO / 描述符
    void bindPipelineState(const RHIPipelineState& pso) override;
    void bindDescriptorSet(uint32_t slot, const RHIDescriptorSet& set) override;
    void bindDescriptorSets(uint32_t firstSlot, uint32_t count,
                            const RHIDescriptorSet* const* sets) override;
    void pushConstants(ShaderStage stage, const void* data, uint32_t size, uint32_t offset = 0) override;

    // 顶点 / 索引
    void bindVertexBuffer(const RHIBuffer& buffer, uint64_t offset = 0) override;
    void bindIndexBuffer(const RHIBuffer& buffer, uint64_t offset = 0, bool uint16 = true) override;

    // Draw
    void draw(uint32_t vertexCount, uint32_t firstVertex = 0, uint32_t firstInstance = 0) override;
    void drawIndexed(uint32_t indexCount, uint32_t firstIndex = 0, int32_t vertexOffset = 0) override;
    void drawIndirect(const RHIBuffer& buffer, uint64_t offset, uint32_t drawCount, uint32_t stride) override;
    void drawIndexedIndirect(const RHIBuffer& buffer, uint64_t offset, uint32_t drawCount, uint32_t stride) override;
    void drawMeshTasks(uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1) override;
    void drawMeshTasksIndirect(const RHIBuffer& buffer, uint64_t offset,
                               uint32_t drawCount, uint32_t stride) override;

    // Dispatch
    void dispatch(uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1) override;
    void dispatchIndirect(const RHIBuffer& buffer, uint64_t offset) override;

    // 复制 / 清除
    void copyBuffer(const RHIBuffer& src, const RHIBuffer& dst, uint64_t size,
                    uint64_t srcOffset = 0, uint64_t dstOffset = 0) override;
    void copyTexture(const RHITexture& src, const RHITexture& dst) override;
    void clearColor(const RHITexture& tex, float r, float g, float b, float a) override;
    void clearDepth(const RHITexture& tex, float depth, uint32_t stencil = 0) override;

    // Barrier
    void textureBarrier(const RHITexture& tex, TextureLayout oldLayout, TextureLayout newLayout) override;
    void bufferBarrier(const RHIBuffer& buf,
                       PipelineStage srcStage, PipelineStage dstStage,
                       BufferAccess srcAccess, BufferAccess dstAccess) override;
    void globalBarrier() override;

    // 渲染通道
    void beginRendering(const RHITextureView* colorViews, uint32_t colorCount,
                        const RHITextureView* depthView, uint32_t width, uint32_t height,
                        bool loadOnly = false) override;
    void endRendering() override;

    // 时间戳
    void writeTimestamp(const RHIQueryPool& pool, uint32_t index) override;
    void resetQueryPool(const RHIQueryPool& pool, uint32_t first, uint32_t count) override;

    void* nativeHandle() const override { return (void*)m_cmd; }
    VkCommandBuffer vkCmd() const { return m_cmd; }

private:
    VkRHIDevice& m_device;
    VkCommandBuffer m_cmd;
    VkPipelineLayout m_boundLayout = VK_NULL_HANDLE;  // 最近一次 bindPipelineState 的布局
    VkPipelineBindPoint m_bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
};

} // namespace rhi
} // namespace somegi
