// rhi/vulkan/vk_command.h
#pragma once
#include "../command_buffer.h"
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

    void bindPipelineState(const RHIPipelineState& pso) override;
    void bindDescriptorSet(uint32_t slot, const RHIDescriptorSet& set) override;
    void pushConstants(ShaderStage stage, const void* data, uint32_t size, uint32_t offset = 0) override;

    void bindVertexBuffer(const RHIBuffer& buffer, uint64_t offset = 0) override;
    void bindIndexBuffer(const RHIBuffer& buffer, uint64_t offset = 0, bool uint16 = true) override;

    void draw(uint32_t vertexCount, uint32_t firstVertex = 0, uint32_t firstInstance = 0) override;
    void drawIndexed(uint32_t indexCount, uint32_t firstIndex = 0, int32_t vertexOffset = 0) override;
    void drawIndexedIndirect(const RHIBuffer& buffer, uint64_t offset, uint32_t drawCount, uint32_t stride) override;
    void drawMeshTasks(uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1) override;

    void dispatch(uint32_t groupX, uint32_t groupY = 1, uint32_t groupZ = 1) override;
    void dispatchIndirect(const RHIBuffer& buffer, uint64_t offset) override;

    void copyBuffer(const RHIBuffer& src, const RHIBuffer& dst, uint64_t size, uint64_t srcOffset = 0, uint64_t dstOffset = 0) override;
    void copyTexture(const RHITexture& src, const RHITexture& dst) override;
    void clearColor(const RHITextureView& view, float r, float g, float b, float a) override;

    void textureBarrier(const RHITexture& tex, TextureLayout oldLayout, TextureLayout newLayout) override;
    void globalBarrier() override;

    void beginRendering(const RHITextureView* colorViews, uint32_t colorCount,
                        const RHITextureView* depthView, uint32_t width, uint32_t height) override;
    void endRendering() override;

    void writeTimestamp(const RHIQueryPool& pool, uint32_t index) override;
    void resetQueryPool(const RHIQueryPool& pool, uint32_t first, uint32_t count) override;

    void* nativeHandle() const override { return (void*)m_cmd; }
    VkCommandBuffer vkCmd() const { return m_cmd; }

private:
    VkRHIDevice& m_device;
    VkCommandBuffer m_cmd;
};

} // namespace rhi
} // namespace somegi
