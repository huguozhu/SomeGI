// rhi/vulkan/vk_context.h
#pragma once
#include "../base/context.h"
#include "../base/descriptor.h"
#include "../base/pipeline_state.h"
#include "../base/sampler.h"         // RHISampler, SamplerDesc
#include "../base/shader.h"          // RHIShader, ShaderDesc
#include "../base/command_buffer.h"  // RHIQueryPool
#include "vk_device.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

namespace somegi {
namespace rhi {

class VkRHIDevice;

class VkContext : public RHIContext {
public:
    // 构造时创建命令池、命令缓冲区、fence（kFramesInFlight 组）
    VkContext(VkRHIDevice& device, uint32_t framesInFlight);
    ~VkContext() override;

    // ── 帧生命周期 ──
    RHICommandBuffer& beginFrame(uint32_t frameIndex) override;
    void endFrame(uint32_t frameIndex,
                  const RHISemaphore* waitSemaphore,
                  const RHISemaphore* signalSemaphore,
                  void* externalFence = nullptr) override;
    RHICommandBuffer& commandBuffer(uint32_t frameIndex) override;
    void waitIdle() override;
    RHIDevice& device() override { return m_device; }

    // ── 资源工厂（委托给 VkRHIDevice，统一入口）──
    std::unique_ptr<RHIDescriptorSetLayout> createDescriptorSetLayout(const DescSetLayoutDesc& desc) {
        return m_device.createDescriptorSetLayout(desc);
    }
    std::unique_ptr<RHIDescriptorSet> createDescriptorSet(const RHIDescriptorSetLayout& layout) {
        return m_device.createDescriptorSet(layout);
    }
    std::unique_ptr<RHIPipelineState> createGraphicsPSO(const GraphicsPSODesc& desc) {
        return m_device.createGraphicsPSO(desc);
    }
    std::unique_ptr<RHIPipelineState> createComputePSO(const ComputePSODesc& desc) {
        return m_device.createComputePSO(desc);
    }
    std::unique_ptr<RHIShader> createShader(const ShaderDesc& desc, const void* bytecode, size_t size) {
        return m_device.createShader(desc, bytecode, size);
    }
    std::unique_ptr<RHISampler> createSampler(const SamplerDesc& desc) {
        return m_device.createSampler(desc);
    }
    std::unique_ptr<RHIQueryPool> createQueryPool(uint32_t count) {
        return m_device.createQueryPool(count);
    }

    // Vulkan 专用访问器（过渡期：ImGui 等需要原生句柄）
    VkCommandBuffer vkCommandBuffer(uint32_t frameIndex) const;
    VkCommandPool vkCommandPool() const { return m_pool; }
    VkRHIDevice& vkDevice() { return m_device; }

private:
    VkRHIDevice& m_device;
    uint32_t m_framesInFlight;
    VkCommandPool m_pool = VK_NULL_HANDLE;

    struct FrameResources {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool fenceSignaled = true;  // 首帧 fence 已 signaled
    };
    std::vector<FrameResources> m_frames;

    // RHI 命令缓冲区包装（临时对象，valid until next beginFrame）
    std::unique_ptr<RHICommandBuffer> m_tempCmd;

    // 内部分配 + 清理
    void createResources();
    void destroyResources();
};

} // namespace rhi
} // namespace somegi
