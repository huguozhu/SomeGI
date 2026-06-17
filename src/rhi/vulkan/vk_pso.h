// rhi/vulkan/vk_pso.h
#pragma once
#include "../base/pipeline_state.h"
#include "vk_device.h"
#include <vulkan/vulkan.h>
#include <vector>

namespace somegi {
namespace rhi {

class VkRHIPipelineState : public RHIPipelineState {
public:
    // 工厂方法
    static std::unique_ptr<RHIPipelineState> createGraphics(VkRHIDevice& device, const GraphicsPSODesc& desc);
    static std::unique_ptr<RHIPipelineState> createCompute(VkRHIDevice& device, const ComputePSODesc& desc);
    static std::unique_ptr<RHIPipelineState> createRayTracing(VkRHIDevice& device, const RayTracingPSODesc& desc);

    ~VkRHIPipelineState() override;
    void* nativeHandle() const override { return (void*)m_pipeline; }

    // 内部使用（Vulkan 后端命令缓冲区需要）
    VkPipelineLayout layout() const { return m_pipelineLayout; }
    VkPipelineBindPoint bindPoint() const { return m_bindPoint; }

private:
    VkRHIDevice& m_device;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipelineBindPoint m_bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    VkRHIPipelineState(VkRHIDevice& d) : m_device(d) {}
};

} // namespace rhi
} // namespace somegi
