// rhi/vulkan/vk_pso.h
#pragma once
#include "../pipeline_state.h"
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

    ~VkRHIPipelineState() override;
    void* nativeHandle() const override { return (void*)m_pipeline; }
    VkPipelineLayout layout() const { return m_pipelineLayout; }
private:
    VkRHIDevice& m_device;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkRHIPipelineState(VkRHIDevice& d) : m_device(d) {}
};

} // namespace rhi
} // namespace somegi
