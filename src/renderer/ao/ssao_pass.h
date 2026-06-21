// SsaoPass —— 屏幕空间环境光遮蔽（M4.1），已迁移到 RHI。
//
// 使用 RHI ComputePSO + RHIDescriptorSet + RHICommandBuffer 替代原始 Vulkan API。
// 保持与 RenderTargets (core::Image) 的兼容：bindFrame 通过非拥有型 RHI 包装
// 桥接 VkImageView → RHITextureView。
#pragma once
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "renderer/core/render_targets.h"
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>  // VkCommandBuffer (兼容重载需要)

namespace somegi {

namespace rhi {
class RHIDevice;
class RHICommandBuffer;
}

class SsaoPass {
public:
    SsaoPass() = default;
    ~SsaoPass();  // 在 .cpp 中定义，RHIDescriptorSet 等在此是完整类型

    void init(rhi::RHIDevice& d);
    void destroy();

    // 绑定 GBuffer 视图（swapchain resize 时重新绑定）
    void bindFrame(const RenderTargets& rt);

    // 录制 SSAO compute dispatch（RHI 路径）
    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt,
                const glm::mat4& proj, const glm::mat4& invProj,
                const glm::mat4& view);


    // ImGui 可调的运行时状态
    bool  enabled      = true;
    float radius       = 0.5f;
    float bias         = 0.025f;
    int   sampleCount  = 16;     // 每像素采样数

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;

    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState>       m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet>        m_set;
};

} // namespace somegi
