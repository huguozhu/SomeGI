// SsaoPass RHI 实现 — 描述符 set=0:
//   binding 0: gNormalRough (sampled image)
//   binding 1: gDepth       (sampled image, depth aspect)
//   binding 2: gOutAO       (storage image, R8)
// push constant: SsaoPC (3×mat4 + 标量参数)

#include "renderer/ao/ssao_pass.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/base/texture.h"
#include "rhi/vulkan/vk_device.h"     // VkRHIDevice — 用于 createFromFile + createNonOwning
#include "rhi/vulkan/vk_shader.h"     // VkRHIShader::createFromFile
#include "rhi/vulkan/vk_texture.h"    // VkRHITextureView::createNonOwning
#include "rhi/vulkan/vk_command.h"    // VkRHICommandBuffer — VkCmd 包装
#include "core/shader.h"              // shaderDir()
#include <cstring>

namespace somegi {

namespace {
// shader 端 SsaoPC 与此对齐；mat4 必须先于标量保证 16B 对齐
struct SsaoPC {
    glm::mat4 proj;
    glm::mat4 invProj;
    glm::mat4 view;
    uint32_t  outSizeX, outSizeY;
    float     invOutSizeX, invOutSizeY;
    float     radius;
    float     bias;
    uint32_t  sampleCount;
    uint32_t  _pad;
};
static_assert(sizeof(SsaoPC) == 224);
}

SsaoPass::~SsaoPass() = default;  // 必须在 .cpp 中定义（unique_ptr 需要完整类型）

void SsaoPass::init(rhi::RHIDevice& d) {
    m_rhiDevice = &d;

    // ════════════════════════════════════════════════════════════
    // Descriptor Set Layout
    // ════════════════════════════════════════════════════════════
    rhi::DescSetLayoutDesc layoutDesc;
    layoutDesc.debugName = "SSAO";
    layoutDesc.bindings = {
        {0, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},  // gNormalRough
        {1, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},  // gDepth
        {2, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},  // gOutAO
    };
    m_setLayout = d.createDescriptorSetLayout(layoutDesc);

    // ════════════════════════════════════════════════════════════
    // Descriptor Set
    // ════════════════════════════════════════════════════════════
    m_set = d.createDescriptorSet(*m_setLayout);

    // ════════════════════════════════════════════════════════════
    // Compute PSO
    // ════════════════════════════════════════════════════════════
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(d);
    rhi::ShaderDesc shaderDesc;
    shaderDesc.stage = rhi::ShaderStage::Compute;
    shaderDesc.entryPoint = "cs_main";
    auto shader = rhi::VkRHIShader::createFromFile(vkDevice, shaderDesc,
        shaderDir() / "ssao" / "ssao.spv");

    rhi::ComputePSODesc psoDesc;
    psoDesc.debugName = "SSAO";
    psoDesc.computeShader = shader.get();
    psoDesc.descriptorSetLayouts = {m_setLayout.get()};
    psoDesc.pushConstants = {
        {rhi::ShaderStage::Compute, 0, sizeof(SsaoPC)}
    };
    m_pipeline = d.createComputePSO(psoDesc);
}

void SsaoPass::destroy() {
    m_set.reset();
    m_pipeline.reset();
    m_setLayout.reset();
    m_rhiDevice = nullptr;
}

void SsaoPass::bindFrame(const RenderTargets& rt) {
    if (!m_rhiDevice || !m_set) return;
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    // 为非拥有型 RHI 视图包装创建临时对象
    auto nrView = rhi::VkRHITextureView::createNonOwning(vkDevice, rt.gNormalRough.view());
    auto dpView = rhi::VkRHITextureView::createNonOwning(vkDevice, rt.depth.view());
    auto aoView = rhi::VkRHITextureView::createNonOwning(vkDevice, rt.ssao.view());

    m_set->write({
        {0, rhi::DescriptorType::SampledImage, nrView.get()},
        {1, rhi::DescriptorType::SampledImage, dpView.get()},
        {2, rhi::DescriptorType::StorageImage, aoView.get()},
    });
}

void SsaoPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt,
                      const glm::mat4& proj, const glm::mat4& invProj,
                      const glm::mat4& view) {
    if (!m_pipeline || !m_set) return;

    // ── 绑定 PSO + 描述符集 ──
    cmd.bindPipelineState(*m_pipeline);
    cmd.bindDescriptorSet(0, *m_set);

    // ── Push Constants ──
    SsaoPC pc{};
    pc.proj = proj;
    pc.invProj = invProj;
    pc.view = view;
    pc.outSizeX = rt.extent.width;
    pc.outSizeY = rt.extent.height;
    pc.invOutSizeX = 1.0f / (float)rt.extent.width;
    pc.invOutSizeY = 1.0f / (float)rt.extent.height;
    pc.radius      = radius;
    pc.bias        = bias;
    pc.sampleCount = (uint32_t)sampleCount;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));

    // ── Dispatch ──
    uint32_t gx = (rt.extent.width  + 7) / 8;
    uint32_t gy = (rt.extent.height + 7) / 8;
    cmd.dispatch(gx, gy, 1);
}


} // namespace somegi
