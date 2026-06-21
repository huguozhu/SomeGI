// SkyboxPass RHI 实现 — Graphics PSO + 1 descriptor set (UBO + cube + sampler).
// 全屏三角形由 SV_VertexID 生成（无顶点输入）。

#include "renderer/core/skybox_pass.h"
#include "rhi/base/device.h"
#include "rhi/base/buffer.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_command.h"
#include "core/path_util.h"
#include <cstring>

namespace somegi {

namespace {
struct SkyUbo { glm::mat4 invViewProj; glm::vec4 cameraPos; };
}

SkyboxPass::~SkyboxPass() = default;

void SkyboxPass::init(rhi::RHIDevice& rhiDevice, rhi::Format colorFmt, rhi::Format depthFmt) {
    m_rhiDevice = &rhiDevice;
    m_colorFmt = colorFmt;
    m_depthFmt = depthFmt;
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(rhiDevice);

    // ── Descriptor Set Layout ──
    rhi::DescSetLayoutDesc layoutDesc;
    layoutDesc.debugName = "Skybox";
    layoutDesc.bindings = {
        {0, rhi::DescriptorType::UniformBuffer, 1, rhi::ShaderStage::Vertex | rhi::ShaderStage::Fragment},
        {1, rhi::DescriptorType::SampledImage,  1, rhi::ShaderStage::Fragment},
        {2, rhi::DescriptorType::Sampler,        1, rhi::ShaderStage::Fragment},
    };
    m_setLayout = rhiDevice.createDescriptorSetLayout(layoutDesc);

    // ── Descriptor Set ──
    m_set = rhiDevice.createDescriptorSet(*m_setLayout);

    // ── UBO ──
    {
        rhi::BufferDesc uboDesc;
        uboDesc.size = sizeof(SkyUbo);
        uboDesc.usage = rhi::BufferUsage::Uniform;
        uboDesc.memory = rhi::MemoryType::HostVisible;
        uboDesc.debugName = "SkyboxUBO";
        m_ubo = rhiDevice.createBuffer(uboDesc);
    }

    // ── Shaders（同一 SPV 文件，不同入口点） ──
    auto spvPath = shaderDir() / "skybox" / "skybox.spv";
    rhi::ShaderDesc vsDesc; vsDesc.stage = rhi::ShaderStage::Vertex;   vsDesc.entryPoint = "vs_main";
    rhi::ShaderDesc fsDesc; fsDesc.stage = rhi::ShaderStage::Fragment; fsDesc.entryPoint = "ps_main";
    auto vs = rhi::VkRHIShader::createFromFile(vkDevice, vsDesc, spvPath);
    auto fs = rhi::VkRHIShader::createFromFile(vkDevice, fsDesc, spvPath);

    // ── Graphics PSO ──
    rhi::GraphicsPSODesc psoDesc;
    psoDesc.debugName = "Skybox";
    psoDesc.vertexShader   = vs.get();
    psoDesc.fragmentShader = fs.get();
    psoDesc.topology = rhi::PrimitiveTopology::TriangleList;
    psoDesc.rasterization.fill = rhi::FillMode::Solid;
    psoDesc.rasterization.cull = rhi::CullMode::None;
    psoDesc.rasterization.frontCCW = true;  // VK_FRONT_FACE_COUNTER_CLOCKWISE
    psoDesc.depthStencil.depthTest  = true;
    psoDesc.depthStencil.depthWrite = false;
    psoDesc.depthStencil.depthCompare = rhi::CompareFunc::LessEqual;
    psoDesc.renderTargets.colorFormats = {colorFmt};
    psoDesc.renderTargets.depthFormat  = depthFmt;
    psoDesc.renderTargets.sampleCount  = 1;
    psoDesc.descriptorSetLayouts = {m_setLayout.get()};
    m_pipeline = rhiDevice.createGraphicsPSO(psoDesc);

    // ── 初始 UBO 绑定 ──
    m_set->write({{0, rhi::DescriptorType::UniformBuffer, nullptr, m_ubo.get()}});
}

void SkyboxPass::destroy() {
    m_set.reset();
    m_pipeline.reset();
    m_setLayout.reset();
    m_ubo.reset();
    m_rhiDevice = nullptr;
}

void SkyboxPass::bindEnv(const rhi::RHITextureView& envCubeView, const rhi::RHISampler& linearSampler) {
    if (!m_set) return;
    m_set->write({
        {1, rhi::DescriptorType::SampledImage, &envCubeView},
        {2, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, &linearSampler},
    });
}
void SkyboxPass::bindEnvRHI(const rhi::RHITextureView& envCubeView, const rhi::RHISampler& linearSampler) {
    if (!m_set) return;
    m_set->write({
        {1, rhi::DescriptorType::SampledImage, &envCubeView},
        {2, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, &linearSampler},
    });
}

void SkyboxPass::updateFrame(const glm::mat4& invViewProj, const glm::vec3& cameraPos) {
    SkyUbo u{};
    u.invViewProj = invViewProj;
    u.cameraPos = glm::vec4(cameraPos, 0.0f);
    std::memcpy(m_ubo->map(), &u, sizeof(u));
    m_ubo->unmap();
}

void SkyboxPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt) {
    if (!m_pipeline || !m_set) return;
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    // ── beginRendering ──
    auto colorView = rhi::VkRHITextureView::createNonOwning(vkDevice, rt.hdrColor.view());
    auto depthView = rhi::VkRHITextureView::createNonOwning(vkDevice, rt.depth.view());
    const rhi::RHITextureView* cv = colorView.get();
    rhi::RenderingAttachmentInfo cAttach{}; cAttach.view=cv; cAttach.loadOp=rhi::AttachmentLoadOp::Load;
    rhi::RenderingAttachmentInfo dAttach{}; dAttach.view=depthView.get(); dAttach.loadOp=rhi::AttachmentLoadOp::Load;
    cmd.beginRendering(&cAttach, 1, &dAttach, rt.extent.width, rt.extent.height);

    cmd.setViewport(0, 0, (float)rt.extent.width, (float)rt.extent.height);
    cmd.setScissor(0, 0, rt.extent.width, rt.extent.height);

    cmd.bindPipelineState(*m_pipeline);
    cmd.bindDescriptorSet(0, *m_set);
    cmd.draw(3, 0, 0);  // 全屏三角形

    cmd.endRendering();
}

} // namespace somegi
