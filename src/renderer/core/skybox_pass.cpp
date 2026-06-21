// SkyboxPass RHI 实现 — Graphics PSO + 1 descriptor set (UBO + cube + sampler).
// 全屏三角形由 SV_VertexID 生成（无顶点输入）。

#include "renderer/core/skybox_pass.h"
#include "core/device.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/vulkan/vk_sampler.h"
#include "core/shader.h"
#include <array>
#include <cstring>

namespace somegi {

namespace {
struct SkyUbo { glm::mat4 invViewProj; glm::vec4 cameraPos; };

static rhi::Format toRhiFormat(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R16G16B16A16_SFLOAT: return rhi::Format::R16G16B16A16_SFLOAT;
        case VK_FORMAT_R32_SFLOAT:          return rhi::Format::R32_SFLOAT;
        case VK_FORMAT_D32_SFLOAT:          return rhi::Format::D32_SFLOAT;
        case VK_FORMAT_B8G8R8A8_UNORM:      return rhi::Format::B8G8R8A8_UNORM;
        case VK_FORMAT_R8G8B8A8_UNORM:      return rhi::Format::R8G8B8A8_UNORM;
        default: return rhi::Format::Unknown;
    }
}
}

SkyboxPass::~SkyboxPass() = default;

void SkyboxPass::init(Device& d, rhi::RHIDevice& rhiDevice, VkFormat colorFmt, VkFormat depthFmt) {
    m_device = &d;
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
    m_ubo = Buffer(d, sizeof(SkyUbo),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

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
    psoDesc.renderTargets.colorFormats = {toRhiFormat(colorFmt)};
    psoDesc.renderTargets.depthFormat  = toRhiFormat(depthFmt);
    psoDesc.renderTargets.sampleCount  = 1;
    psoDesc.descriptorSetLayouts = {m_setLayout.get()};
    m_pipeline = rhiDevice.createGraphicsPSO(psoDesc);

    // ── 初始 UBO 绑定 ──
    auto uboRHI = rhi::VkRHIBuffer::createNonOwning(vkDevice, m_ubo.handle(), sizeof(SkyUbo));
    m_set->write({{0, rhi::DescriptorType::UniformBuffer, nullptr, uboRHI.get()}});
}

void SkyboxPass::destroy() {
    m_set.reset();
    m_pipeline.reset();
    m_setLayout.reset();
    m_ubo.reset();
    m_device = nullptr;
    m_rhiDevice = nullptr;
}

void SkyboxPass::bindEnv(VkImageView envCubeView, VkSampler linearSampler) {
    if (!m_set) return;
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_sampler = rhi::VkRHISampler::createNonOwning(vkDevice, linearSampler);

    auto cubeView = rhi::VkRHITextureView::createNonOwning(vkDevice, envCubeView);
    m_set->write({
        {1, rhi::DescriptorType::SampledImage, cubeView.get()},
        {2, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, m_sampler.get()},
    });
}
void SkyboxPass::bindEnvRHI(VkImageView envCubeView, rhi::RHISampler& linearSampler) {
    if (!m_set) return;
    m_sampler.reset(&linearSampler);
    m_sampler.release(); // 不拥有所有权

    auto& vkDev = *m_rhiDevice;
    auto cubeView = rhi::VkRHITextureView::createNonOwning(static_cast<rhi::VkRHIDevice&>(vkDev), envCubeView);
    m_set->write({
        {1, rhi::DescriptorType::SampledImage, cubeView.get()},
        {2, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, m_sampler.get()},
    });
}

void SkyboxPass::updateFrame(const glm::mat4& invViewProj, const glm::vec3& cameraPos) {
    SkyUbo u{};
    u.invViewProj = invViewProj;
    u.cameraPos = glm::vec4(cameraPos, 0.0f);
    std::memcpy(m_ubo.mapped(), &u, sizeof(u));
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

// 兼容 VkCommandBuffer
void SkyboxPass::record(VkCommandBuffer vkCmd, const RenderTargets& rt) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
    record(rhiCmd, rt);
}

} // namespace somegi
