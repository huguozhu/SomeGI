// GtaoPass RHI 实现 — 描述符 set=0:
//   binding 0: gNormalRough (sampled image)
//   binding 1: gDepth       (sampled image, depth aspect)
//   binding 2: gOutAO       (storage image, R8)
// push constant: GtaoPC (224 bytes)

#include "renderer/ao/gtao_pass.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_command.h"
#include "core/path_util.h"
#include <array>
#include <cstring>

namespace somegi {

namespace {
struct GtaoPC {
    glm::mat4 proj, invProj, view;
    uint32_t outSizeX, outSizeY;
    float invOutSizeX, invOutSizeY;
    float radius, falloff;
    uint32_t sliceCount, samplesPerSlice;
};
static_assert(sizeof(GtaoPC) == 224);
}

GtaoPass::~GtaoPass() = default;

void GtaoPass::init(rhi::RHIDevice& d) {
    m_rhiDevice = &d;

    rhi::DescSetLayoutDesc layoutDesc;
    layoutDesc.debugName = "GTAO";
    layoutDesc.bindings = {
        {0, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
        {1, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
        {2, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},
    };
    m_setLayout = d.createDescriptorSetLayout(layoutDesc);
    m_set = d.createDescriptorSet(*m_setLayout);

    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(d);
    rhi::ShaderDesc shaderDesc;
    shaderDesc.stage = rhi::ShaderStage::Compute;
    shaderDesc.entryPoint = "cs_main";
    auto shader = rhi::VkRHIShader::createFromFile(vkDevice, shaderDesc,
        shaderDir() / "ssao" / "gtao.spv");

    rhi::ComputePSODesc psoDesc;
    psoDesc.debugName = "GTAO";
    psoDesc.computeShader = shader.get();
    psoDesc.descriptorSetLayouts = {m_setLayout.get()};
    psoDesc.pushConstants = {{rhi::ShaderStage::Compute, 0, sizeof(GtaoPC)}};
    m_pipeline = d.createComputePSO(psoDesc);
}

void GtaoPass::destroy() {
    m_set.reset();
    m_pipeline.reset();
    m_setLayout.reset();
    m_rhiDevice = nullptr;
}

void GtaoPass::bindFrame(const RenderTargets& rt) {
    if (!m_set) return;
    m_set->write({
        {0, rhi::DescriptorType::SampledImage, rt.rhiGNormalRoughView()},
        {1, rhi::DescriptorType::SampledImage, rt.rhiDepthView()},
        {2, rhi::DescriptorType::StorageImage, rt.rhiSsaoView()},
    });
}

void GtaoPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt,
                      const glm::mat4& proj, const glm::mat4& view) {
    if (!m_pipeline || !m_set) return;

    cmd.bindPipelineState(*m_pipeline);
    cmd.bindDescriptorSet(0, *m_set);

    GtaoPC pc{};
    pc.proj = proj;
    pc.invProj = glm::inverse(proj);
    pc.view = view;
    pc.outSizeX = rt.extent.width;
    pc.outSizeY = rt.extent.height;
    pc.invOutSizeX = 1.0f / (float)rt.extent.width;
    pc.invOutSizeY = 1.0f / (float)rt.extent.height;
    pc.radius = radiusPixels;
    pc.falloff = falloff;
    pc.sliceCount = (uint32_t)sliceCount;
    pc.samplesPerSlice = (uint32_t)samplesPerSlice;
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));

    uint32_t gx = (rt.extent.width  + 7) / 8;
    uint32_t gy = (rt.extent.height + 7) / 8;
    cmd.dispatch(gx, gy, 1);
}


} // namespace somegi
