// LumenProbePass RHI — 2 compute dispatches (generateRays + projectSH), RHI 管理资源。
#include "renderer/gi/lumen/lumen_probe_pass.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_sampler.h"
#include "rhi/vulkan/vk_acceleration_structure.h"
#include "rhi/vulkan/vk_pso.h"
#include "rhi/base/command_buffer.h"
#include "core/device.h"
#include "core/shader.h"
#include "scene/scene_gpu.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include "renderer/gi/rt/scene_rt_as.h"
#include <array>

namespace somegi {

// Bridge helpers
static VkDescriptorSet VkSet(auto& p) { return (VkDescriptorSet)(uintptr_t)p->nativeHandle(); }
static VkPipelineLayout VkLay(auto& p) { return static_cast<rhi::VkRHIPipelineState&>(*p).layout(); }
static VkPipeline VkPipe(auto& p) { return (VkPipeline)(uintptr_t)p->nativeHandle(); }

namespace {
struct ProbePC {
    float    screenSizeX, screenSizeY;
    float    invScreenSizeX, invScreenSizeY;
    uint32_t probeGridW;
    uint32_t probeGridH;
    uint32_t probeTileSize;
    uint32_t raysPerProbe;
    uint32_t totalProbes;
    float    randomSeed;
    uint32_t useSixAxis;
    uint32_t _pad1, _pad2, _pad3;
};
static_assert(sizeof(ProbePC) == 56);
}

void LumenProbePass::init(rhi::RHIDevice& d) {
    m_rhiDevice = &d;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(d);
    using DS = rhi::DescriptorType; using SS = rhi::ShaderStage;

    // Sampler
    m_linearClamp = d.createSampler({rhi::Filter::Linear, rhi::Filter::Linear, rhi::SamplerMipmapMode::Linear, rhi::SamplerAddressMode::ClampToEdge, rhi::SamplerAddressMode::ClampToEdge, rhi::SamplerAddressMode::ClampToEdge, 16.0f});

    // Descriptor set (11 bindings)
    {
        rhi::DescSetLayoutDesc ld; ld.debugName = "LumenProbe";
        ld.bindings = {
            {0,  DS::UniformBuffer,            1, SS::Compute},
            {1,  DS::SampledImage,              1, SS::Compute},
            {2,  DS::SampledImage,              1, SS::Compute},
            {3,  DS::AccelerationStructure,     1, SS::Compute},
            {4,  DS::SampledImage,              1, SS::Compute},
            {5,  DS::Sampler,                   1, SS::Compute},
            {6,  DS::StorageBuffer,             1, SS::Compute},
            {7,  DS::StorageImage,              1, SS::Compute},
            {8,  DS::SampledImage,              1, SS::Compute},
            {9,  DS::SampledImage,              1, SS::Compute},
            {10, DS::SampledImage,              1, SS::Compute},
        };
        m_setLayout = d.createDescriptorSetLayout(ld);
        m_set = d.createDescriptorSet(*m_setLayout);
    }

    // Compute pipelines
    auto mk = [&](const char* entry) {
        rhi::ShaderDesc sd{SS::Compute}; sd.entryPoint = entry;
        auto cs = rhi::VkRHIShader::createFromFile(vkD, sd, shaderDir() / "gi" / "lumen" / "lumen_probe.spv");
        rhi::ComputePSODesc pd; pd.computeShader = cs.get();
        pd.descriptorSetLayouts = {m_setLayout.get()};
        pd.pushConstants = {{SS::Compute, 0, sizeof(ProbePC)}};
        return d.createComputePSO(pd);
    };
    m_pipelineRays = mk("cs_generateRays");
    m_pipelineSH   = mk("cs_projectSH");
}

void LumenProbePass::destroy() {
    m_linearClamp.reset();
    m_setLayout.reset(); m_set.reset();
    m_pipelineRays.reset(); m_pipelineSH.reset();
    m_rhiDevice = nullptr;
}

void LumenProbePass::bindResources(const LumenResources& res, const SceneRtAS& rtAS,
                                    const SceneGpu&, const VxgiResources& vxgi,
                                    const RenderTargets& rt, VkBuffer frameUbo, bool hasSixAxis) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    using DS = rhi::DescriptorType;

    auto ub  = rhi::VkRHIBuffer::createNonOwning(vkD, frameUbo, VK_WHOLE_SIZE);
    auto nr  = rhi::VkRHITextureView::createNonOwning(vkD, rt.gNormalRough.view());
    auto dp  = rhi::VkRHITextureView::createNonOwning(vkD, rt.depth.view());
    auto tas = rhi::VkRHIAccelerationStructure::createNonOwning(vkD, rtAS.tlas());
    auto vox = rhi::VkRHITextureView::createNonOwning(vkD, vxgi.fullView());
    auto rb  = rhi::VkRHIBuffer::createNonOwning(vkD, res.rayBuffer().handle(), VK_WHOLE_SIZE);
    auto pa  = rhi::VkRHITextureView::createNonOwning(vkD, res.probeAtlas().view());

    auto ax  = rhi::VkRHITextureView::createNonOwning(vkD, hasSixAxis ? vxgi.sixAxisX().view() : vxgi.fullView());
    auto ay  = rhi::VkRHITextureView::createNonOwning(vkD, hasSixAxis ? vxgi.sixAxisY().view() : vxgi.fullView());
    auto az  = rhi::VkRHITextureView::createNonOwning(vkD, hasSixAxis ? vxgi.sixAxisZ().view() : vxgi.fullView());

    m_set->write({
        {0,  DS::UniformBuffer,        nullptr, ub.get()},
        {1,  DS::SampledImage,         nr.get()},
        {2,  DS::SampledImage,         dp.get()},
        {3,  DS::AccelerationStructure, nullptr, nullptr, 0, 0, nullptr, tas.get()},
        {4,  DS::SampledImage,         vox.get()},
        {5,  DS::Sampler,              nullptr, nullptr, 0, 0, m_linearClamp.get()},
        {6,  DS::StorageBuffer,        nullptr, rb.get()},
        {7,  DS::StorageImage,         pa.get()},
        {8,  DS::SampledImage,         ax.get()},
        {9,  DS::SampledImage,         ay.get()},
        {10, DS::SampledImage,         az.get()},
    });
}

void LumenProbePass::record(rhi::RHICommandBuffer& cmd, const LumenResources& res,
                             uint32_t, bool useSixAxis) {
    uint32_t pw = res.probeGridW(), ph = res.probeGridH(), pc = res.probeCount();

    // 1. cs_generateRays
    cmd.bindPipelineState(*m_pipelineRays);
    cmd.bindDescriptorSet(0, *m_set);
    ProbePC upc{};
    upc.screenSizeX    = (float)(pw * LumenResources::kProbeTileSize);
    upc.screenSizeY    = (float)(ph * LumenResources::kProbeTileSize);
    upc.invScreenSizeX = 1.0f / upc.screenSizeX;
    upc.invScreenSizeY = 1.0f / upc.screenSizeY;
    upc.probeGridW     = pw; upc.probeGridH = ph;
    upc.probeTileSize  = LumenResources::kProbeTileSize;
    upc.raysPerProbe   = LumenResources::kRaysPerProbe;
    upc.totalProbes    = pc; upc.randomSeed = 0.0f;
    upc.useSixAxis     = useSixAxis ? 1u : 0u;
    cmd.pushConstants(rhi::ShaderStage::Compute, &upc, sizeof(upc));
    cmd.dispatch((pc * LumenResources::kRaysPerProbe + 63) / 64, 1, 1);

    // 内存屏障（RHI 无独立 memory barrier API，用 globalBarrier 替代）
    cmd.globalBarrier();

    // 2. cs_projectSH
    cmd.bindPipelineState(*m_pipelineSH);
    cmd.pushConstants(rhi::ShaderStage::Compute, &upc, sizeof(upc));
    cmd.dispatch((pc + 63) / 64, 1, 1);
}

void LumenProbePass::record(VkCommandBuffer cmd, const LumenResources& res,
                             uint32_t, bool useSixAxis) {
    uint32_t pw = res.probeGridW();
    uint32_t ph = res.probeGridH();
    uint32_t pc = res.probeCount();

    // 1. cs_generateRays
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkPipe(m_pipelineRays));
    VkDescriptorSet ds = VkSet(m_set);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkLay(m_pipelineRays), 0, 1, &ds, 0, nullptr);

    ProbePC upc{};
    upc.screenSizeX    = (float)(pw * LumenResources::kProbeTileSize);
    upc.screenSizeY    = (float)(ph * LumenResources::kProbeTileSize);
    upc.invScreenSizeX = 1.0f / upc.screenSizeX;
    upc.invScreenSizeY = 1.0f / upc.screenSizeY;
    upc.probeGridW     = pw;
    upc.probeGridH     = ph;
    upc.probeTileSize  = LumenResources::kProbeTileSize;
    upc.raysPerProbe   = LumenResources::kRaysPerProbe;
    upc.totalProbes    = pc;
    upc.randomSeed     = 0.0f;
    upc.useSixAxis     = useSixAxis ? 1u : 0u;
    vkCmdPushConstants(cmd, VkLay(m_pipelineRays), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(upc), &upc);

    uint32_t totalRays = pc * LumenResources::kRaysPerProbe;
    vkCmdDispatch(cmd, (totalRays + 63) / 64, 1, 1);

    // Barrier
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);

    // 2. cs_projectSH
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkPipe(m_pipelineSH));
    vkCmdPushConstants(cmd, VkLay(m_pipelineSH), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(upc), &upc);
    vkCmdDispatch(cmd, (pc + 63) / 64, 1, 1);
}

} // namespace somegi
