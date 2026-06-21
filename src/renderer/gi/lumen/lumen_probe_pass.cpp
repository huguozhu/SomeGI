// LumenProbePass RHI — 2 compute dispatches (generateRays + projectSH), 纯 RHI record()。
#include "renderer/gi/lumen/lumen_probe_pass.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_acceleration_structure.h"
#include "rhi/base/command_buffer.h"
#include "core/device.h"
#include "core/path_util.h"
#include "scene/scene_gpu.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include "renderer/gi/rt/scene_rt_as.h"
#include <array>

namespace somegi {

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
    auto nr  = rt.rhiGNormalRoughView();
    auto dp  = rt.rhiDepthView();
    auto tas = rtAS.tlas();
    auto vox = vxgi.rhiView();
    auto rb  = res.rayBufferRhi();
    auto pa  = res.probeAtlasRhiView();

    auto ax  = hasSixAxis ? vxgi.sixAxisXRhiView() : vxgi.rhiView();
    auto ay  = hasSixAxis ? vxgi.sixAxisYRhiView() : vxgi.rhiView();
    auto az  = hasSixAxis ? vxgi.sixAxisZRhiView() : vxgi.rhiView();

    m_set->write({
        {0,  DS::UniformBuffer,        nullptr, ub.get()},
        {1,  DS::SampledImage,         nr},
        {2,  DS::SampledImage,         dp},
        {3,  DS::AccelerationStructure, nullptr, nullptr, 0, 0, nullptr, tas},
        {4,  DS::SampledImage,         vox},
        {5,  DS::Sampler,              nullptr, nullptr, 0, 0, m_linearClamp.get()},
        {6,  DS::StorageBuffer,        nullptr, rb},
        {7,  DS::StorageImage,         pa},
        {8,  DS::SampledImage,         ax},
        {9,  DS::SampledImage,         ay},
        {10, DS::SampledImage,         az},
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

} // namespace somegi
