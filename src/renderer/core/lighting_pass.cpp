// LightingPass RHI 实现 — 34 bindings set=0 + IBL set=1。record 通过 RHI 桥接 Vk。
#include "renderer/core/lighting_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_sampler.h"
#include "rhi/vulkan/vk_pso.h"
#include "rhi/base/command_buffer.h"
#include <array>
#include <cstring>

namespace somegi {
namespace { struct LightingPC { uint32_t outSizeX,outSizeY; float invOutSizeX,invOutSizeY; };
static_assert(sizeof(LightingPC)==16); }

LightingPass::~LightingPass() = default;

// ════════════════════════════════════════════════════════════════
// Init
// ════════════════════════════════════════════════════════════════
void LightingPass::init(Device& d, rhi::RHIDevice& rhiDevice) {
    m_device = &d; m_rhiDevice = &rhiDevice;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(rhiDevice);

    // ── set=0 layout (34 bindings, UPDATE_AFTER_BIND on shadowMask@33) ──
    {
        rhi::DescSetLayoutDesc ld; ld.debugName = "Lighting"; ld.updateAfterBind = true;
        ld.updateAfterBindBindings = {33};
        ld.bindings = {
            {0, rhi::DescriptorType::UniformBuffer, 1, rhi::ShaderStage::Compute},
            {1, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {2, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {3, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {4, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {5, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},
            {6, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {7, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {8, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {9, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {10, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {11, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {12, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {13, rhi::DescriptorType::Sampler, 1, rhi::ShaderStage::Compute},
            {14, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {15, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {16, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {17, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {18, rhi::DescriptorType::StorageBuffer, 1, rhi::ShaderStage::Compute},
            {19, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {20, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {21, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {22, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {23, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {24, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {25, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {26, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {27, rhi::DescriptorType::StorageBuffer, 1, rhi::ShaderStage::Compute},
            {28, rhi::DescriptorType::StorageBuffer, 1, rhi::ShaderStage::Compute},
            {29, rhi::DescriptorType::StorageBuffer, 1, rhi::ShaderStage::Compute},
            {30, rhi::DescriptorType::StorageBuffer, 1, rhi::ShaderStage::Compute},
            {31, rhi::DescriptorType::StorageBuffer, 1, rhi::ShaderStage::Compute},
            {32, rhi::DescriptorType::StorageBuffer, 1, rhi::ShaderStage::Compute},
            {33, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
        };
        m_setLayout = rhiDevice.createDescriptorSetLayout(ld);
        m_set = rhiDevice.createDescriptorSet(*m_setLayout);
    }

    // ── LPV sampler ──
    m_lpvSampler = rhiDevice.createSampler({rhi::Filter::Linear,rhi::Filter::Linear,
        rhi::SamplerMipmapMode::Linear, rhi::SamplerAddressMode::ClampToEdge,
        rhi::SamplerAddressMode::ClampToEdge, rhi::SamplerAddressMode::ClampToEdge, 0.f});

    // ── NDGI dummy buffer ──
    m_dummyBuf = Buffer(d, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    *static_cast<float*>(m_dummyBuf.mapped()) = 0.0f;

    // ── IBL set=1 layout ──
    {
        rhi::DescSetLayoutDesc ld; ld.debugName = "LightingIBL";
        using SS = rhi::ShaderStage;
        auto vis = static_cast<SS>(static_cast<uint32_t>(SS::Fragment) | static_cast<uint32_t>(SS::Compute));
        ld.bindings = {
            {0, rhi::DescriptorType::SampledImage, 1, vis},
            {1, rhi::DescriptorType::SampledImage, 1, vis},
            {2, rhi::DescriptorType::SampledImage, 1, vis},
            {3, rhi::DescriptorType::Sampler, 1, vis},
            {4, rhi::DescriptorType::UniformBuffer, 1, vis},
        };
        m_iblDsl = rhiDevice.createDescriptorSetLayout(ld);
    }

    // ── Pipeline (set=0 + set=1) ──
    {
        rhi::ShaderDesc sd; sd.stage = rhi::ShaderStage::Compute; sd.entryPoint = "cs_main";
        auto sh = rhi::VkRHIShader::createFromFile(vkD, sd, shaderDir() / "lighting" / "lighting.spv");
        rhi::ComputePSODesc pd; pd.debugName = "Lighting"; pd.computeShader = sh.get();
        pd.descriptorSetLayouts = {m_setLayout.get(), m_iblDsl.get()};
        pd.pushConstants = {{rhi::ShaderStage::Compute, 0, 16}};
        m_pipeline = rhiDevice.createComputePSO(pd);
    }
}

// ════════════════════════════════════════════════════════════════
// Destroy
// ════════════════════════════════════════════════════════════════
void LightingPass::destroy() {
    if (!m_device) return;
    m_pipeline.reset(); m_set.reset(); m_setLayout.reset();
    m_iblSet.reset(); m_iblDsl.reset();
    m_lpvSampler.reset();
    m_iblParamsUbo.reset(); m_dummyBuf.reset();
    m_device = nullptr; m_rhiDevice = nullptr;
}

// ════════════════════════════════════════════════════════════════
// Vk 桥接工具
// ════════════════════════════════════════════════════════════════
static VkDescriptorSet VkSet(auto& p) { return (VkDescriptorSet)(uintptr_t)p->nativeHandle(); }
static VkPipelineLayout VkLay(auto& p) { return static_cast<rhi::VkRHIPipelineState&>(*p).layout(); }

// ════════════════════════════════════════════════════════════════
// bindIblResources
// ════════════════════════════════════════════════════════════════
void LightingPass::bindIblResources(Device& d, const IblResources& ibl) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_iblSet = m_rhiDevice->createDescriptorSet(*m_iblDsl);

    struct IblParams { float intensity; float _pad0,_pad1,_pad2; };
    m_iblParamsUbo = Buffer(d, sizeof(IblParams),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    auto diffV = rhi::VkRHITextureView::createNonOwning(vkD, ibl.diffuseCube.view());
    auto specV = rhi::VkRHITextureView::createNonOwning(vkD, ibl.specularCube.view());
    auto lutV  = rhi::VkRHITextureView::createNonOwning(vkD, ibl.brdfLut.view());
    auto uboB  = rhi::VkRHIBuffer::createNonOwning(vkD, m_iblParamsUbo.handle(), sizeof(IblParams));
    auto ibs   = rhi::VkRHISampler::createNonOwning(vkD, ibl.linear);

    m_iblSet->write({
        {0, rhi::DescriptorType::SampledImage, diffV.get()},
        {1, rhi::DescriptorType::SampledImage, specV.get()},
        {2, rhi::DescriptorType::SampledImage, lutV.get()},
        {3, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, ibs.get()},
        {4, rhi::DescriptorType::UniformBuffer, nullptr, uboB.get()},
    });

    IblParams params{};
    params.intensity = m_iblIntensity;
    std::memcpy(m_iblParamsUbo.mapped(), &params, sizeof(params));
}

void LightingPass::setIblIntensity(float v) {
    m_iblIntensity = v;
    if (m_iblParamsUbo.mapped()) {
        struct IblParams { float intensity; float _pad0,_pad1,_pad2; } p{v};
        std::memcpy(m_iblParamsUbo.mapped(), &p, sizeof(p));
    }
}

// ════════════════════════════════════════════════════════════════
// bindFrame (set=0, 34 writes — 桥接 Vk 写描述符)
// ════════════════════════════════════════════════════════════════
void LightingPass::bindFrame(Device& d, const RenderTargets& rt, VkBuffer frameUbo,
                              const LpvGrid& lpv0, const VxgiResources& vxgi,
                              const PrtResources& prt, const DdgiResources& ddgi,
                              VkBuffer ddgiPS) {
    if (!m_set) return;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    auto uboF = rhi::VkRHIBuffer::createNonOwning(vkD, frameUbo, VK_WHOLE_SIZE);
    auto nrV  = rhi::VkRHITextureView::createNonOwning(vkD, rt.gNormalRough.view());
    auto abV  = rhi::VkRHITextureView::createNonOwning(vkD, rt.gAlbedoMetal.view());
    auto emV  = rhi::VkRHITextureView::createNonOwning(vkD, rt.gEmissiveAO.view());
    auto dpV  = rhi::VkRHITextureView::createNonOwning(vkD, rt.depth.view());
    auto outV = rhi::VkRHITextureView::createNonOwning(vkD, rt.hdrColor.view());
    auto aoV  = rhi::VkRHITextureView::createNonOwning(vkD, rt.ssao.view());
    auto srV  = rhi::VkRHITextureView::createNonOwning(vkD, rt.ssr.view());
    auto sgV  = rhi::VkRHITextureView::createNonOwning(vkD, rt.ssgi.view());
    auto rmV  = rhi::VkRHITextureView::createNonOwning(vkD, rt.rsmGI.view());

    auto lpvRV = rhi::VkRHITextureView::createNonOwning(vkD, lpv0.lpvR.view());
    auto lpvGV = rhi::VkRHITextureView::createNonOwning(vkD, lpv0.lpvG.view());
    auto lpvBV = rhi::VkRHITextureView::createNonOwning(vkD, lpv0.lpvB.view());
    auto vxV   = rhi::VkRHITextureView::createNonOwning(vkD, vxgi.fullView());
    auto prtV  = rhi::VkRHITextureView::createNonOwning(vkD, prt.view());
    auto ddIrV = rhi::VkRHITextureView::createNonOwning(vkD, ddgi.irradiance().view());
    auto ddDsV = rhi::VkRHITextureView::createNonOwning(vkD, ddgi.distance().view());
    auto ddPSB = rhi::VkRHIBuffer::createNonOwning(vkD, ddgiPS, VK_WHOLE_SIZE);
    auto vaV   = rhi::VkRHITextureView::createNonOwning(vkD, vxgi.anisoFullView());
    auto prB   = rhi::VkRHITextureView::createNonOwning(vkD, prt.viewB());
    auto prC   = rhi::VkRHITextureView::createNonOwning(vkD, prt.viewC());
    auto prD   = rhi::VkRHITextureView::createNonOwning(vkD, prt.viewD());
    auto prE   = rhi::VkRHITextureView::createNonOwning(vkD, prt.viewE());
    auto rsV   = rhi::VkRHITextureView::createNonOwning(vkD, rt.restir.view());
    auto rtV   = rhi::VkRHITextureView::createNonOwning(vkD, rt.rtGI.view());
    auto lpS   = rhi::VkRHISampler::createNonOwning(vkD, (VkSampler)(uintptr_t)m_lpvSampler->nativeHandle());

    // NDGI 权重（6 SSBO），b27-32 初始占位
    auto dumB  = rhi::VkRHIBuffer::createNonOwning(vkD, m_dummyBuf.handle(), 4);
    m_set->write({
        {0, rhi::DescriptorType::UniformBuffer, nullptr, uboF.get()},
        {1, rhi::DescriptorType::SampledImage, abV.get()},
        {2, rhi::DescriptorType::SampledImage, nrV.get()},
        {3, rhi::DescriptorType::SampledImage, emV.get()},
        {4, rhi::DescriptorType::SampledImage, dpV.get()},
        {5, rhi::DescriptorType::StorageImage, outV.get()},
        {6, rhi::DescriptorType::SampledImage, aoV.get()},
        {7, rhi::DescriptorType::SampledImage, srV.get()},
        {8, rhi::DescriptorType::SampledImage, sgV.get()},
        {9, rhi::DescriptorType::SampledImage, rmV.get()},
        {10, rhi::DescriptorType::SampledImage, lpvRV.get()},
        {11, rhi::DescriptorType::SampledImage, lpvGV.get()},
        {12, rhi::DescriptorType::SampledImage, lpvBV.get()},
        {13, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, lpS.get()},
        {14, rhi::DescriptorType::SampledImage, vxV.get()},
        {15, rhi::DescriptorType::SampledImage, prtV.get()},
        {16, rhi::DescriptorType::SampledImage, ddIrV.get()},
        {17, rhi::DescriptorType::SampledImage, ddDsV.get()},
        {18, rhi::DescriptorType::StorageBuffer, nullptr, ddPSB.get()},
        {19, rhi::DescriptorType::SampledImage, vaV.get()},
        {20, rhi::DescriptorType::SampledImage, prB.get()},
        {21, rhi::DescriptorType::SampledImage, prC.get()},
        {22, rhi::DescriptorType::SampledImage, prD.get()},
        {23, rhi::DescriptorType::SampledImage, prE.get()},
        {24, rhi::DescriptorType::SampledImage, rsV.get()},
        {25, rhi::DescriptorType::SampledImage, rtV.get()},
        {26, rhi::DescriptorType::SampledImage, nrV.get()}, // duplicate: shader unused
        {27, rhi::DescriptorType::StorageBuffer, nullptr, dumB.get()},
        {28, rhi::DescriptorType::StorageBuffer, nullptr, dumB.get()},
        {29, rhi::DescriptorType::StorageBuffer, nullptr, dumB.get()},
        {30, rhi::DescriptorType::StorageBuffer, nullptr, dumB.get()},
        {31, rhi::DescriptorType::StorageBuffer, nullptr, dumB.get()},
        {32, rhi::DescriptorType::StorageBuffer, nullptr, dumB.get()},
    });
}

// ════════════════════════════════════════════════════════════════
// bindShadowMask (set=0, binding 33)
// ════════════════════════════════════════════════════════════════
void LightingPass::bindShadowMask(Device&, VkImageView v) {
    if (!m_set) return;
    m_shadowMaskView = v;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto sv = rhi::VkRHITextureView::createNonOwning(vkD, v);
    m_set->write({{33, rhi::DescriptorType::SampledImage, sv.get()}});
}

// ════════════════════════════════════════════════════════════════
// setNdgiWeights (set=0, bindings 27-32)
// ════════════════════════════════════════════════════════════════
void LightingPass::setNdgiWeights(Device&, VkBuffer w1,VkBuffer b1,
                                   VkBuffer w2,VkBuffer b2,VkBuffer w3,VkBuffer b3) {
    if (!m_set) return;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    auto W1=rhi::VkRHIBuffer::createNonOwning(vkD,w1,VK_WHOLE_SIZE);
    auto B1=rhi::VkRHIBuffer::createNonOwning(vkD,b1,VK_WHOLE_SIZE);
    auto W2=rhi::VkRHIBuffer::createNonOwning(vkD,w2,VK_WHOLE_SIZE);
    auto B2=rhi::VkRHIBuffer::createNonOwning(vkD,b2,VK_WHOLE_SIZE);
    auto W3=rhi::VkRHIBuffer::createNonOwning(vkD,w3,VK_WHOLE_SIZE);
    auto B3=rhi::VkRHIBuffer::createNonOwning(vkD,b3,VK_WHOLE_SIZE);
    m_set->write({
        {27,rhi::DescriptorType::StorageBuffer,nullptr,W1.get()},
        {28,rhi::DescriptorType::StorageBuffer,nullptr,B1.get()},
        {29,rhi::DescriptorType::StorageBuffer,nullptr,W2.get()},
        {30,rhi::DescriptorType::StorageBuffer,nullptr,B2.get()},
        {31,rhi::DescriptorType::StorageBuffer,nullptr,W3.get()},
        {32,rhi::DescriptorType::StorageBuffer,nullptr,B3.get()},
    });
}

// ════════════════════════════════════════════════════════════════
// record
// ════════════════════════════════════════════════════════════════
void LightingPass::record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt) {
    if (!m_pipeline || !m_set || !m_iblSet) return;

    // 绑定 PSO + 描述符集（set=0 + IBL set=1）
    cmd.bindPipelineState(*m_pipeline);
    const rhi::RHIDescriptorSet* dsets[2] = {m_set.get(), m_iblSet.get()};
    cmd.bindDescriptorSets(0, 2, dsets);

    // push constant
    LightingPC pc{rt.extent.width, rt.extent.height, 1.f/rt.extent.width, 1.f/rt.extent.height};
    cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));

    // dispatch
    uint32_t gx = (rt.extent.width + 7) / 8, gy = (rt.extent.height + 7) / 8;
    cmd.dispatch(gx, gy, 1);
}

void LightingPass::record(VkCommandBuffer vkCmd, const RenderTargets& rt) {
    if (!m_pipeline || !m_set || !m_iblSet) return;
    VkDescriptorSet ds[2] = {VkSet(m_set), VkSet(m_iblSet)};
    vkCmdBindPipeline(vkCmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)(uintptr_t)m_pipeline->nativeHandle());
    vkCmdBindDescriptorSets(vkCmd, VK_PIPELINE_BIND_POINT_COMPUTE, VkLay(m_pipeline), 0, 2, ds, 0, nullptr);
    LightingPC pc{rt.extent.width, rt.extent.height, 1.f/rt.extent.width, 1.f/rt.extent.height};
    vkCmdPushConstants(vkCmd, VkLay(m_pipeline), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t gx = (rt.extent.width + 7) / 8, gy = (rt.extent.height + 7) / 8;
    vkCmdDispatch(vkCmd, gx, gy, 1);
}

} // namespace somegi
