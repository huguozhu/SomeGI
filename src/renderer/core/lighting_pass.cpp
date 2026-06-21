// LightingPass RHI 实现 — 34 bindings set=0 + IBL set=1。record 通过 RHI 桥接 Vk。
#include "renderer/core/lighting_pass.h"
#include "core/device.h"
#include "core/path_util.h"
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
void LightingPass::init(rhi::RHIDevice& rhiDevice) {
    m_rhiDevice = &rhiDevice;
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
    m_dummyBuf = rhiDevice.createBuffer({4, rhi::BufferUsage::Storage, rhi::MemoryType::HostVisible});
    *static_cast<float*>(m_dummyBuf->map()) = 0.0f;

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
    if (!m_rhiDevice) return;
    m_pipeline.reset(); m_set.reset(); m_setLayout.reset();
    m_iblSet.reset(); m_iblDsl.reset();
    m_lpvSampler.reset();
    m_iblParamsUbo.reset(); m_dummyBuf.reset();
    m_rhiDevice = nullptr;
}

// ════════════════════════════════════════════════════════════════
// bindIblResources
// ════════════════════════════════════════════════════════════════
void LightingPass::bindIblResources(const IblResources& ibl) {
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    m_iblSet = m_rhiDevice->createDescriptorSet(*m_iblDsl);

    struct IblParams { float intensity; float _pad0,_pad1,_pad2; };
    m_iblParamsUbo = m_rhiDevice->createBuffer({sizeof(IblParams), rhi::BufferUsage::Uniform, rhi::MemoryType::HostVisible});

    auto diffV = rhi::VkRHITextureView::createNonOwning(vkD, ibl.diffuseCube.view());
    auto specV = rhi::VkRHITextureView::createNonOwning(vkD, ibl.specularCube.view());
    auto lutV  = rhi::VkRHITextureView::createNonOwning(vkD, ibl.brdfLut.view());
    auto uboB  = m_iblParamsUbo.get();
    auto ibs   = rhi::VkRHISampler::createNonOwning(vkD, ibl.linear);

    m_iblSet->write({
        {0, rhi::DescriptorType::SampledImage, diffV.get()},
        {1, rhi::DescriptorType::SampledImage, specV.get()},
        {2, rhi::DescriptorType::SampledImage, lutV.get()},
        {3, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, ibs.get()},
        {4, rhi::DescriptorType::UniformBuffer, nullptr, uboB},
    });

    IblParams params{};
    params.intensity = m_iblIntensity;
    std::memcpy(m_iblParamsUbo->map(), &params, sizeof(params));
}

void LightingPass::setIblIntensity(float v) {
    m_iblIntensity = v;
    if (m_iblParamsUbo && m_iblParamsUbo->map()) {
        struct IblParams { float intensity; float _pad0,_pad1,_pad2; } p{v};
        std::memcpy(m_iblParamsUbo->map(), &p, sizeof(p));
    }
}

// ════════════════════════════════════════════════════════════════
// bindFrame (set=0, 34 writes — 桥接 Vk 写描述符)
// ════════════════════════════════════════════════════════════════
void LightingPass::bindFrame(const RenderTargets& rt, VkBuffer frameUbo,
                              const LpvGrid& lpv0, const VxgiResources& vxgi,
                              const PrtResources& prt, const DdgiResources& ddgi,
                              VkBuffer ddgiPS) {
    if (!m_set) return;
    auto& vkD = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);

    auto uboF = rhi::VkRHIBuffer::createNonOwning(vkD, frameUbo, VK_WHOLE_SIZE);
    auto nrV  = rt.rhiGNormalRoughView();
    auto abV  = rt.rhiGAlbedoMetalView();
    auto emV  = rt.rhiGEmissiveAOView();
    auto dpV  = rt.rhiDepthView();
    auto outV = rt.rhiHdrColorView();
    auto aoV  = rt.rhiSsaoView();
    auto srV  = rt.rhiSsrView();
    auto sgV  = rt.rhiSsgiView();
    auto rmV  = rt.rhiRsmGIView();

    auto lpvRV = lpv0.lpvRView.get();
    auto lpvGV = lpv0.lpvGView.get();
    auto lpvBV = lpv0.lpvBView.get();
    auto vxV   = vxgi.rhiView();
    auto prtV  = prt.rhiView();
    auto ddIrV = ddgi.irradianceRhiView();
    auto ddDsV = ddgi.distanceRhiView();
    auto ddPSB = ddgi.probeStatesRhi();
    auto vaV   = vxgi.anisoRhiView();
    auto prB   = prt.rhiViewB();
    auto prC   = prt.rhiViewC();
    auto prD   = prt.rhiViewD();
    auto prE   = prt.rhiViewE();
    auto rsV   = rt.rhiRestirView();
    auto rtV   = rt.rhiRtGIView();
    auto lpS   = m_lpvSampler.get();

    // NDGI 权重（6 SSBO），b27-32 初始占位
    auto dumB  = m_dummyBuf.get();
    m_set->write({
        {0, rhi::DescriptorType::UniformBuffer, nullptr, uboF.get()},
        {1, rhi::DescriptorType::SampledImage, abV},
        {2, rhi::DescriptorType::SampledImage, nrV},
        {3, rhi::DescriptorType::SampledImage, emV},
        {4, rhi::DescriptorType::SampledImage, dpV},
        {5, rhi::DescriptorType::StorageImage, outV},
        {6, rhi::DescriptorType::SampledImage, aoV},
        {7, rhi::DescriptorType::SampledImage, srV},
        {8, rhi::DescriptorType::SampledImage, sgV},
        {9, rhi::DescriptorType::SampledImage, rmV},
        {10, rhi::DescriptorType::SampledImage, lpvRV},
        {11, rhi::DescriptorType::SampledImage, lpvGV},
        {12, rhi::DescriptorType::SampledImage, lpvBV},
        {13, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, lpS},
        {14, rhi::DescriptorType::SampledImage, vxV},
        {15, rhi::DescriptorType::SampledImage, prtV},
        {16, rhi::DescriptorType::SampledImage, ddIrV},
        {17, rhi::DescriptorType::SampledImage, ddDsV},
        {18, rhi::DescriptorType::StorageBuffer, nullptr, ddPSB},
        {19, rhi::DescriptorType::SampledImage, vaV},
        {20, rhi::DescriptorType::SampledImage, prB},
        {21, rhi::DescriptorType::SampledImage, prC},
        {22, rhi::DescriptorType::SampledImage, prD},
        {23, rhi::DescriptorType::SampledImage, prE},
        {24, rhi::DescriptorType::SampledImage, rsV},
        {25, rhi::DescriptorType::SampledImage, rtV},
        {26, rhi::DescriptorType::SampledImage, nrV}, // duplicate: shader unused
        {27, rhi::DescriptorType::StorageBuffer, nullptr, dumB},
        {28, rhi::DescriptorType::StorageBuffer, nullptr, dumB},
        {29, rhi::DescriptorType::StorageBuffer, nullptr, dumB},
        {30, rhi::DescriptorType::StorageBuffer, nullptr, dumB},
        {31, rhi::DescriptorType::StorageBuffer, nullptr, dumB},
        {32, rhi::DescriptorType::StorageBuffer, nullptr, dumB},
    });
}

// ════════════════════════════════════════════════════════════════
// bindShadowMask (set=0, binding 33)
// ════════════════════════════════════════════════════════════════
void LightingPass::bindShadowMask(const rhi::RHITextureView& sv) {
    if (!m_set) return;
    m_set->write({{33, rhi::DescriptorType::SampledImage, &sv}});
}

// ════════════════════════════════════════════════════════════════
// setNdgiWeights (set=0, bindings 27-32)
// ════════════════════════════════════════════════════════════════
void LightingPass::setNdgiWeights(VkBuffer w1,VkBuffer b1,
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

} // namespace somegi
