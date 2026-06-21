// FrustumCullPass RHI 实现 — 描述符 set=0 (8 bindings):
//   0: drawBuf    (storage buffer)
//   1: uboBuf     (uniform buffer)
//   2: indirectOut (storage buffer)
//   3: countOut   (storage buffer)
//   4-7: hizMip1-4 (sampled image, 可选)

#include "renderer/culling/frustum_cull_pass.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include "core/path_util.h"
#include <array>
#include <cstring>
#include <cmath>

namespace somegi {

struct CullUbo {
    glm::mat4 viewProj;
    glm::vec4 frustum[6];
    glm::vec2 screenSize;
    uint32_t  drawCount;
    uint32_t  hizMaxMip;
};

void extractFrustumPlanes(const glm::mat4& vp, glm::vec4 f[6]) {
    glm::vec4 r1=vp[0],r2=vp[1],r3=vp[2],r4=vp[3];
    f[0]=r4+r1;f[1]=r4-r1;f[2]=r4+r2;f[3]=r4-r2;f[4]=r4+r3;f[5]=r4-r3;
    for(int i=0;i<6;++i){float L=glm::length(glm::vec3(f[i]));if(L>1e-10f)f[i]/=L;}
}

FrustumCullPass::~FrustumCullPass() = default;

void FrustumCullPass::init(rhi::RHIDevice& d, uint32_t maxDraws) {
    m_rhiDevice = &d;
    m_maxDraws = maxDraws;

    // ── Descriptor Set Layout（8 bindings） ──
    rhi::DescSetLayoutDesc layoutDesc;
    layoutDesc.debugName = "FrustumCull";
    using DS = rhi::DescriptorType;
    using SS = rhi::ShaderStage;
    layoutDesc.bindings = {
        {0, DS::StorageBuffer, 1, SS::Compute},
        {1, DS::UniformBuffer, 1, SS::Compute},
        {2, DS::StorageBuffer, 1, SS::Compute},
        {3, DS::StorageBuffer, 1, SS::Compute},
        {4, DS::SampledImage,  1, SS::Compute},
        {5, DS::SampledImage,  1, SS::Compute},
        {6, DS::SampledImage,  1, SS::Compute},
        {7, DS::SampledImage,  1, SS::Compute},
    };
    m_dsl = d.createDescriptorSetLayout(layoutDesc);

    // ── kFramesInFlight 描述符集 ──
    for (auto& s : m_sets)
        s = d.createDescriptorSet(*m_dsl);

    // ── Compute PSO ──
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(d);
    rhi::ShaderDesc shaderDesc;
    shaderDesc.stage = rhi::ShaderStage::Compute;
    shaderDesc.entryPoint = "cs_main";
    auto shader = rhi::VkRHIShader::createFromFile(vkDevice, shaderDesc,
        shaderDir() / "culling" / "frustum_cull.spv");

    rhi::ComputePSODesc psoDesc;
    psoDesc.debugName = "FrustumCull";
    psoDesc.computeShader = shader.get();
    psoDesc.descriptorSetLayouts = {m_dsl.get()};
    m_pipeline = d.createComputePSO(psoDesc);

    // ── Uniform buffer（纯 RHI，HostVisible） ──
    rhi::BufferDesc uboDesc;
    uboDesc.size = sizeof(CullUbo);
    uboDesc.usage = rhi::BufferUsage::Uniform;
    uboDesc.memory = rhi::MemoryType::HostVisible;
    m_ubo = d.createBuffer(uboDesc);
}

void FrustumCullPass::destroy() {
    for (auto& s : m_sets) s.reset();
    m_pipeline.reset();
    m_dsl.reset();
    m_ubo.reset();
    m_rhiDevice = nullptr;
}

// ── 内部：写描述符集 ──
static void writeFrustumDescriptors(rhi::VkRHIDevice& vkDevice,
                                     rhi::RHIDescriptorSet& set,
                                     VkBuffer drawBuf, VkBuffer uboBuf,
                                     VkBuffer indOut, VkBuffer cntOut,
                                     VkImageView hiz[4]) {
    // 非拥有型 RHI 包装
    auto drawRHI  = rhi::VkRHIBuffer::createNonOwning(vkDevice, drawBuf, VK_WHOLE_SIZE);
    auto uboRHI   = rhi::VkRHIBuffer::createNonOwning(vkDevice, uboBuf, sizeof(CullUbo));
    auto indRHI   = rhi::VkRHIBuffer::createNonOwning(vkDevice, indOut, VK_WHOLE_SIZE);
    auto cntRHI   = rhi::VkRHIBuffer::createNonOwning(vkDevice, cntOut, sizeof(uint32_t));

    std::vector<rhi::DescriptorWrite> writes = {
        {0, rhi::DescriptorType::StorageBuffer, nullptr, drawRHI.get()},
        {1, rhi::DescriptorType::UniformBuffer, nullptr, uboRHI.get()},
        {2, rhi::DescriptorType::StorageBuffer, nullptr, indRHI.get()},
        {3, rhi::DescriptorType::StorageBuffer, nullptr, cntRHI.get()},
    };

    // Hi-Z mips (可选)
    for (int i = 0; i < 4; ++i) {
        if (hiz[i] != VK_NULL_HANDLE) {
            auto hizView = rhi::VkRHITextureView::createNonOwning(vkDevice, hiz[i]);
            writes.push_back({4u + (uint32_t)i, rhi::DescriptorType::SampledImage, hizView.get()});
        }
    }
    set.write(writes);
}

// ── RHI record（无 Hi-Z） ──
void FrustumCullPass::record(rhi::RHICommandBuffer& cmd,
                              const rhi::RHIBuffer& drawBuf, uint32_t drawCount,
                              const rhi::RHIBuffer& indirectOut, const rhi::RHIBuffer& countOut,
                              const glm::mat4& vp, VkExtent2D ss, uint32_t fi) {
    VkImageView nullViews[4] = {};
    record(cmd, drawBuf, drawCount, indirectOut, countOut, vp, ss, fi,
           nullViews[0], nullViews[1], nullViews[2], nullViews[3]);
}

// ── RHI record（含 Hi-Z） ──
void FrustumCullPass::record(rhi::RHICommandBuffer& cmd,
                              const rhi::RHIBuffer& drawBuf, uint32_t drawCount,
                              const rhi::RHIBuffer& indirectOut, const rhi::RHIBuffer& countOut,
                              const glm::mat4& vp, VkExtent2D ss, uint32_t fi,
                              VkImageView hizMip1, VkImageView hizMip2,
                              VkImageView hizMip3, VkImageView hizMip4) {
    if (drawCount == 0 || !m_pipeline) return;

    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    // 从 RHI 抽象提取 Vulkan 原生句柄（仅用于描述符写入）
    auto vkCountOut = static_cast<VkBuffer>(countOut.nativeHandle());
    auto vkDrawBuf = static_cast<VkBuffer>(drawBuf.nativeHandle());
    auto vkIndirectOut = static_cast<VkBuffer>(indirectOut.nativeHandle());

    // ── 清零 count buffer（纯 RHI fillBuffer + barrier） ──
    cmd.fillBuffer(countOut, 0, sizeof(uint32_t), 0);
    cmd.bufferBarrier(countOut,
        rhi::PipelineStage::Transfer, rhi::PipelineStage::ComputeShader,
        rhi::BufferAccess::TransferWrite, rhi::BufferAccess::StorageWrite);

    // ── 更新 UBO ──
    CullUbo u{};
    u.viewProj = vp;
    extractFrustumPlanes(vp, u.frustum);
    u.screenSize = glm::vec2(ss.width, ss.height);
    u.drawCount = drawCount;
    u.hizMaxMip = (hizMip1 != VK_NULL_HANDLE) ? 4u : 0u;
    {
        void* ptr = m_ubo->map();
        std::memcpy(ptr, &u, sizeof(u));
        m_ubo->unmap();
    }

    // ── 写描述符集 ──
    VkImageView hiz[4] = {hizMip1, hizMip2, hizMip3, hizMip4};
    writeFrustumDescriptors(vkDevice, *m_sets[fi % 2],
                            vkDrawBuf, static_cast<VkBuffer>(m_ubo->nativeHandle()), vkIndirectOut, vkCountOut, hiz);

    // ── Dispatch ──
    cmd.bindPipelineState(*m_pipeline);
    cmd.bindDescriptorSet(0, *m_sets[fi % 2]);
    cmd.dispatch((drawCount + 255) / 256, 1, 1);
}

} // namespace somegi
