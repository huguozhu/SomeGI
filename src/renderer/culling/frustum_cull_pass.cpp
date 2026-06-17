// FrustumCullPass RHI 实现 — 描述符 set=0 (8 bindings):
//   0: drawBuf    (storage buffer)
//   1: uboBuf     (uniform buffer)
//   2: indirectOut (storage buffer)
//   3: countOut   (storage buffer)
//   4-7: hizMip1-4 (sampled image, 可选)

#include "renderer/culling/frustum_cull_pass.h"
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
#include "core/shader.h"
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

void FrustumCullPass::init(Device& d, rhi::RHIDevice& rhiDevice, uint32_t maxDraws) {
    m_device = &d;
    m_rhiDevice = &rhiDevice;
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
    m_dsl = rhiDevice.createDescriptorSetLayout(layoutDesc);

    // ── kFramesInFlight 描述符集 ──
    for (auto& s : m_sets)
        s = rhiDevice.createDescriptorSet(*m_dsl);

    // ── Compute PSO ──
    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(rhiDevice);
    rhi::ShaderDesc shaderDesc;
    shaderDesc.stage = rhi::ShaderStage::Compute;
    shaderDesc.entryPoint = "cs_main";
    auto shader = rhi::VkRHIShader::createFromFile(vkDevice, shaderDesc,
        shaderDir() / "culling" / "frustum_cull.spv");

    rhi::ComputePSODesc psoDesc;
    psoDesc.debugName = "FrustumCull";
    psoDesc.computeShader = shader.get();
    psoDesc.descriptorSetLayouts = {m_dsl.get()};
    m_pipeline = rhiDevice.createComputePSO(psoDesc);

    // ── Uniform buffer（仍使用 core::Buffer） ──
    m_ubo = Buffer(d, sizeof(CullUbo),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void FrustumCullPass::destroy() {
    for (auto& s : m_sets) s.reset();
    m_pipeline.reset();
    m_dsl.reset();
    m_ubo.reset();
    m_device = nullptr;
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
                              VkBuffer drawBuf, uint32_t drawCount,
                              VkBuffer indirectOut, VkBuffer countOut,
                              const glm::mat4& vp, VkExtent2D ss, uint32_t fi) {
    VkImageView nullViews[4] = {};
    record(cmd, drawBuf, drawCount, indirectOut, countOut, vp, ss, fi,
           nullViews[0], nullViews[1], nullViews[2], nullViews[3]);
}

// ── RHI record（含 Hi-Z） ──
void FrustumCullPass::record(rhi::RHICommandBuffer& cmd,
                              VkBuffer drawBuf, uint32_t drawCount,
                              VkBuffer indirectOut, VkBuffer countOut,
                              const glm::mat4& vp, VkExtent2D ss, uint32_t fi,
                              VkImageView hizMip1, VkImageView hizMip2,
                              VkImageView hizMip3, VkImageView hizMip4) {
    if (drawCount == 0 || !m_pipeline) return;

    auto& vkDevice = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    VkCommandBuffer vkCmd = (VkCommandBuffer)(uintptr_t)cmd.nativeHandle();

    // ── 清零 count buffer（暂时通过原生 Vk API） ──
    vkCmdFillBuffer(vkCmd, countOut, 0, sizeof(uint32_t), 0);
    VkBufferMemoryBarrier2 fb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    fb.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    fb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    fb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    fb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    fb.buffer = countOut; fb.size = VK_WHOLE_SIZE;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.bufferMemoryBarrierCount = 1; di.pBufferMemoryBarriers = &fb;
    vkCmdPipelineBarrier2(vkCmd, &di);

    // ── 更新 UBO ──
    CullUbo u{};
    u.viewProj = vp;
    extractFrustumPlanes(vp, u.frustum);
    u.screenSize = glm::vec2(ss.width, ss.height);
    u.drawCount = drawCount;
    u.hizMaxMip = (hizMip1 != VK_NULL_HANDLE) ? 4u : 0u;
    std::memcpy(m_ubo.mapped(), &u, sizeof(u));

    // ── 写描述符集 ──
    VkImageView hiz[4] = {hizMip1, hizMip2, hizMip3, hizMip4};
    writeFrustumDescriptors(vkDevice, *m_sets[fi % 2],
                            drawBuf, m_ubo.handle(), indirectOut, countOut, hiz);

    // ── Dispatch ──
    cmd.bindPipelineState(*m_pipeline);
    cmd.bindDescriptorSet(0, *m_sets[fi % 2]);
    cmd.dispatch((drawCount + 255) / 256, 1, 1);
}

// ── 兼容 VkCommandBuffer 重载 ──
void FrustumCullPass::record(VkCommandBuffer vkCmd, VkBuffer drawBuf, uint32_t drawCount,
                              VkBuffer indirectOut, VkBuffer countOut,
                              const glm::mat4& vp, VkExtent2D screenSize, uint32_t fi) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
    record(rhiCmd, drawBuf, drawCount, indirectOut, countOut, vp, screenSize, fi);
}

void FrustumCullPass::record(VkCommandBuffer vkCmd, VkBuffer drawBuf, uint32_t drawCount,
                              VkBuffer indirectOut, VkBuffer countOut,
                              const glm::mat4& vp, VkExtent2D screenSize, uint32_t fi,
                              VkImageView hizMip1, VkImageView hizMip2,
                              VkImageView hizMip3, VkImageView hizMip4) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
    rhi::VkRHICommandBuffer rhiCmd(vkDev, vkCmd);
    record(rhiCmd, drawBuf, drawCount, indirectOut, countOut, vp, screenSize, fi,
           hizMip1, hizMip2, hizMip3, hizMip4);
}

} // namespace somegi
