#include "vxgi_aniso_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>
#include <vector>

namespace somegi {

namespace {
struct AnisoPC {
    uint32_t dstSize;
    uint32_t useAnisoSrc;
    uint32_t srcLevel;
    uint32_t _pad;
};
}

void VxgiAnisoPass::init(Device& d, uint32_t mipLevels) {
    m_device = &d;
    m_mipLevels = mipLevels;
    if (mipLevels < 2) return;

    // set: 0=voxelGrid full sampled, 1=aniso full sampled, 2=aniso dst storage
    std::array<VkDescriptorSetLayoutBinding, 3> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    uint32_t setCount = mipLevels - 1;   // mip 1..mipLevels-1
    std::array<VkDescriptorPoolSize, 2> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, setCount * 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, setCount},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = setCount; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    m_sets.resize(setCount);
    std::vector<VkDescriptorSetLayout> layouts(setCount, m_setLayout);
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = setCount; dai.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, m_sets.data()));

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(AnisoPC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "gi" / "vxgi" / "vxgi_aniso_build.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));
}

void VxgiAnisoPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pipeline)       vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_pool)           vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout)      vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    m_sets.clear();
    *this = {};
}

void VxgiAnisoPass::bindResources(Device& d, const VxgiResources& vxgi) {
    for (uint32_t dstLevel = 1; dstLevel < m_mipLevels; ++dstLevel) {
        VkDescriptorImageInfo voxF{};
        voxF.imageView = vxgi.fullView();
        voxF.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        // 用 per-mip view 而非 full view —— 描述符 layout 检查只覆盖被
        // 描述的那一级 mip，避开"其它 mip 还在 GENERAL"的 validation 噪声。
        // 对 dstLevel=1（不读 aniso）也绑一个有效 view（mip 0）当占位。
        VkDescriptorImageInfo anisoF{};
        uint32_t srcMip = (dstLevel >= 2) ? (dstLevel - 1) : 0;
        anisoF.imageView = vxgi.anisoMipView(srcMip);
        anisoF.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo dst{};
        dst.imageView = vxgi.anisoMipView(dstLevel);
        dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 3> w{};
        auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorImageInfo* p) {
            W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            W.dstSet = m_sets[dstLevel - 1]; W.dstBinding = bi; W.descriptorCount = 1;
            W.descriptorType = t; W.pImageInfo = p;
        };
        setImg(w[0], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &voxF);
        setImg(w[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &anisoF);
        setImg(w[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dst);
        vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
    }
}

void VxgiAnisoPass::record(VkCommandBuffer cmd, const VxgiResources& vxgi) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

    auto barrierAnisoMip = [&](uint32_t mip,
                               VkImageLayout oldL, VkImageLayout newL,
                               VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                               VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
        b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
        b.oldLayout = oldL; b.newLayout = newL;
        b.image = vxgi.aniso().image();
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    };

    uint32_t resolution = vxgi.resolution();
    for (uint32_t dstLevel = 1; dstLevel < m_mipLevels; ++dstLevel) {
        // 入口约定：所有 aniso mip 都在 SHADER_READ_ONLY（App 端 pre-pass
        // 已统一转过；每 iter 自己 dst 写完也回到 SR_O）。
        // 1. dst mip SHADER_READ_ONLY → GENERAL（写）
        barrierAnisoMip(dstLevel,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        uint32_t useAniso = (dstLevel == 1) ? 0u : 1u;
        uint32_t srcLevel = (dstLevel == 1) ? 0u : (dstLevel - 1);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0, 1, &m_sets[dstLevel - 1], 0, nullptr);

        AnisoPC pc{};
        pc.dstSize = resolution >> dstLevel;
        if (pc.dstSize == 0) pc.dstSize = 1;
        pc.useAnisoSrc = useAniso;
        pc.srcLevel = srcLevel;
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        uint32_t g = (pc.dstSize + 3) / 4;
        vkCmdDispatch(cmd, g, g, g);

        // 2. dst mip GENERAL → SHADER_READ_ONLY（给下 iter 当 src 或最终
        //    lighting 用）
        barrierAnisoMip(dstLevel,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
}

}
