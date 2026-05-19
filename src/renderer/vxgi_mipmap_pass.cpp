#include "vxgi_mipmap_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include "core/image.h"
#include <array>

namespace somegi {

namespace {
struct MipmapPC {
    uint32_t dstSize;
    uint32_t _p0, _p1, _p2;
};
}

void VxgiMipmapPass::init(Device& d, uint32_t mipLevels) {
    m_device = &d;
    m_mipLevels = mipLevels;
    if (mipLevels < 2) return;   // 没 mip 1 → 啥都不做

    // set=0：0=src sampled image, 1=dst storage image.
    std::array<VkDescriptorSetLayoutBinding, 2> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    uint32_t setCount = mipLevels - 1;   // mip 1..mipLevels-1
    std::array<VkDescriptorPoolSize, 2> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, setCount},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, setCount},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = setCount; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    m_sets.resize(setCount, VK_NULL_HANDLE);
    std::vector<VkDescriptorSetLayout> layouts(setCount, m_setLayout);
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = setCount; dai.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, m_sets.data()));

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(MipmapPC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "gi" / "vxgi" / "vxgi_mipmap.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));
}

void VxgiMipmapPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pipeline)       vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_pool)           vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout)      vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    m_pipeline = VK_NULL_HANDLE; m_pipelineLayout = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE; m_setLayout = VK_NULL_HANDLE;
    m_sets.clear();
    m_device = nullptr;
}

void VxgiMipmapPass::bindResources(Device& d, const VxgiResources& vxgi) {
    for (uint32_t dstLevel = 1; dstLevel < m_mipLevels; ++dstLevel) {
        VkDescriptorImageInfo src{};
        src.imageView = vxgi.mipView(dstLevel - 1);
        src.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo dst{};
        dst.imageView = vxgi.mipView(dstLevel);
        dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 2> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = m_sets[dstLevel - 1]; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &src;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = m_sets[dstLevel - 1]; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &dst;
        vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
    }
}

void VxgiMipmapPass::record(VkCommandBuffer cmd, const VxgiResources& vxgi) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

    auto barrierMip = [&](uint32_t mip,
                          VkImageLayout oldL, VkImageLayout newL,
                          VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                          VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
        b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
        b.oldLayout = oldL; b.newLayout = newL;
        b.image = vxgi.image().image();
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    };

    uint32_t resolution = vxgi.resolution();
    for (uint32_t dstLevel = 1; dstLevel < m_mipLevels; ++dstLevel) {
        // src 已经是 GENERAL（上一级 dispatch 写完）或 GENERAL（上一帧 mip 0
        // 是 inject/voxelize 写完的初值）。统一转 SHADER_READ_ONLY 才能用
        // SAMPLED_IMAGE 描述符读。
        barrierMip(dstLevel - 1,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        // dst 之前在 GENERAL（clear 阶段已转）或 UNDEFINED；统一保持
        // GENERAL（不需要 transition，但要 barrier 强制 clear write 完成）。
        barrierMip(dstLevel,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0, 1, &m_sets[dstLevel - 1], 0, nullptr);

        uint32_t dstSize = resolution >> dstLevel;
        if (dstSize == 0) dstSize = 1;
        MipmapPC pc{dstSize, 0, 0, 0};
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        uint32_t g = (dstSize + 3) / 4;
        vkCmdDispatch(cmd, g, g, g);
    }
}

}
