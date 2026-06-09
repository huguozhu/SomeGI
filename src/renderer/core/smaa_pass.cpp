#include "renderer/core/smaa_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include "renderer/core/render_targets.h"
#include <array>

namespace somegi {

void SmaaPass::init(Device& d, VkExtent2D ext) {
    m_device = &d;

    // Edge texture (R16G16_SFLOAT — guaranteed storage support)
    ImageDesc edgeDesc{};
    edgeDesc.format = VK_FORMAT_R16G16_SFLOAT;
    edgeDesc.extent = {ext.width, ext.height, 1};
    edgeDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    m_edgeTex = Image(d, edgeDesc);

    // ---- Edge detection pipeline ----
    {
        std::array<VkDescriptorSetLayoutBinding, 2> b{};
        b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_edgeSetLayout));

        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1; plci.pSetLayouts = &m_edgeSetLayout;
        VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_edgePipelineLayout));

        ShaderModule cs(d, shaderDir() / "aa" / "smaa_edge.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = cs.handle(); stage.pName = "cs_main";
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = stage; cpci.layout = m_edgePipelineLayout;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_edgePipeline));

        std::array<VkDescriptorPoolSize, 2> ps{{
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        }};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
        VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_edgePool));

        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = m_edgePool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_edgeSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_edgeSet));
    }

    // ---- Blending pipeline ----
    {
        std::array<VkDescriptorSetLayoutBinding, 3> b{};
        b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_blendSetLayout));

        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1; plci.pSetLayouts = &m_blendSetLayout;
        VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_blendPipelineLayout));

        ShaderModule cs(d, shaderDir() / "aa" / "smaa_blend.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = cs.handle(); stage.pName = "cs_main";
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = stage; cpci.layout = m_blendPipelineLayout;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_blendPipeline));

        std::array<VkDescriptorPoolSize, 2> ps{{
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        }};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
        VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_blendPool));

        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = m_blendPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_blendSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_blendSet));
    }
}

void SmaaPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_blendPool)      vkDestroyDescriptorPool(dev, m_blendPool, nullptr);
    if (m_blendPipeline)  vkDestroyPipeline(dev, m_blendPipeline, nullptr);
    if (m_blendPipelineLayout) vkDestroyPipelineLayout(dev, m_blendPipelineLayout, nullptr);
    if (m_blendSetLayout) vkDestroyDescriptorSetLayout(dev, m_blendSetLayout, nullptr);
    if (m_edgePool)       vkDestroyDescriptorPool(dev, m_edgePool, nullptr);
    if (m_edgePipeline)   vkDestroyPipeline(dev, m_edgePipeline, nullptr);
    if (m_edgePipelineLayout) vkDestroyPipelineLayout(dev, m_edgePipelineLayout, nullptr);
    if (m_edgeSetLayout)  vkDestroyDescriptorSetLayout(dev, m_edgeSetLayout, nullptr);
    m_edgeTex.reset();
    *this = {};
}

void SmaaPass::bindResources(Device& d, const RenderTargets& rt) {
    // Edge detection: bind input + edge output
    {
        VkDescriptorImageInfo input{};
        input.imageView = rt.aaHdr.view();
        input.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo edgeOut{};
        edgeOut.imageView = m_edgeTex.view();
        edgeOut.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 2> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = m_edgeSet; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &input;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = m_edgeSet; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &edgeOut;
        vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
    }

    // Blending: bind input + edge input + output
    {
        VkDescriptorImageInfo input{};
        input.imageView = rt.aaHdr.view();
        input.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo edgeIn{};
        edgeIn.imageView = m_edgeTex.view();
        edgeIn.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo out{};
        out.imageView = rt.ldrTonemap.view();
        out.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 3> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = m_blendSet; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &input;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = m_blendSet; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[1].pImageInfo = &edgeIn;
        w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[2].dstSet = m_blendSet; w[2].dstBinding = 2; w[2].descriptorCount = 1;
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo = &out;
        vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
    }
}

void SmaaPass::bindOutput(Device& d, VkImageView outView) {
    VkDescriptorImageInfo out{};
    out.imageView = outView;
    out.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = m_blendSet; w.dstBinding = 2; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w.pImageInfo = &out;
    vkUpdateDescriptorSets(d.device(), 1, &w, 0, nullptr);
}

void SmaaPass::record(VkCommandBuffer cmd, const RenderTargets& rt) {
    uint32_t gx = (rt.extent.width  + 7) / 8;
    uint32_t gy = (rt.extent.height + 7) / 8;

    auto edgeBarrier = [&](VkImageLayout oldL, VkImageLayout newL,
                           VkAccessFlags2 srcAcc, VkAccessFlags2 dstAcc,
                           VkPipelineStageFlags2 srcStg, VkPipelineStageFlags2 dstStg) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
        b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
        b.oldLayout = oldL; b.newLayout = newL;
        b.image = m_edgeTex.image();
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    };

    // Transition edge texture to GENERAL for pass 1 write
    edgeBarrier(m_edgeLayout, VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    m_edgeLayout = VK_IMAGE_LAYOUT_GENERAL;

    // Pass 1: Edge detection (writes m_edgeTex at GENERAL)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_edgePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_edgePipelineLayout, 0, 1, &m_edgeSet, 0, nullptr);
    vkCmdDispatch(cmd, gx, gy, 1);

    // Barrier: edge GENERAL → SHADER_READ_ONLY for pass 2 read
    edgeBarrier(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    m_edgeLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Pass 2: Edge-guided blending (reads m_edgeTex at SHADER_READ_ONLY)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_blendPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_blendPipelineLayout, 0, 1, &m_blendSet, 0, nullptr);
    vkCmdDispatch(cmd, gx, gy, 1);

    // Transition edge back to GENERAL ready for next frame's pass 1
    edgeBarrier(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
    m_edgeLayout = VK_IMAGE_LAYOUT_GENERAL;
}

}
