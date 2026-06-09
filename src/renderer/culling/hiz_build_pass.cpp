#include "renderer/culling/hiz_build_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include "renderer/core/render_targets.h"
#include <array>
#include <cstring>
#include <algorithm>

namespace somegi {

static Image mkHiZMip(Device& d, uint32_t w, uint32_t h) {
    ImageDesc id{};
    id.format = VK_FORMAT_R32_SFLOAT;
    id.extent = {w, h, 1};
    id.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    return Image(d, id);
}

void HiZBuildPass::init(Device& d, VkExtent2D extent) {
    m_device = &d; m_extent = extent;

    uint32_t w = extent.width, h = extent.height;
    m_mip1 = mkHiZMip(d, std::max(1u, w/2), std::max(1u, h/2));
    m_mip2 = mkHiZMip(d, std::max(1u, w/4), std::max(1u, h/4));
    m_mip3 = mkHiZMip(d, std::max(1u, w/8), std::max(1u, h/8));
    m_mip4 = mkHiZMip(d, std::max(1u, w/16), std::max(1u, h/16));

    // Descriptor layout: 0=depth SR_O, 1-4=storage images
    std::array<VkDescriptorSetLayoutBinding, 5> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT};
    for (uint32_t i = 1; i <= 4; ++i)
        b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 5, b.data()};
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_dsl));

    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &m_dsl, 1, &pc};
    VK_CHECK(vkCreatePipelineLayout(d.device(), &pli, nullptr, &m_pl));

    ShaderModule sh(d, shaderDir() / "culling" / "hiz_build.spv");
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.module = sh.handle(); cp.stage.pName = "cs_main";
    cp.layout = m_pl;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cp, nullptr, &m_pipe));

    std::array<VkDescriptorPoolSize, 2> ps = {{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0, m_pool, 1, &m_dsl};
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &ai, &m_set));
}

void HiZBuildPass::destroy() {
    if (!m_device) return; auto dev = m_device->device();
    if (m_pipe) vkDestroyPipeline(dev, m_pipe, nullptr);
    if (m_pl) vkDestroyPipelineLayout(dev, m_pl, nullptr);
    if (m_pool) vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_dsl) vkDestroyDescriptorSetLayout(dev, m_dsl, nullptr);
    m_mip1.reset(); m_mip2.reset(); m_mip3.reset(); m_mip4.reset();
    m_device = nullptr;
}

void HiZBuildPass::record(VkCommandBuffer cmd, const RenderTargets& rt) {
    auto dev = m_device->device();

    // Write descriptors
    VkDescriptorImageInfo depthI{};
    depthI.imageView = rt.depth.view();
    depthI.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    auto stor = [](VkImageView v) {
        VkDescriptorImageInfo i{}; i.imageView = v; i.imageLayout = VK_IMAGE_LAYOUT_GENERAL; return i;
    };
    VkDescriptorImageInfo m1 = stor(m_mip1.view()), m2 = stor(m_mip2.view());
    VkDescriptorImageInfo m3 = stor(m_mip3.view()), m4 = stor(m_mip4.view());

    std::array<VkWriteDescriptorSet, 5> w{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[0].dstSet = m_set; w[0].dstBinding = 0;
    w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &depthI;
    for (uint32_t i = 1; i <= 4; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = m_set; w[i].dstBinding = i;
        w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    w[1].pImageInfo = &m1; w[2].pImageInfo = &m2; w[3].pImageInfo = &m3; w[4].pImageInfo = &m4;
    vkUpdateDescriptorSets(dev, 5, w.data(), 0, nullptr);

    // Barrier: depth SR_O → SR_O (execution only, no layout change)
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    b.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.image = rt.depth.image();
    b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pl, 0, 1, &m_set, 0, nullptr);

    struct { uint32_t w, h; } pc{m_extent.width, m_extent.height};
    vkCmdPushConstants(cmd, m_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &pc);
    vkCmdDispatch(cmd, (m_extent.width+15)/16, (m_extent.height+15)/16, 1);
}

VkExtent2D HiZBuildPass::mipExtent(uint32_t level) const {
    uint32_t div = 1u << std::min(level, 4u);
    return {std::max(1u, m_extent.width/div), std::max(1u, m_extent.height/div)};
}

} // namespace somegi
