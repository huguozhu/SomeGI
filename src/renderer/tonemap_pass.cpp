#include "tonemap_pass.h"
#include "core/device.h"
#include <array>

namespace somegi {

namespace {
struct TonemapPC {
    uint32_t hdrMode;
    float    exposure;
    uint32_t pad0, pad1;
};
static_assert(sizeof(TonemapPC) == 16, "TonemapPC must match shader push constant layout");
}

void TonemapPass::init(Device& d, VkSampler linearSampler) {
    m_device = &d;
    m_sampler = linearSampler;

    std::array<VkDescriptorSetLayoutBinding, 3> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER,       1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TonemapPC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    auto sd = shaderDir();
    ShaderModule cs(d, sd / "tonemap" / "tonemap.spv");

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs.handle();
    stage.pName = "cs_main";

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage;
    cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));

    std::array<VkDescriptorPoolSize, 3> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_SAMPLER,       1u * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1u * kFramesInFlight},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = kFramesInFlight; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetLayout layouts[] = {m_setLayout, m_setLayout};
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = kFramesInFlight; dai.pSetLayouts = layouts;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, m_sets));
}

void TonemapPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pool) vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_pipeline) vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_setLayout) vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    m_device = nullptr;
}

void TonemapPass::bindOutput(Device& d, VkImageView outView, uint32_t frameIdx) {
    VkDescriptorImageInfo ldrInfo{};
    ldrInfo.imageView = outView;
    ldrInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = m_sets[frameIdx]; w.dstBinding = 2; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w.pImageInfo = &ldrInfo;
    vkUpdateDescriptorSets(d.device(), 1, &w, 0, nullptr);
}

void TonemapPass::bindTargets(Device& d, const RenderTargets& rt) {
    VkDescriptorImageInfo hdrInfo{};
    hdrInfo.imageView = rt.hdrColor.view();
    hdrInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo sampInfo{};
    sampInfo.sampler = m_sampler;

    VkDescriptorImageInfo ldrInfo{};
    ldrInfo.imageView = rt.ldrTonemap.view();
    ldrInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    for (uint32_t fi = 0; fi < kFramesInFlight; ++fi) {
        std::array<VkWriteDescriptorSet, 3> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = m_sets[fi]; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &hdrInfo;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = m_sets[fi]; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[1].pImageInfo = &sampInfo;
        w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[2].dstSet = m_sets[fi]; w[2].dstBinding = 2; w[2].descriptorCount = 1;
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo = &ldrInfo;
        vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
    }
}

void TonemapPass::record(VkCommandBuffer cmd, const RenderTargets& rt, uint32_t frameIdx,
                          bool hdrMode, float exposure) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_pipelineLayout, 0, 1, &m_sets[frameIdx], 0, nullptr);

    TonemapPC tpc{};
    tpc.hdrMode = hdrMode ? 1u : 0u;
    tpc.exposure = exposure;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(tpc), &tpc);

    uint32_t gx = (rt.extent.width  + 7) / 8;
    uint32_t gy = (rt.extent.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);
}

}
