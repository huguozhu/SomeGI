#include "renderer/gi/lpv/lpv_propagate_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>

namespace somegi {

namespace {
struct PropagatePC {
    uint32_t gridResolution;
    float    occlusionAmplifier;
    float    gvOcclusionStrength;   // B.8
    uint32_t _p1;
};
static_assert(sizeof(PropagatePC) == 16, "PropagatePC must match shader push constant layout");
}

void LpvPropagatePass::init(Device& d) {
    m_device = &d;

    // 0..2: sampled src R/G/B, 3..5: storage dst R/G/B, 6: sampled GV (B.8)
    std::array<VkDescriptorSetLayoutBinding, 7> b{};
    for (uint32_t i = 0; i < 7; ++i) {
        b[i].binding = i;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bool sampledKind = (i < 3) || (i == 6);
        b[i].descriptorType = sampledKind ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                          : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    // 两组 set：grid0 → grid1 / grid1 → grid0。
    std::array<VkDescriptorPoolSize, 2> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8},   // 2 sets × (3 src + 1 GV)
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 2; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetLayout layouts[2] = {m_setLayout, m_setLayout};
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 2; dai.pSetLayouts = layouts;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, m_sets));

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(PropagatePC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "gi" / "lpv" / "lpv_propagate.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));
}

void LpvPropagatePass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pipeline)       vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_pool)           vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout)      vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    m_pipeline = VK_NULL_HANDLE; m_pipelineLayout = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE; m_setLayout = VK_NULL_HANDLE;
    m_sets[0] = m_sets[1] = VK_NULL_HANDLE;
    m_device = nullptr;
}

void LpvPropagatePass::bindResources(Device& d, const LpvGrid& g0, const LpvGrid& g1,
                                      const Image& gv) {
    auto sampledRO = [](VkImageView v) {
        VkDescriptorImageInfo i{};
        i.imageView = v; i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return i;
    };
    auto storageGen = [](VkImageView v) {
        VkDescriptorImageInfo i{};
        i.imageView = v; i.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        return i;
    };

    auto writeOne = [&](VkDescriptorSet set, const LpvGrid& src, const LpvGrid& dst) {
        VkDescriptorImageInfo sR = sampledRO(src.lpvR.view());
        VkDescriptorImageInfo sG = sampledRO(src.lpvG.view());
        VkDescriptorImageInfo sB = sampledRO(src.lpvB.view());
        VkDescriptorImageInfo dR = storageGen(dst.lpvR.view());
        VkDescriptorImageInfo dG = storageGen(dst.lpvG.view());
        VkDescriptorImageInfo dB = storageGen(dst.lpvB.view());
        VkDescriptorImageInfo gvI = sampledRO(gv.view());
        std::array<VkWriteDescriptorSet, 7> w{};
        auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorImageInfo* p) {
            W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            W.dstSet = set; W.dstBinding = bi; W.descriptorCount = 1;
            W.descriptorType = t; W.pImageInfo = p;
        };
        setImg(w[0], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &sR);
        setImg(w[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &sG);
        setImg(w[2], 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &sB);
        setImg(w[3], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dR);
        setImg(w[4], 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dG);
        setImg(w[5], 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dB);
        setImg(w[6], 6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gvI);
        vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
    };

    writeOne(m_sets[0], g0, g1);   // src=g0, dst=g1
    writeOne(m_sets[1], g1, g0);   // src=g1, dst=g0
}

void LpvPropagatePass::record(VkCommandBuffer cmd, int srcIdx, uint32_t gridResolution,
                              float occlusionAmp, float gvOcclusionStr) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_sets[srcIdx & 1], 0, nullptr);

    PropagatePC pc{};
    pc.gridResolution = gridResolution;
    pc.occlusionAmplifier = occlusionAmp;
    pc.gvOcclusionStrength = gvOcclusionStr;
    pc._p1 = 0;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t g = (gridResolution + 3) / 4;
    vkCmdDispatch(cmd, g, g, g);
}

}
