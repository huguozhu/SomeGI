#include "taa_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include "render_targets.h"
#include <array>

namespace somegi {

namespace {
struct TaaPC {
    float jitterX, jitterY;
    float prevJitterX, prevJitterY;
    glm::mat4 invViewProj;
    glm::mat4 prevViewProj;
    float blendAlpha;
    float _pad;       // std140: align float2 invRes to 8-byte boundary
    float invResX, invResY;
};
static_assert(sizeof(TaaPC) == 160, "TaaPC must match shader push constant layout");
}

void TaaPass::init(Device& d) {
    m_device = &d;

    std::array<VkDescriptorSetLayoutBinding, 4> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(TaaPC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "aa" / "taa.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));

    std::array<VkDescriptorPoolSize, 2> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3u * kFramesInFlight},
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

void TaaPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pool)           vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_pipeline)       vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_setLayout)      vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    *this = {};
}

void TaaPass::bindResources(Device& d, const RenderTargets& rt, uint32_t frameIdx) {
    VkDescriptorImageInfo curr{};
    curr.imageView = rt.aaHdr.view();
    curr.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo hist{};
    hist.imageView = rt.aaHistory.view();
    hist.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo depth{};
    depth.imageView = rt.depth.view();
    depth.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo out{};
    out.imageView = rt.ldrTonemap.view();
    out.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 4> w{};
    auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorImageInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_sets[frameIdx]; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = t; W.pImageInfo = p;
    };
    setImg(w[0], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &curr);
    setImg(w[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &hist);
    setImg(w[2], 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &depth);
    setImg(w[3], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &out);
    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void TaaPass::record(VkCommandBuffer cmd, const RenderTargets& rt,
                     const glm::vec2& jitter, const glm::vec2& prevJitter,
                     const glm::mat4& invViewProj, const glm::mat4& prevViewProj,
                     uint32_t frameIdx, float blendAlpha) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_sets[frameIdx], 0, nullptr);

    TaaPC pc{};
    pc.jitterX = jitter.x; pc.jitterY = jitter.y;
    pc.prevJitterX = prevJitter.x; pc.prevJitterY = prevJitter.y;
    pc.invViewProj = invViewProj;
    pc.prevViewProj = prevViewProj;
    pc.blendAlpha = blendAlpha;
    pc.invResX = 1.0f / rt.extent.width;
    pc.invResY = 1.0f / rt.extent.height;

    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t gx = (rt.extent.width  + 7) / 8;
    uint32_t gy = (rt.extent.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);
}

}
