// RsmSamplePass 实现 —— compute pass，描述符布局比 SsgiPass 大一圈：
// 多了 3 张 RSM 输入（pos/normal/flux）+ 1 个 RsmFrameUbo。

#include "rsm_sample_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>

namespace somegi {

namespace {
// 与 shaders/gi/rsm/rsm_sample.slang 中 RsmSamplePC 严格对齐。
struct RsmSamplePC {
    uint32_t outSizeX, outSizeY;
    float    invOutSizeX, invOutSizeY;
    float    radius;
    uint32_t sampleCount;
    float    intensity;
    uint32_t _pad;
};
}

void RsmSamplePass::init(Device& d) {
    m_device = &d;

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_linearClamp));

    // set=0 layout，9 个 binding（详见 .h 注释）。
    std::array<VkDescriptorSetLayoutBinding, 9> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[6] = {6, VK_DESCRIPTOR_TYPE_SAMPLER,        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[7] = {7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[8] = {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    std::array<VkDescriptorPoolSize, 4> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  5},
        {VK_DESCRIPTOR_TYPE_SAMPLER,        1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_set));

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(RsmSamplePC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "gi" / "rsm" / "rsm_sample.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));
}

void RsmSamplePass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pipeline)       vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_pool)           vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout)      vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    if (m_linearClamp)    vkDestroySampler(dev, m_linearClamp, nullptr);
    m_pipeline = VK_NULL_HANDLE; m_pipelineLayout = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE; m_setLayout = VK_NULL_HANDLE;
    m_linearClamp = VK_NULL_HANDLE;
    m_device = nullptr;
}

void RsmSamplePass::bindFrame(Device& d, const RenderTargets& rt,
                              VkBuffer frameUbo, VkBuffer rsmFrameUbo,
                              const Image& rsmPos, const Image& rsmN, const Image& rsmFlux) {
    VkDescriptorBufferInfo frameInfo{frameUbo, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo rsmInfo{rsmFrameUbo, 0, VK_WHOLE_SIZE};
    auto sampledRO = [](VkImageView v) {
        VkDescriptorImageInfo i{};
        i.imageView = v; i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return i;
    };
    VkDescriptorImageInfo nr = sampledRO(rt.gNormalRough.view());
    VkDescriptorImageInfo dp = sampledRO(rt.depth.view());
    VkDescriptorImageInfo rp = sampledRO(rsmPos.view());
    VkDescriptorImageInfo rn = sampledRO(rsmN.view());
    VkDescriptorImageInfo rf = sampledRO(rsmFlux.view());
    VkDescriptorImageInfo smp{}; smp.sampler = m_linearClamp;
    VkDescriptorImageInfo gi{};
    gi.imageView = rt.rsmGI.view();
    gi.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 9> w{};
    auto setBuf = [&](VkWriteDescriptorSet& W, uint32_t bi, const VkDescriptorBufferInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_set; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; W.pBufferInfo = p;
    };
    auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorImageInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_set; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = t; W.pImageInfo = p;
    };
    setBuf(w[0], 0, &frameInfo);
    setImg(w[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &nr);
    setImg(w[2], 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &dp);
    setImg(w[3], 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &rp);
    setImg(w[4], 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &rn);
    setImg(w[5], 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &rf);
    setImg(w[6], 6, VK_DESCRIPTOR_TYPE_SAMPLER,       &smp);
    setBuf(w[7], 7, &rsmInfo);
    setImg(w[8], 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gi);
    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void RsmSamplePass::record(VkCommandBuffer cmd, const RenderTargets& rt) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

    RsmSamplePC pc{};
    pc.outSizeX = rt.extent.width;  pc.outSizeY = rt.extent.height;
    pc.invOutSizeX = 1.0f / (float)rt.extent.width;
    pc.invOutSizeY = 1.0f / (float)rt.extent.height;
    pc.radius      = radius;
    pc.sampleCount = (uint32_t)sampleCount;
    pc.intensity   = intensity;
    pc._pad        = 0;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t gx = (rt.extent.width  + 7) / 8;
    uint32_t gy = (rt.extent.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);
}

}
