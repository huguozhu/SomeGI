// SsaoPass 实现 —— 算法详见 shaders/ssao/ssao.slang，类设计见 ssao_pass.h。
// 这里只做 Vulkan 描述符 / 管线 / 录制的封装。
// set=0 的 3 个 binding：gNormalRough / gDepth (sampled, 用 .Load)
// + 输出 gOutAO (storage R8)。push constant 三矩阵 + 四个标量参数。

#include "ssao_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>
#include <cstring>

namespace somegi {

namespace {
// shader 端 SsaoPC 与此对齐；mat4 必须先于标量保证 16B 对齐。
struct SsaoPC {
    glm::mat4 proj;
    glm::mat4 invProj;
    glm::mat4 view;
    uint32_t  outSizeX, outSizeY;
    float     invOutSizeX, invOutSizeY;
    float     radius;
    float     bias;
    uint32_t  sampleCount;
    uint32_t  _pad;
};
static_assert(sizeof(SsaoPC) == 224, "SsaoPC must match shader push constant layout");
}

void SsaoPass::init(Device& d) {
    m_device = &d;

    // Set=0:
    //  0: gNormalRough (sampled)
    //  1: gDepth       (sampled, depth aspect)
    //  2: gOutAO       (storage R8)
    std::array<VkDescriptorSetLayoutBinding, 3> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    std::array<VkDescriptorPoolSize, 2> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_set));

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(SsaoPC);

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "ssao" / "ssao.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs.handle();
    stage.pName = "cs_main";

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));
}

void SsaoPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pipeline)       vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_pool)           vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout)      vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    m_pipeline = VK_NULL_HANDLE; m_pipelineLayout = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE; m_setLayout = VK_NULL_HANDLE;
    m_device = nullptr;
}

void SsaoPass::bindFrame(Device& d, const RenderTargets& rt) {
    auto sampledRO = [](VkImageView v) {
        VkDescriptorImageInfo i{};
        i.imageView = v;
        i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return i;
    };
    VkDescriptorImageInfo nr = sampledRO(rt.gNormalRough.view());
    VkDescriptorImageInfo dp = sampledRO(rt.depth.view());
    VkDescriptorImageInfo ao{};
    ao.imageView = rt.ssao.view();
    ao.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 3> w{};
    auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorImageInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_set; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = t; W.pImageInfo = p;
    };
    setImg(w[0], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &nr);
    setImg(w[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &dp);
    setImg(w[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ao);
    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void SsaoPass::record(VkCommandBuffer cmd, const RenderTargets& rt,
                      const glm::mat4& proj, const glm::mat4& invProj,
                      const glm::mat4& view) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

    SsaoPC pc{};
    pc.proj = proj; pc.invProj = invProj; pc.view = view;
    pc.outSizeX = rt.extent.width;
    pc.outSizeY = rt.extent.height;
    pc.invOutSizeX = 1.0f / (float)rt.extent.width;
    pc.invOutSizeY = 1.0f / (float)rt.extent.height;
    pc.radius      = radius;
    pc.bias        = bias;
    pc.sampleCount = (uint32_t)sampleCount;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t gx = (rt.extent.width  + 7) / 8;
    uint32_t gy = (rt.extent.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);
}

}
