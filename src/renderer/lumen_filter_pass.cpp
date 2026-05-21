#include "lumen_filter_pass.h"
#include "lumen_resources.h"
#include "render_targets.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>

namespace somegi {

namespace {
struct FilterPC {
    float    invScreenSizeX, invScreenSizeY;
    float    probeTileSize;
    uint32_t probeGridW;
    uint32_t probeGridH;
    float    sigmaDepth;
    float    normalPower;
    float    sigmaDist;
    float    temporalAlpha;
};
static_assert(sizeof(FilterPC) == 36, "FilterPC must match shader push constant layout");
}

void LumenFilterPass::init(Device& d) {
    m_device = &d;

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_NEAREST; si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_pointClamp));

    // 0:FrameUBO 1:NormalRough 2:Depth 3:ProbeAtlas(in)
    // 4:FilteredAtlas(out) 5:PrevAtlas(in) 6:Sampler
    std::array<VkDescriptorSetLayoutBinding, 7> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[6] = {6, VK_DESCRIPTOR_TYPE_SAMPLER,        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    std::array<VkDescriptorPoolSize, 4> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  4},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1},
        {VK_DESCRIPTOR_TYPE_SAMPLER,        1},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_set));

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(FilterPC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "gi" / "lumen" / "lumen_filter.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle();
    stage.pName = "cs_spatialFilter";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));
}

void LumenFilterPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pipeline)       vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_pool)           vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout)      vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    if (m_pointClamp)     vkDestroySampler(dev, m_pointClamp, nullptr);
    *this = {};
}

void LumenFilterPass::bindResources(Device& d, const LumenResources& res,
                                     const RenderTargets& rt, VkBuffer frameUbo) {
    VkDescriptorBufferInfo ub{frameUbo, 0, VK_WHOLE_SIZE};

    VkDescriptorImageInfo nr{};
    nr.imageView = rt.gNormalRough.view();
    nr.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo depth{};
    depth.imageView = rt.depth.view();
    depth.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo inAtlas{};
    inAtlas.imageView = res.probeAtlas().view();
    inAtlas.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo outAtlas{};
    outAtlas.imageView = res.filteredAtlas().view();
    outAtlas.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo prev{};
    prev.imageView = res.prevAtlas().view();
    prev.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo smp{}; smp.sampler = m_pointClamp;

    std::array<VkWriteDescriptorSet, 7> w{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_set; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &ub;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[1].dstSet = m_set; w[1].dstBinding = 1; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[1].pImageInfo = &nr;
    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[2].dstSet = m_set; w[2].dstBinding = 2; w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[2].pImageInfo = &depth;
    w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[3].dstSet = m_set; w[3].dstBinding = 3; w[3].descriptorCount = 1;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[3].pImageInfo = &inAtlas;
    w[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[4].dstSet = m_set; w[4].dstBinding = 4; w[4].descriptorCount = 1;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[4].pImageInfo = &outAtlas;
    w[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[5].dstSet = m_set; w[5].dstBinding = 5; w[5].descriptorCount = 1;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[5].pImageInfo = &prev;
    w[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[6].dstSet = m_set; w[6].dstBinding = 6; w[6].descriptorCount = 1;
    w[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[6].pImageInfo = &smp;

    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void LumenFilterPass::record(VkCommandBuffer cmd, const LumenResources& res,
                              const RenderTargets& rt) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

    FilterPC pc{};
    pc.invScreenSizeX = 1.0f / (float)rt.extent.width;
    pc.invScreenSizeY = 1.0f / (float)rt.extent.height;
    pc.probeTileSize  = (float)LumenResources::kProbeTileSize;
    pc.probeGridW     = res.probeGridW();
    pc.probeGridH     = res.probeGridH();
    pc.sigmaDepth     = sigmaDepth;
    pc.normalPower    = normalPower;
    pc.sigmaDist      = sigmaDist;
    pc.temporalAlpha  = temporalAlpha;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t groups = (res.probeCount() + 63) / 64;
    vkCmdDispatch(cmd, groups, 1, 1);
}

}
