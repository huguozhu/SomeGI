#include "prt_bake_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>

namespace somegi {

namespace {
struct BakePC {
    float    prtGridMinX, prtGridMinY, prtGridMinZ;
    float    prtCellSize;
    uint32_t prtResolution;
    uint32_t numSamples;
    float    voxelGridDim;
    uint32_t _pad0;
    float    vxgiGridMinX, vxgiGridMinY, vxgiGridMinZ;
    float    vxgiCellSize;
    uint32_t vxgiResolution;
    uint32_t _pad1, _pad2, _pad3;
};
static_assert(sizeof(BakePC) == 64, "BakePC must match shader push constant layout");
}

void PrtBakePass::init(Device& d) {
    m_device = &d;

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_linearClamp));

    // 0: voxel sampled, 1: sampler, 2..6: prt transfer A/B/C/D/E storage（SH16 五张 atlas）
    std::array<VkDescriptorSetLayoutBinding, 7> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER,       1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[6] = {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    std::array<VkDescriptorPoolSize, 3> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLER,       1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5},   // SH16 五张 atlas
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_set));

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(BakePC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "gi" / "prt" / "prt_bake.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));
}

void PrtBakePass::destroy() {
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

void PrtBakePass::bindResources(Device& d, const VxgiResources& vxgi, const PrtResources& prt) {
    VkDescriptorImageInfo vox{};
    vox.imageView = vxgi.fullView();
    vox.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo smp{}; smp.sampler = m_linearClamp;
    auto storageGen = [](VkImageView v) {
        VkDescriptorImageInfo i{};
        i.imageView = v; i.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        return i;
    };
    VkDescriptorImageInfo prxA = storageGen(prt.view());
    VkDescriptorImageInfo prxB = storageGen(prt.viewB());
    VkDescriptorImageInfo prxC = storageGen(prt.viewC());
    VkDescriptorImageInfo prxD = storageGen(prt.viewD());
    VkDescriptorImageInfo prxE = storageGen(prt.viewE());

    std::array<VkWriteDescriptorSet, 7> w{};
    auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorImageInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_set; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = t; W.pImageInfo = p;
    };
    setImg(w[0], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &vox);
    setImg(w[1], 1, VK_DESCRIPTOR_TYPE_SAMPLER,       &smp);
    setImg(w[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &prxA);
    setImg(w[3], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &prxB);
    setImg(w[4], 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &prxC);
    setImg(w[5], 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &prxD);
    setImg(w[6], 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &prxE);
    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void PrtBakePass::record(VkCommandBuffer cmd,
                         const glm::vec3& prtGridMin, float prtCellSize, uint32_t prtResolution,
                         const glm::vec3& vxgiGridMin, float vxgiCellSize, uint32_t vxgiResolution,
                         uint32_t numSamples) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

    BakePC pc{};
    pc.prtGridMinX = prtGridMin.x; pc.prtGridMinY = prtGridMin.y; pc.prtGridMinZ = prtGridMin.z;
    pc.prtCellSize = prtCellSize;
    pc.prtResolution = prtResolution;
    pc.numSamples = numSamples;
    pc.voxelGridDim = vxgiCellSize * float(vxgiResolution);
    pc.vxgiGridMinX = vxgiGridMin.x; pc.vxgiGridMinY = vxgiGridMin.y; pc.vxgiGridMinZ = vxgiGridMin.z;
    pc.vxgiCellSize = vxgiCellSize;
    pc.vxgiResolution = vxgiResolution;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t g = (prtResolution + 3) / 4;
    vkCmdDispatch(cmd, g, g, g);
}

}
