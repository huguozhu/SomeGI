#include "vxgi_resolve_6axis_pass.h"
#include "vxgi_resources.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>

namespace somegi {

namespace {
struct SixAxisPC {
    uint32_t gridResolution;
    uint32_t mipLevels;
    float    cellSize;
    float    strength;
    float    gridMinX, gridMinY, gridMinZ;
    float    _pad0;
};
static_assert(sizeof(SixAxisPC) == 32, "SixAxisPC must match shader push constant layout");
}

void VxgiResolve6AxisPass::init(Device& d) {
    m_device = &d;

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 16.0f;
    VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_linearClamp));

    // 0:voxel src, 1:aniso src, 2:sampler, 3:axisX dst, 4:axisY dst, 5:axisZ dst
    std::array<VkDescriptorSetLayoutBinding, 6> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLER,       1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    std::array<VkDescriptorPoolSize, 3> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2},
        {VK_DESCRIPTOR_TYPE_SAMPLER,       1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_set));

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(SixAxisPC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "gi" / "vxgi" / "vxgi_resolve_6axis.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle();
    stage.pName = "cs_resolve6Axis";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));
}

void VxgiResolve6AxisPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pipeline)       vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_pool)           vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout)      vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    if (m_linearClamp)    vkDestroySampler(dev, m_linearClamp, nullptr);
    *this = {};
}

void VxgiResolve6AxisPass::bindResources(Device& d, const VxgiResources& vxgi) {
    VkDescriptorImageInfo vox{};
    vox.imageView = vxgi.fullView();
    vox.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo ani{};
    ani.imageView = vxgi.anisoFullView();
    ani.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo smp{}; smp.sampler = m_linearClamp;
    VkDescriptorImageInfo ax{};
    ax.imageView = vxgi.sixAxisX().view();
    ax.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo ay{};
    ay.imageView = vxgi.sixAxisY().view();
    ay.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo az{};
    az.imageView = vxgi.sixAxisZ().view();
    az.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 6> w{};
    auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorImageInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_set; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = t; W.pImageInfo = p;
    };
    setImg(w[0], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &vox);
    setImg(w[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ani);
    setImg(w[2], 2, VK_DESCRIPTOR_TYPE_SAMPLER,       &smp);
    setImg(w[3], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ax);
    setImg(w[4], 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ay);
    setImg(w[5], 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &az);
    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void VxgiResolve6AxisPass::record(VkCommandBuffer cmd, uint32_t gridResolution,
                                   uint32_t mipLevels, float cellSize,
                                   const glm::vec3& gridMin, float strength) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

    SixAxisPC pc{};
    pc.gridResolution = gridResolution;
    pc.mipLevels = mipLevels;
    pc.cellSize = cellSize;
    pc.strength = strength;
    pc.gridMinX = gridMin.x; pc.gridMinY = gridMin.y; pc.gridMinZ = gridMin.z;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t g = (gridResolution + 3) / 4;
    vkCmdDispatch(cmd, g, g, g);
}

}
