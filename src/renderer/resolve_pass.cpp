#include "resolve_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include "render_targets.h"
#include "scene/scene_gpu.h"
#include <array>

namespace somegi {

namespace {
struct ResolvePC {
    uint32_t outSizeX, outSizeY;
    float    invOutSizeX, invOutSizeY;
};
static_assert(sizeof(ResolvePC) == 16, "ResolvePC must match shader");
}

void ResolvePass::init(Device& d, uint32_t maxTextures) {
    m_device = &d;

    // set=0:
    // 0: gFrame           (UBO FrameUniforms)
    // 1: gVisBuffer       (sampled R32G32_UINT)
    // 2: gVertices        (SSBO, float array)
    // 3: gIndices         (SSBO, uint array)
    // 4: gMaterials       (SSBO, MaterialGpu array)
    // 5: gLinear          (sampler)
    // 6: gTextures        (sampled image array)
    // 7: gOutAlbedoMetal  (storage R8G8B8A8_UNORM)
    // 8: gOutNormalRough  (storage R16G16B16A16_SFLOAT)
    // 9: gOutEmissiveAO   (storage R8G8B8A8_UNORM)
    std::array<VkDescriptorSetLayoutBinding, 10> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,   1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_SAMPLER,          1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[6] = {6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,    maxTextures,  VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[7] = {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,    1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[8] = {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,    1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[9] = {9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,    1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    // Partially-bound for textures array (binding 6)
    std::array<VkDescriptorBindingFlags, 10> bf{};
    bf[6] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bfci.bindingCount = (uint32_t)bf.size(); bfci.pBindingFlags = bf.data();
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.pNext = &bfci;
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ResolvePC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "gbuffer" / "vis_resolve.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));

    std::array<VkDescriptorPoolSize, 5> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  1u * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,   (1u + maxTextures) * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  3u * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_SAMPLER,         1u * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,   3u * kFramesInFlight},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = kFramesInFlight; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetLayout layouts[] = {m_setLayout, m_setLayout};
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = kFramesInFlight; dai.pSetLayouts = layouts;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, m_sets));
}

void ResolvePass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pool)           vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_pipeline)       vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_setLayout)      vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    m_device = nullptr;
}

void ResolvePass::bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount,
                             VkBuffer frameUbo) {
    VkDescriptorBufferInfo uboInfo{frameUbo, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo vertInfo{gpu.vertexBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo idxInfo{gpu.indexBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo matInfo{gpu.materialBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo samp{gpu.linearSampler};

    std::vector<VkDescriptorImageInfo> texInfos(textureCount);
    for (uint32_t i = 0; i < textureCount; ++i) {
        texInfos[i].imageView = gpu.images[i].view();
        texInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    for (uint32_t fi = 0; fi < kFramesInFlight; ++fi) {
        VkWriteDescriptorSet wUbo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wUbo.dstSet = m_sets[fi]; wUbo.dstBinding = 0; wUbo.descriptorCount = 1;
        wUbo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; wUbo.pBufferInfo = &uboInfo;
        vkUpdateDescriptorSets(d.device(), 1, &wUbo, 0, nullptr);

        std::array<VkWriteDescriptorSet, 3> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = m_sets[fi]; w[0].dstBinding = 2; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &vertInfo;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = m_sets[fi]; w[1].dstBinding = 3; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &idxInfo;
        w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[2].dstSet = m_sets[fi]; w[2].dstBinding = 4; w[2].descriptorCount = 1;
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[2].pBufferInfo = &matInfo;
        vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);

        std::array<VkWriteDescriptorSet, 2> w2{};
        w2[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w2[0].dstSet = m_sets[fi]; w2[0].dstBinding = 5; w2[0].descriptorCount = 1;
        w2[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w2[0].pImageInfo = &samp;
        w2[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w2[1].dstSet = m_sets[fi]; w2[1].dstBinding = 6; w2[1].descriptorCount = textureCount;
        w2[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w2[1].pImageInfo = texInfos.data();
        vkUpdateDescriptorSets(d.device(), (uint32_t)w2.size(), w2.data(), 0, nullptr);
    }
}

void ResolvePass::bindTargets(Device& d, const RenderTargets& rt, uint32_t frameIdx) {
    VkDescriptorImageInfo visInfo{};
    visInfo.imageView = rt.visBuffer.view();
    visInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo albedoInfo{};
    albedoInfo.imageView = rt.gAlbedoMetal.view();
    albedoInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo normalInfo{};
    normalInfo.imageView = rt.gNormalRough.view();
    normalInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo emissiveInfo{};
    emissiveInfo.imageView = rt.gEmissiveAO.view();
    emissiveInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 4> w{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_sets[frameIdx]; w[0].dstBinding = 1; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &visInfo;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[1].dstSet = m_sets[frameIdx]; w[1].dstBinding = 7; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &albedoInfo;
    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[2].dstSet = m_sets[frameIdx]; w[2].dstBinding = 8; w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo = &normalInfo;
    w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[3].dstSet = m_sets[frameIdx]; w[3].dstBinding = 9; w[3].descriptorCount = 1;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[3].pImageInfo = &emissiveInfo;
    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void ResolvePass::record(VkCommandBuffer cmd, const RenderTargets& rt, uint32_t frameIdx) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_sets[frameIdx], 0, nullptr);

    ResolvePC pc{};
    pc.outSizeX = rt.extent.width;  pc.outSizeY = rt.extent.height;
    pc.invOutSizeX = 1.0f / (float)rt.extent.width;
    pc.invOutSizeY = 1.0f / (float)rt.extent.height;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t gx = (rt.extent.width  + 7) / 8;
    uint32_t gy = (rt.extent.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);
}

}
