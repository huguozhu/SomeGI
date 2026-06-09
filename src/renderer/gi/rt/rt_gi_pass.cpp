#include "renderer/gi/rt/rt_gi_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include "scene/scene_gpu.h"
#include <array>

namespace somegi {

static constexpr uint32_t kMaxTextures = 128;

namespace {
struct RtPC {
    uint32_t outSizeX, outSizeY;
    float invOutSizeX, invOutSizeY;
};
static_assert(sizeof(RtPC) == 16, "RtPC must match shader push constant layout");
}

void RtGiPass::init(Device& d) {
    m_device = &d;

    // Sampler
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_linearClamp));

    // Descriptor set layout:
    // 0: UBO (FrameUniforms)
    // 1: gNormalRough (sampled image)
    // 2: gDepth (sampled image)
    // 3: gTLAS (acceleration structure)
    // 4: gInstances (storage buffer)
    // 5: gVertices (storage buffer)
    // 6: gIndices (storage buffer)
    // 7: gMaterials (storage buffer)
    // 8: sampler
    // 9: gTextures[] (sampled image array)
    // 10: gOutRt (storage image)
    std::array<VkDescriptorSetLayoutBinding, 11> b{};
    b[0]  = {0,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,                  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1]  = {1,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,                   1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2]  = {2,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,                   1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3]  = {3,  VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,      1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4]  = {4,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,                  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5]  = {5,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,                  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[6]  = {6,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,                  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[7]  = {7,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,                  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[8]  = {8,  VK_DESCRIPTOR_TYPE_SAMPLER,                         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[9]  = {9,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,   kMaxTextures, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[10] = {10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,                   1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    // Pool
    std::array<VkDescriptorPoolSize, 6> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,              1},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  kMaxTextures + 3},
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,   1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,               4},
        {VK_DESCRIPTOR_TYPE_SAMPLER,                      1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,                1},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_set));

    // Pipeline layout
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(RtPC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    // Compute shader
    ShaderModule cs(d, shaderDir() / "gi" / "rt" / "rt_gi.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));
}

void RtGiPass::destroy() {
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

void RtGiPass::bindFrame(Device& d, const RenderTargets& rt, VkBuffer frameUbo,
                          const SceneRtAS& rtAS, const SceneGpu& sceneGpu) {
    auto dev = d.device();

    VkDescriptorBufferInfo uboInfo{frameUbo, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo instBuf{rtAS.instanceDataBuffer(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo vertBuf{sceneGpu.vertexBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo idxBuf{sceneGpu.indexBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo matBuf{sceneGpu.materialBuffer.handle(), 0, VK_WHOLE_SIZE};

    auto sampledRO = [](VkImageView v) {
        VkDescriptorImageInfo i{};
        i.imageView = v; i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return i;
    };
    VkDescriptorImageInfo nr   = sampledRO(rt.gNormalRough.view());
    VkDescriptorImageInfo dp   = sampledRO(rt.depth.view());
    VkDescriptorImageInfo smp{}; smp.sampler = m_linearClamp;

    VkDescriptorImageInfo outRt{};
    outRt.imageView = rt.rtGI.view();
    outRt.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // TLAS acceleration structure descriptor info
    VkWriteDescriptorSetAccelerationStructureKHR tlasAI = rtAS.tlasWriteInfo();

    // Texture array: fill with scene images, fallback to whiteTex.
    std::vector<VkDescriptorImageInfo> texImgs;
    texImgs.reserve(kMaxTextures);
    for (uint32_t i = 0; i < kMaxTextures; ++i) {
        VkDescriptorImageInfo ii{};
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        if (i < sceneGpu.images.size())
            ii.imageView = sceneGpu.images[i].view();
        else
            ii.imageView = sceneGpu.whiteTex.view();
        texImgs.push_back(ii);
    }

    std::array<VkWriteDescriptorSet, 11> w{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_set; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &uboInfo;

    auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorImageInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_set; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = t; W.pImageInfo = p;
    };
    auto setBuf = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorBufferInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_set; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = t; W.pBufferInfo = p;
    };

    setImg(w[1],  1,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,              &nr);
    setImg(w[2],  2,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,              &dp);
    setImg(w[8],  8,  VK_DESCRIPTOR_TYPE_SAMPLER,                    &smp);
    setImg(w[10], 10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              &outRt);
    setBuf(w[4],  4,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             &instBuf);
    setBuf(w[5],  5,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             &vertBuf);
    setBuf(w[6],  6,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             &idxBuf);
    setBuf(w[7],  7,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             &matBuf);

    // Binding 3: TLAS (has pNext for acceleration structure).
    w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[3].dstSet = m_set; w[3].dstBinding = 3; w[3].descriptorCount = 1;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    w[3].pNext = &tlasAI;

    // Binding 9: texture array.
    w[9] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[9].dstSet = m_set; w[9].dstBinding = 9; w[9].descriptorCount = kMaxTextures;
    w[9].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[9].pImageInfo = texImgs.data();

    vkUpdateDescriptorSets(dev, (uint32_t)w.size(), w.data(), 0, nullptr);
}

void RtGiPass::record(VkCommandBuffer cmd, const RenderTargets& rt) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

    RtPC pc;
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
