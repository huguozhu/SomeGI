#include "vxgi_voxelize_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>
#include <cstring>

namespace somegi {

namespace {
struct VoxelizePC {
    float    model[16];
    uint32_t firstIndex;
    uint32_t indexCount;
    int32_t  vertexOffset;
    int32_t  materialIndex;
    float    gridMinX, gridMinY, gridMinZ;
    float    cellSize;
    uint32_t gridResolution;
    uint32_t _pad0, _pad1, _pad2;
};
static_assert(sizeof(VoxelizePC) == 112, "VoxelizePC must match shader push constant layout");
}

void VxgiVoxelizePass::init(Device& d, uint32_t maxTextures) {
    m_device = &d;
    m_maxTextures = maxTextures;

    // set=0：0=verts SSBO, 1=indices SSBO, 2=materials SSBO, 3=sampler,
    // 4=texture array, 5=voxelGrid storage image (mip 0).
    std::array<VkDescriptorSetLayoutBinding, 6> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLER,        1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  maxTextures,  VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1,            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    std::array<VkDescriptorBindingFlags, 6> bf{0u, 0u, 0u, 0u,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, 0u};
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bfci.bindingCount = (uint32_t)bf.size(); bfci.pBindingFlags = bf.data();

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.pNext = &bfci;
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    std::array<VkDescriptorPoolSize, 4> ps{{
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_SAMPLER,        1},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  maxTextures},
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
    pc.size = sizeof(VoxelizePC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "gi" / "vxgi" / "vxgi_voxelize.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));
}

void VxgiVoxelizePass::destroy() {
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

void VxgiVoxelizePass::bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount,
                                 const VxgiResources& vxgi) {
    VkDescriptorBufferInfo vb{gpu.vertexBuffer.handle(),   0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo ib{gpu.indexBuffer.handle(),    0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo mb{gpu.materialBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo  smp{}; smp.sampler = gpu.linearSampler;
    VkDescriptorImageInfo  vox{};
    vox.imageView = vxgi.mipView(0);
    vox.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::vector<VkDescriptorImageInfo> imgs;
    imgs.reserve(m_maxTextures);
    for (uint32_t i = 0; i < m_maxTextures; ++i) {
        VkDescriptorImageInfo ii{};
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii.imageView = (i < textureCount && i < gpu.images.size())
                       ? gpu.images[i].view() : gpu.whiteTex.view();
        imgs.push_back(ii);
    }

    std::array<VkWriteDescriptorSet, 6> w{};
    auto setBuf = [&](VkWriteDescriptorSet& W, uint32_t bi, const VkDescriptorBufferInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_set; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; W.pBufferInfo = p;
    };
    auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorImageInfo* p, uint32_t cnt = 1) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_set; W.dstBinding = bi; W.descriptorCount = cnt;
        W.descriptorType = t; W.pImageInfo = p;
    };
    setBuf(w[0], 0, &vb);
    setBuf(w[1], 1, &ib);
    setBuf(w[2], 2, &mb);
    setImg(w[3], 3, VK_DESCRIPTOR_TYPE_SAMPLER,       &smp);
    setImg(w[4], 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, imgs.data(), m_maxTextures);
    setImg(w[5], 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &vox);
    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void VxgiVoxelizePass::record(VkCommandBuffer cmd, const SceneCpu& cpu, const SceneGpu& gpu,
                              const glm::vec3& gridMin, float cellSize, uint32_t gridResolution) {
    (void)gpu;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

    VoxelizePC pc{};
    pc.gridMinX = gridMin.x; pc.gridMinY = gridMin.y; pc.gridMinZ = gridMin.z;
    pc.cellSize = cellSize;
    pc.gridResolution = gridResolution;

    for (auto& n : cpu.nodes) {
        if (n.meshIndex < 0) continue;
        const Mesh& M = cpu.meshes[n.meshIndex];
        // 把 model 矩阵摊平进 PC（glm::mat4 是 column-major 16 floats）。
        std::memcpy(pc.model, &n.worldTransform[0][0], sizeof(pc.model));
        for (auto& p : M.primitives) {
            pc.firstIndex    = p.firstIndex;
            pc.indexCount    = p.indexCount;
            pc.vertexOffset  = p.vertexOffset;
            pc.materialIndex = p.materialIndex;
            vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc), &pc);
            uint32_t triCount = p.indexCount / 3u;
            uint32_t gx = (triCount + 63u) / 64u;
            if (gx > 0) vkCmdDispatch(cmd, gx, 1, 1);
        }
    }
}

}
