#include "lpv_inject_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>

namespace somegi {

namespace {
struct InjectPC {
    uint32_t rsmSizeX, rsmSizeY;
    uint32_t gridResolution;
    uint32_t _pad0;
    float    gridMinX, gridMinY, gridMinZ;
    float    cellSize;
};
}

void LpvInjectPass::init(Device& d, uint32_t rsmSize) {
    m_device = &d;
    m_rsmSize = rsmSize;

    // set=0 layout：3 sampled (RSM) + 3 storage (LPV)。
    // 0..2: RSM sampled, 3..5: LPV R/G/B storage, 6: GV storage (B.8)
    std::array<VkDescriptorSetLayoutBinding, 7> b{};
    for (uint32_t i = 0; i < 7; ++i) {
        b[i].binding = i;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].descriptorType = (i < 3) ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                      : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    std::array<VkDescriptorPoolSize, 2> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4},   // +1 for GV
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_set));

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(InjectPC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "gi" / "lpv" / "lpv_inject.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));
}

void LpvInjectPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pipeline)       vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_pool)           vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout)      vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    m_pipeline = VK_NULL_HANDLE; m_pipelineLayout = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE; m_setLayout = VK_NULL_HANDLE;
    m_device = nullptr;
    m_lpvR = m_lpvG = m_lpvB = m_gv = VK_NULL_HANDLE;
}

void LpvInjectPass::bindResources(Device& d,
                                  const Image& rsmPos, const Image& rsmN, const Image& rsmFlux,
                                  const LpvGrid& grid, const Image& gv) {
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
    VkDescriptorImageInfo rp = sampledRO(rsmPos.view());
    VkDescriptorImageInfo rn = sampledRO(rsmN.view());
    VkDescriptorImageInfo rf = sampledRO(rsmFlux.view());
    VkDescriptorImageInfo lr = storageGen(grid.lpvR.view());
    VkDescriptorImageInfo lg = storageGen(grid.lpvG.view());
    VkDescriptorImageInfo lb = storageGen(grid.lpvB.view());
    VkDescriptorImageInfo gvI = storageGen(gv.view());

    std::array<VkWriteDescriptorSet, 7> w{};
    auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorImageInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_set; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = t; W.pImageInfo = p;
    };
    setImg(w[0], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &rp);
    setImg(w[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &rn);
    setImg(w[2], 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &rf);
    setImg(w[3], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &lr);
    setImg(w[4], 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &lg);
    setImg(w[5], 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &lb);
    setImg(w[6], 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gvI);
    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);

    m_lpvR = grid.lpvR.image();
    m_lpvG = grid.lpvG.image();
    m_lpvB = grid.lpvB.image();
    m_gv   = gv.image();
}

void LpvInjectPass::record(VkCommandBuffer cmd, uint32_t gridResolution,
                           const glm::vec3& gridMin, float cellSize) {
    // 1. clear 三张 grid image 到 0。它们必须先转 TRANSFER_DST_OPTIMAL，
    //    clear 后再转回 GENERAL 给 dispatch 写。这里 oldLayout 用 UNDEFINED
    //    —— 上一帧结束 grid 是 SHADER_READ_ONLY（lighting 读完）或第一帧
    //    UNDEFINED；都允许 discard。
    auto barr = [&](VkImage img, VkImageLayout oldL, VkImageLayout newL,
                    VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                    VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
        b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
        b.oldLayout = oldL; b.newLayout = newL;
        b.image = img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    };
    auto clear = [&](VkImage img) {
        VkClearColorValue zero{};
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &zero, 1, &r);
    };

    VkImage imgs[4] = {m_lpvR, m_lpvG, m_lpvB, m_gv};
    for (auto img : imgs) {
        barr(img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
             VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        clear(img);
        barr(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
             VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    // 2. dispatch inject。
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

    InjectPC pc{};
    pc.rsmSizeX = m_rsmSize; pc.rsmSizeY = m_rsmSize;
    pc.gridResolution = gridResolution;
    pc.gridMinX = gridMin.x; pc.gridMinY = gridMin.y; pc.gridMinZ = gridMin.z;
    pc.cellSize = cellSize;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t gx = (m_rsmSize + 7) / 8;
    uint32_t gy = (m_rsmSize + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);
}

}
