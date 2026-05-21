#include "ddgi_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>

namespace somegi {

namespace {
struct UpdatePC {
    float    ddgiOriginX, ddgiOriginY, ddgiOriginZ;
    float    _pad0;
    float    ddgiSpacingX, ddgiSpacingY, ddgiSpacingZ;
    float    _pad1;
    uint32_t probesX, probesY, probesZ;
    uint32_t raysPerProbe;
    float    randomRotation;
    float    voxelGridDim;
    float    vxgiCellSize;
    uint32_t vxgiResolution;
    float    vxgiGridMinX, vxgiGridMinY, vxgiGridMinZ;
    float    _pad2;
};
static_assert(sizeof(UpdatePC) == 80, "UpdatePC must match shader push constant layout");

struct BlendPC {
    uint32_t probesX, probesY, probesZ;
    uint32_t raysPerProbe;
    uint32_t octaIrr;
    uint32_t octaDist;
    float    hysteresis;
    float    maxRayDistance;
};
static_assert(sizeof(BlendPC) == 32, "BlendPC must match shader push constant layout");
struct ClassifyPC {
    uint32_t probeCount;
    uint32_t raysPerProbe;
    float    closeHitDist;
    float    closeHitFrac;
};
static_assert(sizeof(ClassifyPC) == 16, "ClassifyPC must match shader push constant layout");
}

void DdgiPass::init(Device& d) {
    m_device = &d;

    // 共享 sampler (linear clamp)
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_linearClamp));

    // === Update pipeline set layout: 0=voxelGrid sampled, 1=sampler, 2=rayBuf storage ===
    {
        std::array<VkDescriptorSetLayoutBinding, 3> b{};
        b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER,        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));
    }

    // === Blend pipeline set layout: 0=rayBuf, 1=irrAtlas storage, 2=distAtlas storage ===
    {
        std::array<VkDescriptorSetLayoutBinding, 3> b{};
        b[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayoutBlend));
    }

    // === Classify pipeline set layout: 0=rayBuf, 1=probeStates storage ===
    {
        std::array<VkDescriptorSetLayoutBinding, 2> b{};
        b[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayoutClassify));
    }

    // pool 分三组 set
    std::array<VkDescriptorPoolSize, 4> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1},
        {VK_DESCRIPTOR_TYPE_SAMPLER,        1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},   // update(1) + blend(1) + classify(2)
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 3; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    {
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_setUpdate));
        dai.pSetLayouts = &m_setLayoutBlend;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_setBlend));
        dai.pSetLayouts = &m_setLayoutClassify;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_setClassify));
    }

    // Update pipeline layout
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = sizeof(UpdatePC);
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));
    }
    // Blend pipeline layout
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = sizeof(BlendPC);
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayoutBlend;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayoutBlend));
    }
    // Classify pipeline layout
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = sizeof(ClassifyPC);
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayoutClassify;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayoutClassify));
    }

    // Compute pipelines
    auto makeCompute = [&](const std::filesystem::path& spv, const char* entry,
                           VkPipelineLayout layout) {
        ShaderModule cs(d, spv);
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = entry;
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = stage; cpci.layout = layout;
        VkPipeline p = VK_NULL_HANDLE;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &p));
        return p;
    };
    m_pipelineUpdate    = makeCompute(shaderDir() / "gi" / "ddgi" / "ddgi_update.spv",
                                       "cs_main", m_pipelineLayout);
    m_pipelineClassify  = makeCompute(shaderDir() / "gi" / "ddgi" / "ddgi_classify.spv",
                                       "cs_main", m_pipelineLayoutClassify);
    m_pipelineBlendIrr  = makeCompute(shaderDir() / "gi" / "ddgi" / "ddgi_blend.spv",
                                       "cs_irradiance", m_pipelineLayoutBlend);
    m_pipelineBlendDist = makeCompute(shaderDir() / "gi" / "ddgi" / "ddgi_blend.spv",
                                       "cs_distance", m_pipelineLayoutBlend);
}

void DdgiPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pipelineUpdate)    vkDestroyPipeline(dev, m_pipelineUpdate, nullptr);
    if (m_pipelineClassify)  vkDestroyPipeline(dev, m_pipelineClassify, nullptr);
    if (m_pipelineBlendIrr)  vkDestroyPipeline(dev, m_pipelineBlendIrr, nullptr);
    if (m_pipelineBlendDist) vkDestroyPipeline(dev, m_pipelineBlendDist, nullptr);
    if (m_pipelineLayout)         vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_pipelineLayoutBlend)    vkDestroyPipelineLayout(dev, m_pipelineLayoutBlend, nullptr);
    if (m_pipelineLayoutClassify) vkDestroyPipelineLayout(dev, m_pipelineLayoutClassify, nullptr);
    if (m_pool)               vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout)          vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    if (m_setLayoutBlend)     vkDestroyDescriptorSetLayout(dev, m_setLayoutBlend, nullptr);
    if (m_setLayoutClassify)  vkDestroyDescriptorSetLayout(dev, m_setLayoutClassify, nullptr);
    if (m_linearClamp)      vkDestroySampler(dev, m_linearClamp, nullptr);
    *this = {};
}

void DdgiPass::bindResources(Device& d, const DdgiResources& ddgi, const VxgiResources& vxgi) {
    // Update set
    VkDescriptorImageInfo voxi{};
    voxi.imageView = vxgi.fullView();
    voxi.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo smp{}; smp.sampler = m_linearClamp;
    VkDescriptorBufferInfo rb{ddgi.rayBuffer().handle(), 0, VK_WHOLE_SIZE};

    std::array<VkWriteDescriptorSet, 6> w{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_setUpdate; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &voxi;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[1].dstSet = m_setUpdate; w[1].dstBinding = 1; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[1].pImageInfo = &smp;
    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[2].dstSet = m_setUpdate; w[2].dstBinding = 2; w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[2].pBufferInfo = &rb;

    // Blend set
    VkDescriptorImageInfo iat{};
    iat.imageView = ddgi.irradiance().view();
    iat.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo dat{};
    dat.imageView = ddgi.distance().view();
    dat.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[3].dstSet = m_setBlend; w[3].dstBinding = 0; w[3].descriptorCount = 1;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[3].pBufferInfo = &rb;
    w[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[4].dstSet = m_setBlend; w[4].dstBinding = 1; w[4].descriptorCount = 1;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[4].pImageInfo = &iat;
    w[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[5].dstSet = m_setBlend; w[5].dstBinding = 2; w[5].descriptorCount = 1;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[5].pImageInfo = &dat;

    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);

    // B.5 classify set: rayBuf + probeStates
    VkDescriptorBufferInfo psbuf{ddgi.probeStates().handle(), 0, VK_WHOLE_SIZE};
    std::array<VkWriteDescriptorSet, 2> wc{};
    wc[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wc[0].dstSet = m_setClassify; wc[0].dstBinding = 0; wc[0].descriptorCount = 1;
    wc[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wc[0].pBufferInfo = &rb;
    wc[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wc[1].dstSet = m_setClassify; wc[1].dstBinding = 1; wc[1].descriptorCount = 1;
    wc[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wc[1].pBufferInfo = &psbuf;
    vkUpdateDescriptorSets(d.device(), (uint32_t)wc.size(), wc.data(), 0, nullptr);
}

void DdgiPass::record(VkCommandBuffer cmd, const DdgiResources& ddgi,
                      const glm::vec3& ddgiOrigin, const glm::vec3& ddgiSpacing,
                      const glm::vec3& vxgiGridMin, float vxgiCellSize, uint32_t vxgiResolution,
                      float randomRotation, uint32_t /*frameIndex*/) {
    // 1. update pass：dispatch (probeCount * raysPerProbe / 64)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineUpdate);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_setUpdate, 0, nullptr);

    UpdatePC upc{};
    upc.ddgiOriginX = ddgiOrigin.x; upc.ddgiOriginY = ddgiOrigin.y; upc.ddgiOriginZ = ddgiOrigin.z;
    upc.ddgiSpacingX = ddgiSpacing.x; upc.ddgiSpacingY = ddgiSpacing.y; upc.ddgiSpacingZ = ddgiSpacing.z;
    upc.probesX = DdgiResources::kProbesX;
    upc.probesY = DdgiResources::kProbesY;
    upc.probesZ = DdgiResources::kProbesZ;
    upc.raysPerProbe = DdgiResources::kRaysPerProbe;
    upc.randomRotation = randomRotation;
    upc.vxgiCellSize = vxgiCellSize;
    upc.vxgiResolution = vxgiResolution;
    upc.vxgiGridMinX = vxgiGridMin.x; upc.vxgiGridMinY = vxgiGridMin.y; upc.vxgiGridMinZ = vxgiGridMin.z;
    upc.voxelGridDim = vxgiCellSize * float(vxgiResolution);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(upc), &upc);

    uint32_t totalRays = DdgiResources::kProbeCount * DdgiResources::kRaysPerProbe;
    uint32_t groups = (totalRays + 63) / 64;
    vkCmdDispatch(cmd, groups, 1, 1);

    // 2. ray buffer write → 后续 read barrier（classify + blend 都消费）
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);

    // 3. classify：每 probe 看自己 64 rays 的近距离命中比例
    {
        ClassifyPC cpc{};
        cpc.probeCount = DdgiResources::kProbeCount;
        cpc.raysPerProbe = DdgiResources::kRaysPerProbe;
        cpc.closeHitDist = vxgiCellSize * 0.5f;   // < 半 voxel cell 视为近距离
        cpc.closeHitFrac = 0.7f;                   // > 70% 近距离 → inactive

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineClassify);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayoutClassify, 0, 1, &m_setClassify, 0, nullptr);
        vkCmdPushConstants(cmd, m_pipelineLayoutClassify, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(cpc), &cpc);
        uint32_t cgroups = (DdgiResources::kProbeCount + 63) / 64;
        vkCmdDispatch(cmd, cgroups, 1, 1);

        // probeStates write → lighting read barrier（lighting 在 frame 后期读）
        vkCmdPipelineBarrier2(cmd, &di);
    }

    // 4. blend irradiance + distance（共享 set 不切，pipeline 各自 bind）
    BlendPC bpc{};
    bpc.probesX = DdgiResources::kProbesX;
    bpc.probesY = DdgiResources::kProbesY;
    bpc.probesZ = DdgiResources::kProbesZ;
    bpc.raysPerProbe = DdgiResources::kRaysPerProbe;
    bpc.octaIrr = DdgiResources::kOctaIrr;
    bpc.octaDist = DdgiResources::kOctaDist;
    bpc.hysteresis = 0.92f;   // 论文 0.97；此处稍激进让动态变化更跟得上
    bpc.maxRayDistance = vxgiCellSize * float(vxgiResolution);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayoutBlend, 0, 1, &m_setBlend, 0, nullptr);
    vkCmdPushConstants(cmd, m_pipelineLayoutBlend, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(bpc), &bpc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineBlendIrr);
    {
        uint32_t aw = DdgiResources::irradianceAtlasW();
        uint32_t ah = DdgiResources::irradianceAtlasH();
        vkCmdDispatch(cmd, (aw + 7) / 8, (ah + 7) / 8, 1);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineBlendDist);
    {
        uint32_t aw = DdgiResources::distanceAtlasW();
        uint32_t ah = DdgiResources::distanceAtlasH();
        vkCmdDispatch(cmd, (aw + 7) / 8, (ah + 7) / 8, 1);
    }
}

}
