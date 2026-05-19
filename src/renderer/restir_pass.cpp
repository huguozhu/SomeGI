// RestirPass —— C.4 ReSTIR DI 实现。

#include "restir_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>

namespace somegi {

namespace {
struct InitPC {
    uint32_t outX, outY;
    float    invX, invY;
    uint32_t numLights;
    uint32_t numCandidates;
    uint32_t frameIndex;
    uint32_t pad0;
};
struct SpatialPC {
    uint32_t outX, outY;
    float    invX, invY;
    uint32_t numLights;
    uint32_t numNeighbors;
    float    radiusPixels;
    uint32_t frameIndex;
};
struct ShadePC {
    uint32_t outX, outY;
    float    invX, invY;
    uint32_t numLights;
    float    shadowSteps;
    float    intensityScale;
    uint32_t pad0;
};

VkImageMemoryBarrier2 makeBarrier2D(VkImage img,
    VkImageLayout oldL, VkImageLayout newL,
    VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
    VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
    b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
    b.oldLayout = oldL; b.newLayout = newL;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return b;
}
void issueBarrier(VkCommandBuffer cmd, VkImageMemoryBarrier2& b) {
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}
} // anon

void RestirPass::init(Device& d) {
    m_device = &d;

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_linearClamp));

    // 共享 pool：3 个 set
    std::array<VkDescriptorPoolSize, 5> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},   // 每 set 一个 light SSBO
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  16},  // 多 GBuffer + 3D + reservoir reads
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  4},   // reservoir writes + restir out
        {VK_DESCRIPTOR_TYPE_SAMPLER,        1},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 3; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    initInitPipeline();
    initSpatialPipeline();
    initShadePipeline();
}

void RestirPass::initInitPipeline() {
    // bindings: 0=FrameUBO, 1=albedoMetal, 2=normalRough, 3=depth,
    //           4=lights SSBO, 5=reservoirOutA storage
    std::array<VkDescriptorSetLayoutBinding, 6> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_device->device(), &li, nullptr, &m_initSetLayout));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_initSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(m_device->device(), &dai, &m_initSet));

    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(InitPC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_initSetLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(m_device->device(), &plci, nullptr, &m_initPlLayout));

    ShaderModule cs(*m_device, shaderDir() / "gi" / "restir" / "restir_init.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_initPlLayout;
    VK_CHECK(vkCreateComputePipelines(m_device->device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_initPipeline));
}

void RestirPass::initSpatialPipeline() {
    // bindings: 0=Frame, 1=albedo, 2=normal, 3=depth, 4=lights SSBO,
    //           5=reservoirInA sampled, 6=reservoirOutB storage
    std::array<VkDescriptorSetLayoutBinding, 7> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[6] = {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_device->device(), &li, nullptr, &m_spatialSetLayout));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_spatialSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(m_device->device(), &dai, &m_spatialSet));

    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SpatialPC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_spatialSetLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(m_device->device(), &plci, nullptr, &m_spatialPlLayout));

    ShaderModule cs(*m_device, shaderDir() / "gi" / "restir" / "restir_spatial.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_spatialPlLayout;
    VK_CHECK(vkCreateComputePipelines(m_device->device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_spatialPipeline));
}

void RestirPass::initShadePipeline() {
    // bindings: 0=Frame, 1=albedo, 2=normal, 3=depth, 4=lights SSBO,
    //           5=reservoir sampled (B), 6=voxelGrid sampled,
    //           7=sampler, 8=outRestir storage
    std::array<VkDescriptorSetLayoutBinding, 9> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[6] = {6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[7] = {7, VK_DESCRIPTOR_TYPE_SAMPLER,        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[8] = {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_device->device(), &li, nullptr, &m_shadeSetLayout));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_shadeSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(m_device->device(), &dai, &m_shadeSet));

    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadePC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_shadeSetLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(m_device->device(), &plci, nullptr, &m_shadePlLayout));

    ShaderModule cs(*m_device, shaderDir() / "gi" / "restir" / "restir_shade.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_shadePlLayout;
    VK_CHECK(vkCreateComputePipelines(m_device->device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_shadePipeline));
}

void RestirPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    auto destroyTriple = [&](VkPipeline& p, VkPipelineLayout& pl, VkDescriptorSetLayout& sl) {
        if (p)  { vkDestroyPipeline(dev, p, nullptr); p = VK_NULL_HANDLE; }
        if (pl) { vkDestroyPipelineLayout(dev, pl, nullptr); pl = VK_NULL_HANDLE; }
        if (sl) { vkDestroyDescriptorSetLayout(dev, sl, nullptr); sl = VK_NULL_HANDLE; }
    };
    destroyTriple(m_initPipeline,    m_initPlLayout,    m_initSetLayout);
    destroyTriple(m_spatialPipeline, m_spatialPlLayout, m_spatialSetLayout);
    destroyTriple(m_shadePipeline,   m_shadePlLayout,   m_shadeSetLayout);
    if (m_pool)        { vkDestroyDescriptorPool(dev, m_pool, nullptr); m_pool = VK_NULL_HANDLE; }
    if (m_linearClamp) { vkDestroySampler(dev, m_linearClamp, nullptr); m_linearClamp = VK_NULL_HANDLE; }
    m_initSet = m_spatialSet = m_shadeSet = VK_NULL_HANDLE;
    m_device = nullptr;
}

void RestirPass::bindResources(Device& d,
                               const RestirResources& res,
                               const VxgiResources& vxgi,
                               const RenderTargets& rt,
                               VkBuffer frameUbo) {
    VkDevice dev = d.device();
    auto sampledRO = [](VkImageView v) {
        VkDescriptorImageInfo i{};
        i.imageView = v;
        i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return i;
    };
    auto storage = [](VkImageView v) {
        VkDescriptorImageInfo i{};
        i.imageView = v;
        i.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        return i;
    };

    VkDescriptorBufferInfo uboInfo{frameUbo, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo lightInfo{res.lightBuffer(), 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo am  = sampledRO(rt.gAlbedoMetal.view());
    VkDescriptorImageInfo nr  = sampledRO(rt.gNormalRough.view());
    VkDescriptorImageInfo dp  = sampledRO(rt.depth.view());
    VkDescriptorImageInfo rA  = sampledRO(res.reservoirA().view());   // spatial 当 src
    VkDescriptorImageInfo rB  = sampledRO(res.reservoirB().view());   // shade 当 src
    VkDescriptorImageInfo wA  = storage(res.reservoirA().view());     // init 写
    VkDescriptorImageInfo wB  = storage(res.reservoirB().view());     // spatial 写
    VkDescriptorImageInfo vox = sampledRO(vxgi.fullView());
    VkDescriptorImageInfo smp{}; smp.sampler = m_linearClamp;
    VkDescriptorImageInfo outR = storage(rt.restir.view());

    auto setBuf = [&](VkWriteDescriptorSet& W, VkDescriptorSet ds, uint32_t bi,
                      VkDescriptorType t, const VkDescriptorBufferInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = ds; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = t; W.pBufferInfo = p;
    };
    auto setImg = [&](VkWriteDescriptorSet& W, VkDescriptorSet ds, uint32_t bi,
                      VkDescriptorType t, const VkDescriptorImageInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = ds; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = t; W.pImageInfo = p;
    };

    // init set
    {
        std::array<VkWriteDescriptorSet, 6> w{};
        setBuf(w[0], m_initSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &uboInfo);
        setImg(w[1], m_initSet, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &am);
        setImg(w[2], m_initSet, 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &nr);
        setImg(w[3], m_initSet, 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &dp);
        setBuf(w[4], m_initSet, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &lightInfo);
        setImg(w[5], m_initSet, 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  &wA);
        vkUpdateDescriptorSets(dev, (uint32_t)w.size(), w.data(), 0, nullptr);
    }
    // spatial set
    {
        std::array<VkWriteDescriptorSet, 7> w{};
        setBuf(w[0], m_spatialSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &uboInfo);
        setImg(w[1], m_spatialSet, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &am);
        setImg(w[2], m_spatialSet, 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &nr);
        setImg(w[3], m_spatialSet, 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &dp);
        setBuf(w[4], m_spatialSet, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &lightInfo);
        setImg(w[5], m_spatialSet, 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &rA);
        setImg(w[6], m_spatialSet, 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  &wB);
        vkUpdateDescriptorSets(dev, (uint32_t)w.size(), w.data(), 0, nullptr);
    }
    // shade set
    {
        std::array<VkWriteDescriptorSet, 9> w{};
        setBuf(w[0], m_shadeSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &uboInfo);
        setImg(w[1], m_shadeSet, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &am);
        setImg(w[2], m_shadeSet, 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &nr);
        setImg(w[3], m_shadeSet, 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &dp);
        setBuf(w[4], m_shadeSet, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &lightInfo);
        setImg(w[5], m_shadeSet, 5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &rB);
        setImg(w[6], m_shadeSet, 6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  &vox);
        setImg(w[7], m_shadeSet, 7, VK_DESCRIPTOR_TYPE_SAMPLER,        &smp);
        setImg(w[8], m_shadeSet, 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  &outR);
        vkUpdateDescriptorSets(dev, (uint32_t)w.size(), w.data(), 0, nullptr);
    }
}

void RestirPass::record(VkCommandBuffer cmd,
                        const RestirResources& res,
                        const RenderTargets& rt,
                        uint32_t numLightsArg,
                        uint32_t numCandidatesArg,
                        uint32_t numNeighborsArg,
                        float    spatialRadiusPx,
                        uint32_t shadowStepsArg,
                        float    intensityScaleArg,
                        uint32_t frameIndex) {
    uint32_t W = rt.extent.width, H = rt.extent.height;
    uint32_t gx = (W + 7) / 8, gy = (H + 7) / 8;

    // ----- 1. init: 写 reservoirA -----
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_initPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_initPlLayout, 0, 1, &m_initSet, 0, nullptr);
    InitPC ip{};
    ip.outX = W; ip.outY = H;
    ip.invX = 1.0f / (float)W; ip.invY = 1.0f / (float)H;
    ip.numLights = numLightsArg;
    ip.numCandidates = numCandidatesArg;
    ip.frameIndex = frameIndex;
    vkCmdPushConstants(cmd, m_initPlLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(ip), &ip);
    vkCmdDispatch(cmd, gx, gy, 1);

    // reservoirA: GENERAL → SHADER_READ_ONLY（spatial 当 src 采样）
    {
        auto b = makeBarrier2D(res.reservoirA().image(),
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        issueBarrier(cmd, b);
    }

    // ----- 2. spatial: 写 reservoirB -----
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_spatialPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_spatialPlLayout, 0, 1, &m_spatialSet, 0, nullptr);
    SpatialPC sp{};
    sp.outX = W; sp.outY = H;
    sp.invX = 1.0f / (float)W; sp.invY = 1.0f / (float)H;
    sp.numLights = numLightsArg;
    sp.numNeighbors = numNeighborsArg;
    sp.radiusPixels = spatialRadiusPx;
    sp.frameIndex = frameIndex;
    vkCmdPushConstants(cmd, m_spatialPlLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(sp), &sp);
    vkCmdDispatch(cmd, gx, gy, 1);

    // reservoirB: GENERAL → SHADER_READ_ONLY（shade 当 src 采样）
    {
        auto b = makeBarrier2D(res.reservoirB().image(),
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        issueBarrier(cmd, b);
    }

    // ----- 3. shade: 读 reservoirB → 写 rt.restir -----
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_shadePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_shadePlLayout, 0, 1, &m_shadeSet, 0, nullptr);
    ShadePC shp{};
    shp.outX = W; shp.outY = H;
    shp.invX = 1.0f / (float)W; shp.invY = 1.0f / (float)H;
    shp.numLights = numLightsArg;
    shp.shadowSteps = (float)shadowStepsArg;
    shp.intensityScale = intensityScaleArg;
    vkCmdPushConstants(cmd, m_shadePlLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(shp), &shp);
    vkCmdDispatch(cmd, gx, gy, 1);

    // 收尾：reservoirA/B 转回 GENERAL 给下一帧用
    {
        auto bA = makeBarrier2D(res.reservoirA().image(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        issueBarrier(cmd, bA);
        auto bB = makeBarrier2D(res.reservoirB().image(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        issueBarrier(cmd, bB);
    }
}

}
