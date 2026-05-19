// SdfgiPass —— C.3 SDFGI-lite 实现。
//
// 内部 4 个子 pipeline：seed / jfa / finalize / trace。每一步独立 set
// layout & pool。seedA / seedB ping-pong；JFA 步数由 resolution 决定
//（log2(R) 步），128 → 7 步。
//
// 为减少代码量，这里把 4 个 pipeline 的 set / pool / descriptor 都集中在
// 同一个 pool 里 alloc；JFA 和 finalize 的两组 set 在 init 时一次 alloc。

#include "sdfgi_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>

namespace somegi {

// ===== Push constant 类型（与 .slang 端逐字节对应）==========================
namespace {
struct SeedPC {
    uint32_t resolution;
    float    threshold;
    float    pad0, pad1;
};
struct JfaPC {
    uint32_t resolution;
    int32_t  k;
    uint32_t pad0, pad1;
};
struct FinalizePC {
    uint32_t resolution;
    float    maxDist;
    float    pad0, pad1;
};
struct TracePC {
    uint32_t outSizeX, outSizeY;
    float    invOutSizeX, invOutSizeY;
    uint32_t numRays;
    uint32_t maxSteps;
    float    rayMaxCells;
    float    hitEpsCells;
    uint32_t frameIndex;
    uint32_t pad0, pad1, pad2;
};

VkImageMemoryBarrier2 makeBarrier3D(VkImage img,
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

// ===== init =================================================================

void SdfgiPass::init(Device& d) {
    m_device = &d;

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_linearClamp));

    // 共享一个大 descriptor pool。set 数：seed=1, jfa=2, finalize=2, trace=1 = 6
    // 描述符数：保守上界
    std::array<VkDescriptorPoolSize, 4> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  16},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  8},
        {VK_DESCRIPTOR_TYPE_SAMPLER,        1},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 6; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    initSeedPipeline();
    initJfaPipeline();
    initFinalizePipeline();
    initTracePipeline();
}

void SdfgiPass::initSeedPipeline() {
    // bindings: 0=voxelGrid sampled, 1=seedOut storage
    std::array<VkDescriptorSetLayoutBinding, 2> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_device->device(), &li, nullptr, &m_seedSetLayout));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_seedSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(m_device->device(), &dai, &m_seedSet));

    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SeedPC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_seedSetLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(m_device->device(), &plci, nullptr, &m_seedPlLayout));

    ShaderModule cs(*m_device, shaderDir() / "gi" / "sdfgi" / "sdfgi_seed.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_seedPlLayout;
    VK_CHECK(vkCreateComputePipelines(m_device->device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_seedPipeline));
}

void SdfgiPass::initJfaPipeline() {
    // bindings: 0=seedSrc sampled (作 storage 读简化为 sampled), 1=seedDst storage
    std::array<VkDescriptorSetLayoutBinding, 2> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_device->device(), &li, nullptr, &m_jfaSetLayout));

    std::array<VkDescriptorSetLayout, 2> layouts{m_jfaSetLayout, m_jfaSetLayout};
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 2; dai.pSetLayouts = layouts.data();
    std::array<VkDescriptorSet, 2> sets{};
    VK_CHECK(vkAllocateDescriptorSets(m_device->device(), &dai, sets.data()));
    m_jfaSetAB = sets[0]; m_jfaSetBA = sets[1];

    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(JfaPC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_jfaSetLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(m_device->device(), &plci, nullptr, &m_jfaPlLayout));

    ShaderModule cs(*m_device, shaderDir() / "gi" / "sdfgi" / "sdfgi_jfa.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_jfaPlLayout;
    VK_CHECK(vkCreateComputePipelines(m_device->device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_jfaPipeline));
}

void SdfgiPass::initFinalizePipeline() {
    // bindings: 0=seed sampled, 1=udf storage
    std::array<VkDescriptorSetLayoutBinding, 2> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_device->device(), &li, nullptr, &m_finSetLayout));

    std::array<VkDescriptorSetLayout, 2> layouts{m_finSetLayout, m_finSetLayout};
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 2; dai.pSetLayouts = layouts.data();
    std::array<VkDescriptorSet, 2> sets{};
    VK_CHECK(vkAllocateDescriptorSets(m_device->device(), &dai, sets.data()));
    m_finSetA = sets[0]; m_finSetB = sets[1];

    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FinalizePC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_finSetLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(m_device->device(), &plci, nullptr, &m_finPlLayout));

    ShaderModule cs(*m_device, shaderDir() / "gi" / "sdfgi" / "sdfgi_finalize.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_finPlLayout;
    VK_CHECK(vkCreateComputePipelines(m_device->device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_finPipeline));
}

void SdfgiPass::initTracePipeline() {
    // bindings: 0=FrameUBO, 1=normal sampled, 2=depth sampled, 3=sdf sampled,
    //           4=voxelGrid sampled, 5=sampler, 6=outSdfgi storage
    std::array<VkDescriptorSetLayoutBinding, 7> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_SAMPLER,        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[6] = {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_device->device(), &li, nullptr, &m_traceSetLayout));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_traceSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(m_device->device(), &dai, &m_traceSet));

    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TracePC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_traceSetLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(m_device->device(), &plci, nullptr, &m_tracePlLayout));

    ShaderModule cs(*m_device, shaderDir() / "gi" / "sdfgi" / "sdfgi_trace.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage; cpci.layout = m_tracePlLayout;
    VK_CHECK(vkCreateComputePipelines(m_device->device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_tracePipeline));
}

void SdfgiPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    auto destroyPipeline = [&](VkPipeline& p, VkPipelineLayout& pl, VkDescriptorSetLayout& sl) {
        if (p) { vkDestroyPipeline(dev, p, nullptr); p = VK_NULL_HANDLE; }
        if (pl) { vkDestroyPipelineLayout(dev, pl, nullptr); pl = VK_NULL_HANDLE; }
        if (sl) { vkDestroyDescriptorSetLayout(dev, sl, nullptr); sl = VK_NULL_HANDLE; }
    };
    destroyPipeline(m_seedPipeline,  m_seedPlLayout,  m_seedSetLayout);
    destroyPipeline(m_jfaPipeline,   m_jfaPlLayout,   m_jfaSetLayout);
    destroyPipeline(m_finPipeline,   m_finPlLayout,   m_finSetLayout);
    destroyPipeline(m_tracePipeline, m_tracePlLayout, m_traceSetLayout);
    if (m_pool) { vkDestroyDescriptorPool(dev, m_pool, nullptr); m_pool = VK_NULL_HANDLE; }
    if (m_linearClamp) { vkDestroySampler(dev, m_linearClamp, nullptr); m_linearClamp = VK_NULL_HANDLE; }
    m_seedSet = m_jfaSetAB = m_jfaSetBA = VK_NULL_HANDLE;
    m_finSetA = m_finSetB = m_traceSet = VK_NULL_HANDLE;
    m_device = nullptr;
}

// ===== bindResources ========================================================
//
// 这里假设：
//   - voxelGrid: 全 mip SHADER_READ_ONLY（实际进 record 时也是该状态）
//   - seedA / seedB / udf: GENERAL（用于 storage 写）
// 我们一次性把所有 set 都更新好，layout 用 GENERAL（seed 描述符虽然是
// SAMPLED_IMAGE 但 GENERAL 也合法）。voxelGrid / depth / normal 这些
// SAMPLED_IMAGE 用 SHADER_READ_ONLY_OPTIMAL。

void SdfgiPass::bindResources(Device& d,
                              const SdfgiResources& sdfgi,
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

    // seed set：voxelGrid sampled (mip 0 view) → seedA storage
    {
        // voxelGrid full view 是整个 mipchain；shader 用 Load(int4(uvw,0)) 直接
        // 取 mip 0，这里用 fullView() 提供。
        VkDescriptorImageInfo vox = sampledRO(vxgi.fullView());
        VkDescriptorImageInfo dst = storage(sdfgi.seedA().view());
        std::array<VkWriteDescriptorSet, 2> w{};
        auto setImg = [&](VkWriteDescriptorSet& W, VkDescriptorSet ds, uint32_t bi,
                          VkDescriptorType t, const VkDescriptorImageInfo* p) {
            W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            W.dstSet = ds; W.dstBinding = bi; W.descriptorCount = 1;
            W.descriptorType = t; W.pImageInfo = p;
        };
        setImg(w[0], m_seedSet, 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &vox);
        setImg(w[1], m_seedSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dst);
        vkUpdateDescriptorSets(dev, (uint32_t)w.size(), w.data(), 0, nullptr);
    }

    // JFA set AB：src=A (sampled), dst=B (storage)
    // JFA set BA：src=B (sampled), dst=A (storage)
    auto bindJfa = [&](VkDescriptorSet ds, VkImageView srcView, VkImageView dstView) {
        VkDescriptorImageInfo src = sampledRO(srcView);
        VkDescriptorImageInfo dst = storage(dstView);
        std::array<VkWriteDescriptorSet, 2> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = ds; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &src;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = ds; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &dst;
        vkUpdateDescriptorSets(dev, (uint32_t)w.size(), w.data(), 0, nullptr);
    };
    bindJfa(m_jfaSetAB, sdfgi.seedA().view(), sdfgi.seedB().view());
    bindJfa(m_jfaSetBA, sdfgi.seedB().view(), sdfgi.seedA().view());

    // Finalize set：src 是 seedA 或 seedB（按 JFA 落点决定），dst=udf
    auto bindFin = [&](VkDescriptorSet ds, VkImageView srcView) {
        VkDescriptorImageInfo src = sampledRO(srcView);
        VkDescriptorImageInfo dst = storage(sdfgi.udf().view());
        std::array<VkWriteDescriptorSet, 2> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = ds; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &src;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = ds; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &dst;
        vkUpdateDescriptorSets(dev, (uint32_t)w.size(), w.data(), 0, nullptr);
    };
    bindFin(m_finSetA, sdfgi.seedA().view());
    bindFin(m_finSetB, sdfgi.seedB().view());

    // Trace set
    {
        VkDescriptorBufferInfo ubo{frameUbo, 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo nr  = sampledRO(rt.gNormalRough.view());
        VkDescriptorImageInfo dp  = sampledRO(rt.depth.view());
        VkDescriptorImageInfo sdf = sampledRO(sdfgi.udf().view());
        VkDescriptorImageInfo vox = sampledRO(vxgi.fullView());
        VkDescriptorImageInfo smp{}; smp.sampler = m_linearClamp;
        VkDescriptorImageInfo out = storage(rt.ssgi.view());

        std::array<VkWriteDescriptorSet, 7> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = m_traceSet; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &ubo;
        auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t,
                          const VkDescriptorImageInfo* p) {
            W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            W.dstSet = m_traceSet; W.dstBinding = bi; W.descriptorCount = 1;
            W.descriptorType = t; W.pImageInfo = p;
        };
        setImg(w[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &nr);
        setImg(w[2], 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &dp);
        setImg(w[3], 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &sdf);
        setImg(w[4], 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &vox);
        setImg(w[5], 5, VK_DESCRIPTOR_TYPE_SAMPLER,       &smp);
        setImg(w[6], 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &out);
        vkUpdateDescriptorSets(dev, (uint32_t)w.size(), w.data(), 0, nullptr);
    }
}

// ===== record ===============================================================

void SdfgiPass::record(VkCommandBuffer cmd,
                       const SdfgiResources& sdfgi,
                       const RenderTargets& rt,
                       uint32_t frameIndex,
                       float    seedThr,
                       float    maxDistC,
                       uint32_t numRaysIn,
                       uint32_t maxStepsIn,
                       float    rayMaxC,
                       float    hitEpsC) {
    uint32_t R = sdfgi.resolution();
    uint32_t g = (R + 3) / 4;

    VkImage A = sdfgi.seedA().image();
    VkImage B = sdfgi.seedB().image();
    VkImage U = sdfgi.udf().image();

    // ----- 1. seed -----------------------------------------------------------
    // 入口约定：A、B、U 都在 GENERAL（外部首次 transition），状态合法即可写。

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_seedPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_seedPlLayout, 0, 1, &m_seedSet, 0, nullptr);
    SeedPC sp{}; sp.resolution = R; sp.threshold = seedThr;
    vkCmdPushConstants(cmd, m_seedPlLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sp), &sp);
    vkCmdDispatch(cmd, g, g, g);

    // seed 写完 A，下步要把 A 转到 SHADER_READ_ONLY 让 JFA 当 src 采样。
    {
        auto b = makeBarrier3D(A,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        issueBarrier(cmd, b);
    }

    // ----- 2. JFA 多步：k = R/2, R/4, ..., 1 ----------------------------------
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_jfaPipeline);

    bool srcIsA = true;   // 当前 src 是 A
    int step = 0;
    for (uint32_t k = R / 2; k >= 1; k = k >> 1) {
        VkDescriptorSet ds = srcIsA ? m_jfaSetAB : m_jfaSetBA;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_jfaPlLayout, 0, 1, &ds, 0, nullptr);
        JfaPC jp{}; jp.resolution = R; jp.k = (int)k;
        vkCmdPushConstants(cmd, m_jfaPlLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(jp), &jp);
        vkCmdDispatch(cmd, g, g, g);

        // 写完 dst（B 或 A）。下一轮 dst 变 src，src 变 dst → 互转 layout。
        VkImage dstImg = srcIsA ? B : A;
        VkImage srcImg = srcIsA ? A : B;

        // dst 刚被 storage 写：GENERAL → SR_O（下一轮当 src 用，或 finalize 用）
        auto bDst = makeBarrier3D(dstImg,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        issueBarrier(cmd, bDst);

        // 上一轮 src 现在不再被采样 → 下一轮要当 dst 写：SR_O → GENERAL
        auto bSrc = makeBarrier3D(srcImg,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        issueBarrier(cmd, bSrc);

        srcIsA = !srcIsA;
        ++step;
        if (k == 1) break;
    }
    // JFA 收敛后：dst（最后写入的那张）已经在 SR_O（被上面 bDst 转过）。
    // src 在 GENERAL（被 bSrc 推过去给 "下一轮"，但循环已结束）。
    // finalize 要采样 dst 那张；选对 set。最后一轮 srcIsA 已 toggle 过。
    // 即：进入循环时 srcIsA=true，每轮末 toggle；
    // 最后一次写入的 dst 是 toggle 前的 (srcIsA ? B : A)；toggle 后 srcIsA
    // 已是反向 → "最后写入" = (toggle 后 srcIsA ? A : B)。换言之，循环退出
    // 时 srcIsA 指向 "下一轮的 src"，也就是上一轮的 dst，也就是 finalize 要
    // 用的那张 seed 图。
    bool finalSrcIsA = srcIsA;

    // ----- 3. finalize -------------------------------------------------------
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_finPipeline);
    VkDescriptorSet finDs = finalSrcIsA ? m_finSetA : m_finSetB;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_finPlLayout, 0, 1, &finDs, 0, nullptr);
    FinalizePC fp{}; fp.resolution = R; fp.maxDist = maxDistC;
    vkCmdPushConstants(cmd, m_finPlLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(fp), &fp);
    vkCmdDispatch(cmd, g, g, g);

    // udf 写完 → SR_O 给 trace 采样
    {
        auto b = makeBarrier3D(U,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        issueBarrier(cmd, b);
    }

    // ----- 4. trace ----------------------------------------------------------
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_tracePlLayout, 0, 1, &m_traceSet, 0, nullptr);
    TracePC tp{};
    tp.outSizeX = rt.extent.width; tp.outSizeY = rt.extent.height;
    tp.invOutSizeX = 1.0f / (float)rt.extent.width;
    tp.invOutSizeY = 1.0f / (float)rt.extent.height;
    tp.numRays = numRaysIn;
    tp.maxSteps = maxStepsIn;
    tp.rayMaxCells = rayMaxC;
    tp.hitEpsCells = hitEpsC;
    tp.frameIndex = frameIndex;
    vkCmdPushConstants(cmd, m_tracePlLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(tp), &tp);
    uint32_t gx = (rt.extent.width  + 7) / 8;
    uint32_t gy = (rt.extent.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    // ----- 5. 收尾：把所有 image 转回"空闲约定状态"--------------------------
    // udf 留在 SR_O 给 lighting 边缘读（其实 lighting 不读 udf，留 SR_O 也行）。
    // seedA / seedB：当前一个在 SR_O，一个在 GENERAL。下一帧 seed pass 写 A，
    // 需要 A 在 GENERAL。逻辑：finalSrcIsA == true 表示 finalize 读了 A
    // → 下一帧需把 A 转 GENERAL；B 当时在 GENERAL（toggle 后未再 dispatch）
    // → 已是 GENERAL，OK。反之亦然。
    {
        VkImage needGeneral = finalSrcIsA ? A : B;
        auto b = makeBarrier3D(needGeneral,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        issueBarrier(cmd, b);

        // udf：SR_O → GENERAL 给下一帧 finalize 写
        auto b2 = makeBarrier3D(U,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        issueBarrier(cmd, b2);
    }
}

}
