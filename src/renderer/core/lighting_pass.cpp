// LightingPass 实现 —— Vulkan 资源 / 描述符 / 管线 / 录制的封装。
// 算法和 shader binding 约定见 lighting_pass.h 的类注释和
// shaders/lighting/lighting.slang 的文件头。
//
// 调用顺序约定：init → setTechnique → bindFrame → 每帧 record。
// onSwapchainResized 时 GBuffer image 换新 → 调用方需重新 bindFrame；
// GI 模式切换通过 FrameUBO 的各路闸门（counts/lpvCounts/vxgiCounts 等）
// 在 shader 内分支，不需要重建 pipeline；仅 set=1 描述符布局变更时
// （如 IBLTechnique 初始化）才通过 setTechnique 重建 pipeline。

#include "renderer/core/lighting_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>
#include <cstring>

namespace somegi {

namespace {
// push constant 数据布局，shader 端 LightingPC 与此对齐。
struct LightingPC {
    uint32_t outSizeX, outSizeY;
    float    invOutSizeX, invOutSizeY;
};
static_assert(sizeof(LightingPC) == 16, "LightingPC must match shader push constant layout");
}

void LightingPass::init(Device& d) {
    m_device = &d;

    // === Set=0 layout ===
    // 0: UBO (FrameUniforms)
    // 1: gAlbedoMetal      (sampled image, .Load only — no sampler binding)
    // 2: gNormalRough
    // 3: gEmissiveAO
    // 4: gDepth
    // 5: gOutHdr           (storage image)
    // 6: gSsao             (sampled image, R8)
    // 7: gSsr              (sampled image, RGBA16F)
    // 8: gSsgi             (sampled image, RGBA16F)
    // 9: gRsmGI            (sampled image, RGBA16F) — M5
    // 10/11/12: gLpvR/G/B  (sampled 3D image, RGBA16F) — M6
    // 13: gLpvSampler      (linear clamp sampler) — M6/M7 共用
    // 14: gVoxelGrid       (sampled 3D image full mipchain, RGBA16F) — M7
    // 15: gPrtTransfer     (sampled 3D image, RGBA16F) — M8
    // 16: gDdgiIrr         (sampled 2D atlas, RGBA16F) — M11
    // 17: gDdgiDist        (sampled 2D atlas, RG16F) — M11
    // 18: gDdgiProbeStates (storage buffer, uint per probe) — B.5
    // 19: gVxgiAniso       (sampled 3D image full mipchain, RGBA16F) — B.6
    // 20/21: gPrtTransferB/C (sampled 3D image, RGBA16F) — B.9 SH9
    // 22/23: gPrtTransferD/E (sampled 3D image, RGBA16F) — B.10 SH16
    // 24: gRestir         (sampled 2D image, RGBA16F) — C.4 ReSTIR DI
    // 25: gRtGI           (sampled 2D image, RGBA16F) — M9 RT GI
    std::array<VkDescriptorSetLayoutBinding, 33> b{};
    b[0]  = {0,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1]  = {1,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2]  = {2,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3]  = {3,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4]  = {4,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5]  = {5,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[6]  = {6,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[7]  = {7,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[8]  = {8,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[9]  = {9,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[10] = {10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[11] = {11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[12] = {12, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[13] = {13, VK_DESCRIPTOR_TYPE_SAMPLER,        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[14] = {14, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[15] = {15, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[16] = {16, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[17] = {17, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[18] = {18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[19] = {19, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[20] = {20, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[21] = {21, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[22] = {22, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[23] = {23, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[24] = {24, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[25] = {25, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[26] = {26, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[27] = {27, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // NDGI W1
    b[28] = {28, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // NDGI B1
    b[29] = {29, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // NDGI W2
    b[30] = {30, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // NDGI B2
    b[31] = {31, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // NDGI W3
    b[32] = {32, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // NDGI B3

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    std::array<VkDescriptorPoolSize, 5> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 23},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1},
        {VK_DESCRIPTOR_TYPE_SAMPLER,        1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7},  // +6 NDGI weights
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_set));

    // NDGI 权重 binding 27-32 初始占位 buffer（4-byte 零值）
    m_dummyBuf = Buffer(d, 4,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    *static_cast<float*>(m_dummyBuf.mapped()) = 0.0f;

    // LPV trilinear sampler：clamp-to-edge 防 grid 边外 wrap 出问题。
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_lpvSampler));

    // 创建 IBL set=1 描述符布局（固定：3 张 cube 贴图 + sampler + params UBO）
    createIblDescriptorSetLayout();
    // 构建 pipeline（set=0 + set=1 IBL，之后不再重建）
    buildPipeline();
}

void LightingPass::createIblDescriptorSetLayout() {
    constexpr VkShaderStageFlags kStages =
        VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    std::array<VkDescriptorSetLayoutBinding, 5> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, kStages};  // diffuse cube
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, kStages};  // specular cube
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, kStages};  // brdf lut
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLER,        1, kStages};  // linear sampler
    b[4] = {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, kStages};  // params (intensity)

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_device->device(), &li, nullptr, &m_iblDsl));
}

void LightingPass::buildPipeline() {
    auto& d = *m_device;

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(LightingPC);

    std::array<VkDescriptorSetLayout, 2> sets{m_setLayout, m_iblDsl};

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 2;
    plci.pSetLayouts = sets.data();
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule cs(d, shaderDir() / "lighting" / "lighting.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs.handle();
    stage.pName = "cs_main";

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage;
    cpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline));
}

void LightingPass::destroyPipeline() {
    if (!m_device) return;
    if (m_pipeline)       vkDestroyPipeline(m_device->device(), m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(m_device->device(), m_pipelineLayout, nullptr);
    m_pipeline = VK_NULL_HANDLE; m_pipelineLayout = VK_NULL_HANDLE;
}

void LightingPass::bindIblResources(Device& d, const IblResources& ibl) {
    // 分配 IBL set=1 descriptor pool + set
    std::array<VkDescriptorPoolSize, 3> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  3},
        {VK_DESCRIPTOR_TYPE_SAMPLER,        1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_iblPool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_iblPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_iblDsl;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_iblSet));

    // 创建 params UBO（host-coherent，ImGui slider 直接写 mapped 内存）
    struct IblParams { float intensity; float _pad0, _pad1, _pad2; };
    m_iblParamsUbo = Buffer(d, sizeof(IblParams),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // 写入 IBL 描述符
    auto cubeInfo = [](VkImageView v) {
        VkDescriptorImageInfo i{};
        i.imageView = v;
        i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return i;
    };
    VkDescriptorImageInfo diffI = cubeInfo(ibl.diffuseCube.view());
    VkDescriptorImageInfo specI = cubeInfo(ibl.specularCube.view());
    VkDescriptorImageInfo lutI  = cubeInfo(ibl.brdfLut.view());
    VkDescriptorImageInfo smpI{}; smpI.sampler = ibl.linear;
    VkDescriptorBufferInfo uboI{m_iblParamsUbo.handle(), 0, VK_WHOLE_SIZE};

    std::array<VkWriteDescriptorSet, 5> w{};
    auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, const VkDescriptorImageInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_iblSet; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = (bi == 3) ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        W.pImageInfo = p;
    };
    setImg(w[0], 0, &diffI);
    setImg(w[1], 1, &specI);
    setImg(w[2], 2, &lutI);
    setImg(w[3], 3, &smpI);
    w[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[4].dstSet = m_iblSet; w[4].dstBinding = 4; w[4].descriptorCount = 1;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[4].pBufferInfo = &uboI;

    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);

    // 写入默认 intensity
    IblParams params{};
    params.intensity = m_iblIntensity;
    std::memcpy(m_iblParamsUbo.mapped(), &params, sizeof(params));
}

void LightingPass::destroy() {
    if (!m_device) return;
    destroyPipeline();
    auto dev = m_device->device();
    if (m_pool)      vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout) vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    if (m_iblPool)   vkDestroyDescriptorPool(dev, m_iblPool, nullptr);
    if (m_iblDsl)    vkDestroyDescriptorSetLayout(dev, m_iblDsl, nullptr);
    if (m_lpvSampler) vkDestroySampler(dev, m_lpvSampler, nullptr);
    m_pool = VK_NULL_HANDLE; m_setLayout = VK_NULL_HANDLE;
    m_iblPool = VK_NULL_HANDLE; m_iblDsl = VK_NULL_HANDLE;
    m_lpvSampler = VK_NULL_HANDLE;
    m_dummyBuf.reset();
    m_iblParamsUbo.reset();
    m_device = nullptr;
}

void LightingPass::bindFrame(Device& d, const RenderTargets& rt, VkBuffer frameUbo,
                             const LpvGrid& lpvGrid0, const VxgiResources& vxgi,
                             const PrtResources& prt, const DdgiResources& ddgi,
                             VkBuffer ddgiProbeStatesBuf) {
    VkDescriptorBufferInfo uboInfo{frameUbo, 0, VK_WHOLE_SIZE};

    auto sampledRO = [](VkImageView v) {
        VkDescriptorImageInfo i{};
        i.imageView = v;
        i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return i;
    };
    VkDescriptorImageInfo am = sampledRO(rt.gAlbedoMetal.view());
    VkDescriptorImageInfo nr = sampledRO(rt.gNormalRough.view());
    VkDescriptorImageInfo ea = sampledRO(rt.gEmissiveAO.view());
    VkDescriptorImageInfo dp = sampledRO(rt.depth.view());
    VkDescriptorImageInfo ss = sampledRO(rt.ssao.view());
    VkDescriptorImageInfo sr = sampledRO(rt.ssr.view());
    VkDescriptorImageInfo gi = sampledRO(rt.ssgi.view());
    VkDescriptorImageInfo rs = sampledRO(rt.rsmGI.view());
    VkDescriptorImageInfo lr = sampledRO(lpvGrid0.lpvR.view());
    VkDescriptorImageInfo lg = sampledRO(lpvGrid0.lpvG.view());
    VkDescriptorImageInfo lb = sampledRO(lpvGrid0.lpvB.view());
    VkDescriptorImageInfo ls{}; ls.sampler = m_lpvSampler;
    VkDescriptorImageInfo vx = sampledRO(vxgi.fullView());
    VkDescriptorImageInfo px  = sampledRO(prt.view());
    VkDescriptorImageInfo pxB = sampledRO(prt.viewB());
    VkDescriptorImageInfo pxC = sampledRO(prt.viewC());
    VkDescriptorImageInfo pxD = sampledRO(prt.viewD());
    VkDescriptorImageInfo pxE = sampledRO(prt.viewE());
    VkDescriptorImageInfo dgi = sampledRO(ddgi.irradiance().view());
    VkDescriptorImageInfo dgd = sampledRO(ddgi.distance().view());
    VkDescriptorBufferInfo psbuf{ddgiProbeStatesBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo vxa = sampledRO(vxgi.anisoFullView());
    VkDescriptorImageInfo rst = sampledRO(rt.restir.view());   // C.4 ReSTIR DI
    VkDescriptorImageInfo rtG = sampledRO(rt.rtGI.view());    // M9 RT GI
    VkDescriptorImageInfo lum = sampledRO(rt.lumenGI.view()); // L.5 Lumen-lite gather

    VkDescriptorImageInfo hd{};
    hd.imageView = rt.hdrColor.view();
    hd.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 33> w{};
    // ... (w[0] through w[24] are set as before)
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_set; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &uboInfo;
    auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorImageInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_set; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = t; W.pImageInfo = p;
    };
    setImg(w[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &am);
    setImg(w[2], 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &nr);
    setImg(w[3], 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ea);
    setImg(w[4], 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &dp);
    setImg(w[5], 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hd);
    setImg(w[6], 6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ss);
    setImg(w[7], 7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &sr);
    setImg(w[8],  8,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &gi);
    setImg(w[9],  9,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &rs);
    setImg(w[10], 10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &lr);
    setImg(w[11], 11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &lg);
    setImg(w[12], 12, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &lb);
    setImg(w[13], 13, VK_DESCRIPTOR_TYPE_SAMPLER,       &ls);
    setImg(w[14], 14, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &vx);
    setImg(w[15], 15, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &px);
    setImg(w[16], 16, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &dgi);
    setImg(w[17], 17, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &dgd);
    w[18] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[18].dstSet = m_set; w[18].dstBinding = 18; w[18].descriptorCount = 1;
    w[18].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[18].pBufferInfo = &psbuf;
    setImg(w[19], 19, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &vxa);
    setImg(w[20], 20, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &pxB);
    setImg(w[21], 21, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &pxC);
    setImg(w[22], 22, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &pxD);
    setImg(w[23], 23, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &pxE);
    setImg(w[24], 24, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &rst);
    setImg(w[25], 25, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &rtG);   // M9 RT GI
    setImg(w[26], 26, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &lum);  // L.5 Lumen-lite

    // NDGI weights (bindings 27-32): 初始指向 dummy zero buffer
    VkDescriptorBufferInfo dummyInfo{m_dummyBuf.handle(), 0, VK_WHOLE_SIZE};
    for (uint32_t i = 27; i <= 32; ++i) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = m_set; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &dummyInfo;
    }

    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void LightingPass::setNdgiWeights(Device& d,
    VkBuffer w1, VkBuffer b1, VkBuffer w2, VkBuffer b2,
    VkBuffer w3, VkBuffer b3) {
    VkDescriptorBufferInfo infos[6]{};
    infos[0] = {w1, 0, VK_WHOLE_SIZE}; infos[1] = {b1, 0, VK_WHOLE_SIZE};
    infos[2] = {w2, 0, VK_WHOLE_SIZE}; infos[3] = {b2, 0, VK_WHOLE_SIZE};
    infos[4] = {w3, 0, VK_WHOLE_SIZE}; infos[5] = {b3, 0, VK_WHOLE_SIZE};

    std::array<VkWriteDescriptorSet, 6> w{};
    for (uint32_t i = 0; i < 6; ++i) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = m_set; w[i].dstBinding = 27 + i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void LightingPass::setIblIntensity(float v) {
    m_iblIntensity = v;
    struct { float intensity; float _pad0, _pad1, _pad2; } p{};
    p.intensity = v;
    std::memcpy(m_iblParamsUbo.mapped(), &p, sizeof(p));
}

void LightingPass::record(VkCommandBuffer cmd, const RenderTargets& rt) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

    VkDescriptorSet sets[2] = {m_set, m_iblSet};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 2, sets, 0, nullptr);

    LightingPC pc{};
    pc.outSizeX    = rt.extent.width;
    pc.outSizeY    = rt.extent.height;
    pc.invOutSizeX = 1.0f / (float)rt.extent.width;
    pc.invOutSizeY = 1.0f / (float)rt.extent.height;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t gx = (rt.extent.width  + 7) / 8;
    uint32_t gy = (rt.extent.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);
}

}
