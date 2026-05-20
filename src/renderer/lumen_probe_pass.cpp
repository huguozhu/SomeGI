#include "lumen_probe_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include "scene/scene_gpu.h"
#include "vxgi_resources.h"
#include "scene_rt_as.h"
#include <array>

namespace somegi {

namespace {
struct ProbePC {
    float    screenSizeX, screenSizeY;
    float    invScreenSizeX, invScreenSizeY;
    uint32_t probeGridW;
    uint32_t probeGridH;
    uint32_t probeTileSize;
    uint32_t raysPerProbe;
    uint32_t totalProbes;
    float    randomSeed;
    uint32_t useSixAxis;
    uint32_t _pad1, _pad2, _pad3;   // align to 64 bytes
};
static_assert(sizeof(ProbePC) <= 128);
}

void LumenProbePass::init(Device& d) {
    m_device = &d;

    // Sampler: linear, clamp
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 16.0f;
    VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_linearClamp));

    // 0:FrameUBO 1:NormalRough 2:Depth 3:TLAS 4:VoxelGrid 5:Sampler
    // 6:RayBuf 7:ProbeAtlas 8:SixAxisX 9:SixAxisY 10:SixAxisZ
    std::array<VkDescriptorSetLayoutBinding, 11> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,              1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,               1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,               1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,               1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_SAMPLER,                     1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[6] = {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,              1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[7] = {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,               1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[8] = {8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,               1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[9] = {9, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,               1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[10]= {10,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,               1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    // Pool
    std::array<VkDescriptorPoolSize, 6> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,            1},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,             6},   // +3 sixAxis
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLER,                   1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,            1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,             1},
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
    pc.size = sizeof(ProbePC);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    // Compute pipelines (two entry points from same .spv)
    auto spv = shaderDir() / "gi" / "lumen" / "lumen_probe.spv";
    auto makeCompute = [&](const char* entry) {
        ShaderModule cs(d, spv);
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = entry;
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = stage; cpci.layout = m_pipelineLayout;
        VkPipeline p = VK_NULL_HANDLE;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &p));
        return p;
    };
    m_pipelineRays = makeCompute("cs_generateRays");
    m_pipelineSH   = makeCompute("cs_projectSH");
}

void LumenProbePass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_pipelineSH)     vkDestroyPipeline(dev, m_pipelineSH, nullptr);
    if (m_pipelineRays)   vkDestroyPipeline(dev, m_pipelineRays, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_pool)           vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout)      vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    if (m_linearClamp)    vkDestroySampler(dev, m_linearClamp, nullptr);
    *this = {};
}

void LumenProbePass::bindResources(Device& d, const LumenResources& res,
                                    const SceneRtAS& rtAS, const SceneGpu& /*sceneGpu*/,
                                    const VxgiResources& vxgi, const RenderTargets& rt,
                                    VkBuffer frameUbo, bool hasSixAxis) {
    VkDescriptorBufferInfo ub{frameUbo, 0, VK_WHOLE_SIZE};

    VkDescriptorImageInfo nr{};
    nr.imageView = rt.gNormalRough.view();
    nr.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo depth{};
    depth.imageView = rt.depth.view();
    depth.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSetAccelerationStructureKHR tlasW = rtAS.tlasWriteInfo();

    VkDescriptorImageInfo vox{};
    vox.imageView = vxgi.fullView();
    vox.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo smp{}; smp.sampler = m_linearClamp;

    VkDescriptorBufferInfo rb{res.rayBuffer().handle(), 0, VK_WHOLE_SIZE};

    VkDescriptorImageInfo pa{};
    pa.imageView = res.probeAtlas().view();
    pa.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 11> w{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_set; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &ub;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[1].dstSet = m_set; w[1].dstBinding = 1; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[1].pImageInfo = &nr;
    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[2].dstSet = m_set; w[2].dstBinding = 2; w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[2].pImageInfo = &depth;
    w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[3].dstSet = m_set; w[3].dstBinding = 3; w[3].descriptorCount = 1;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    w[3].pNext = &tlasW;
    w[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[4].dstSet = m_set; w[4].dstBinding = 4; w[4].descriptorCount = 1;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[4].pImageInfo = &vox;
    w[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[5].dstSet = m_set; w[5].dstBinding = 5; w[5].descriptorCount = 1;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[5].pImageInfo = &smp;
    w[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[6].dstSet = m_set; w[6].dstBinding = 6; w[6].descriptorCount = 1;
    w[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[6].pBufferInfo = &rb;
    w[7] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[7].dstSet = m_set; w[7].dstBinding = 7; w[7].descriptorCount = 1;
    w[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[7].pImageInfo = &pa;

    // L.3b 6-axis bindings (use vxgi isotropic as fallback if not available)
    VkDescriptorImageInfo ax{};
    ax.imageView = hasSixAxis ? vxgi.sixAxisX().view() : vxgi.fullView();
    ax.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo ay{};
    ay.imageView = hasSixAxis ? vxgi.sixAxisY().view() : vxgi.fullView();
    ay.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo az{};
    az.imageView = hasSixAxis ? vxgi.sixAxisZ().view() : vxgi.fullView();
    az.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    w[8] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[8].dstSet = m_set; w[8].dstBinding = 8; w[8].descriptorCount = 1;
    w[8].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[8].pImageInfo = &ax;
    w[9] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[9].dstSet = m_set; w[9].dstBinding = 9; w[9].descriptorCount = 1;
    w[9].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[9].pImageInfo = &ay;
    w[10] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[10].dstSet = m_set; w[10].dstBinding = 10; w[10].descriptorCount = 1;
    w[10].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[10].pImageInfo = &az;

    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void LumenProbePass::record(VkCommandBuffer cmd, const LumenResources& res,
                             uint32_t frameIndex, bool useSixAxis) {
    uint32_t pw = res.probeGridW();
    uint32_t ph = res.probeGridH();
    uint32_t pc = res.probeCount();

    // 1. cs_generateRays
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineRays);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

    ProbePC upc{};
    upc.screenSizeX    = (float)(pw * LumenResources::kProbeTileSize);
    upc.screenSizeY    = (float)(ph * LumenResources::kProbeTileSize);
    upc.invScreenSizeX = 1.0f / upc.screenSizeX;
    upc.invScreenSizeY = 1.0f / upc.screenSizeY;
    upc.probeGridW     = pw;
    upc.probeGridH     = ph;
    upc.probeTileSize  = LumenResources::kProbeTileSize;
    upc.raysPerProbe   = LumenResources::kRaysPerProbe;
    upc.totalProbes    = pc;
    upc.randomSeed     = (float)(frameIndex % 359) * 0.0174533f;
    upc.useSixAxis     = useSixAxis ? 1u : 0u;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(upc), &upc);

    uint32_t totalRays = pc * LumenResources::kRaysPerProbe;
    vkCmdDispatch(cmd, (totalRays + 63) / 64, 1, 1);

    // Barrier: ray buffer write → SH project read
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);

    // 2. cs_projectSH
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineSH);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(upc), &upc);
    vkCmdDispatch(cmd, (pc + 63) / 64, 1, 1);
}

}
