#include "renderer/gi/ndgi/ndgi_pass.h"
#include "renderer/gi/ndgi/ndgi_resources.h"
#include "core/device.h"
#include "renderer/gi/rt/scene_rt_as.h"
#include "renderer/core/render_targets.h"
#include "scene/scene.h"
#include <array>

namespace somegi {

void NdgiPass::init(Device& d, bool rtSupported) {
    m_device = &d;
    m_rtSupported = rtSupported;
    if (!rtSupported) return;
    auto sd = shaderDir() / "gi" / "ndgi";

    // ===== Probe Trace Pipeline =====
    {
        std::array<VkDescriptorSetLayoutBinding, 10> tb{};
        tb[0] = {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT};
        tb[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};   // instances
        tb[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};   // vertices
        tb[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};   // indices
        tb[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};   // materials
        tb[5] = {5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 128, VK_SHADER_STAGE_COMPUTE_BIT};  // textures
        tb[6] = {6, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
        tb[7] = {7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};   // frame UBO
        tb[8] = {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};   // sample buf
        tb[9] = {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};   // sample count

        VkDescriptorSetLayoutCreateInfo dsci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsci.bindingCount = (uint32_t)tb.size(); dsci.pBindings = tb.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &dsci, nullptr, &m_traceDsl));

        VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64}; // ProbeTracePC
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1; plci.pSetLayouts = &m_traceDsl;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_tracePipelineLayout));

        ShaderModule shader(d, sd / "ndgi_probe_trace.spv");
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = shader.handle(); cpci.stage.pName = "cs_main";
        cpci.layout = m_tracePipelineLayout;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_tracePipeline));

        // Descriptor pool + set for trace
        std::array<VkDescriptorPoolSize, 5> tps{{
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 128},
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        }};
        VkDescriptorPoolCreateInfo tpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        tpci.maxSets = 1; tpci.poolSizeCount = (uint32_t)tps.size(); tpci.pPoolSizes = tps.data();
        VK_CHECK(vkCreateDescriptorPool(d.device(), &tpci, nullptr, &m_tracePool));
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = m_tracePool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_traceDsl;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_traceSet));
    }

    // ===== Weight Init & Training Pipeline =====
    {
        // 8 bindings: 0-5 weights, 6 samples, 7 sample count
        std::array<VkDescriptorSetLayoutBinding, 8> ib{};
        for (uint32_t i = 0; i < 8; ++i)
            ib[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};

        VkDescriptorSetLayoutCreateInfo dsci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsci.bindingCount = (uint32_t)ib.size(); dsci.pBindings = ib.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &dsci, nullptr, &m_initDsl));

        // Init pipeline
        {
            VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
            VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            plci.setLayoutCount = 1; plci.pSetLayouts = &m_initDsl;
            plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
            VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_initPipelineLayout));
            ShaderModule shader(d, sd / "ndgi_init.spv");
            VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            cpci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module=shader.handle(); cpci.stage.pName="cs_main";
            cpci.layout=m_initPipelineLayout;
            VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_initPipeline));
        }
        // Training pipeline
        {
            VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
            VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            plci.setLayoutCount = 1; plci.pSetLayouts = &m_initDsl;
            plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
            VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_trainPipelineLayout));
            ShaderModule shader(d, sd / "ndgi_train.spv");
            VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            cpci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module=shader.handle(); cpci.stage.pName="cs_main";
            cpci.layout=m_trainPipelineLayout;
            VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_trainPipeline));
        }

        VkDescriptorPoolSize ips{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
        VkDescriptorPoolCreateInfo ipci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ipci.maxSets = 1; ipci.poolSizeCount = 1; ipci.pPoolSizes = &ips;
        VK_CHECK(vkCreateDescriptorPool(d.device(), &ipci, nullptr, &m_initPool));
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = m_initPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_initDsl;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_initSet));
    }
}

void NdgiPass::destroy() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_tracePool) vkDestroyDescriptorPool(dev, m_tracePool, nullptr);
    if (m_traceDsl) vkDestroyDescriptorSetLayout(dev, m_traceDsl, nullptr);
    if (m_tracePipeline) vkDestroyPipeline(dev, m_tracePipeline, nullptr);
    if (m_tracePipelineLayout) vkDestroyPipelineLayout(dev, m_tracePipelineLayout, nullptr);
    if (m_initPool) vkDestroyDescriptorPool(dev, m_initPool, nullptr);
    if (m_initDsl) vkDestroyDescriptorSetLayout(dev, m_initDsl, nullptr);
    if (m_initPipeline) vkDestroyPipeline(dev, m_initPipeline, nullptr);
    if (m_initPipelineLayout) vkDestroyPipelineLayout(dev, m_initPipelineLayout, nullptr);
    if (m_trainPipeline) vkDestroyPipeline(dev, m_trainPipeline, nullptr);
    if (m_trainPipelineLayout) vkDestroyPipelineLayout(dev, m_trainPipelineLayout, nullptr);
    m_device = nullptr;
}

void NdgiPass::bindResources(Device& d, NdgiResources& res, SceneRtAS& rtAS,
                              const SceneGpu& scene, const RenderTargets& rt,
                              VkBuffer frameUbo) {
    if (!m_rtSupported || m_traceSet == VK_NULL_HANDLE) return;

    // Write trace descriptor set
    std::array<VkWriteDescriptorSet, 10> w{};

    // TLAS via write info (matches RtGiPass pattern)
    VkWriteDescriptorSetAccelerationStructureKHR tlasAI = rtAS.tlasWriteInfo();
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_traceSet; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    w[0].pNext = &tlasAI;

    VkDescriptorBufferInfo instI{rtAS.instanceDataBuffer(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo vertI{scene.vertexBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo indI{scene.indexBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo matI{scene.materialBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo frameI{frameUbo, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo sampI{res.sampleBuf().handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo cntI{res.sampleCount().handle(), 0, VK_WHOLE_SIZE};

    auto setBuf = [&](uint32_t i, uint32_t bi, const VkDescriptorBufferInfo* p) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = m_traceSet; w[i].dstBinding = bi; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = p;
    };
    setBuf(1, 1, &instI);
    setBuf(2, 2, &vertI);
    setBuf(3, 3, &indI);
    setBuf(4, 4, &matI);
    // Binding 7 = UniformBuffer (frame UBO), 需要用 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    w[7] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[7].dstSet = m_traceSet; w[7].dstBinding = 7; w[7].descriptorCount = 1;
    w[7].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[7].pBufferInfo = &frameI;
    setBuf(8, 8, &sampI);
    setBuf(9, 9, &cntI);

    // sampler + textures
    VkDescriptorImageInfo smpI{}; smpI.sampler = scene.linearSampler;
    w[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[6].dstSet = m_traceSet; w[6].dstBinding = 6; w[6].descriptorCount = 1;
    w[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[6].pImageInfo = &smpI;

    std::vector<VkDescriptorImageInfo> texInfos;
    texInfos.reserve(128);
    for (uint32_t i = 0; i < 128; ++i) {
        VkDescriptorImageInfo ii{};
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii.imageView = (i < (uint32_t)scene.images.size()) ? scene.images[i].view() : scene.whiteTex.view();
        texInfos.push_back(ii);
    }
    w[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[5].dstSet = m_traceSet; w[5].dstBinding = 5; w[5].descriptorCount = 128;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[5].pImageInfo = texInfos.data();

    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);

    writeInitDescriptors(d, res);
}

void NdgiPass::writeInitDescriptors(Device& d, NdgiResources& res) {
    if (m_initSet == VK_NULL_HANDLE) return;
    std::array<VkDescriptorBufferInfo, 8> initInfos{{
        {res.weights1().handle(), 0, VK_WHOLE_SIZE},
        {res.bias1().handle(), 0, VK_WHOLE_SIZE},
        {res.weights2().handle(), 0, VK_WHOLE_SIZE},
        {res.bias2().handle(), 0, VK_WHOLE_SIZE},
        {res.weights3().handle(), 0, VK_WHOLE_SIZE},
        {res.bias3().handle(), 0, VK_WHOLE_SIZE},
        {res.sampleBuf().handle(), 0, VK_WHOLE_SIZE},
        {res.sampleCount().handle(), 0, VK_WHOLE_SIZE},
    }};
    std::array<VkWriteDescriptorSet, 8> iw{};
    for (uint32_t i = 0; i < 8; ++i) {
        iw[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        iw[i].dstSet = m_initSet; iw[i].dstBinding = i; iw[i].descriptorCount = 1;
        iw[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; iw[i].pBufferInfo = &initInfos[i];
    }
    vkUpdateDescriptorSets(d.device(), (uint32_t)iw.size(), iw.data(), 0, nullptr);
}

void NdgiPass::initWeights(VkCommandBuffer cmd) {
    if (!m_rtSupported || m_initSet == VK_NULL_HANDLE || m_initPipeline == VK_NULL_HANDLE) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_initPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_initPipelineLayout, 0, 1, &m_initSet, 0, nullptr);
    struct { uint32_t seed; float scale; uint32_t p0, p1; } pc{42, 1.0f, 0, 0};
    vkCmdPushConstants(cmd, m_initPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &pc);
    vkCmdDispatch(cmd, 1, 1, 1);  // 64 threads handle all weights
}

void NdgiPass::record(VkCommandBuffer cmd, NdgiResources& res, uint32_t frameIndex,
                       glm::vec3 origin, glm::vec3 spacing) {
    if (!m_rtSupported || m_traceSet == VK_NULL_HANDLE) return;

    // Reset sample count
    vkCmdFillBuffer(cmd, res.sampleCount().handle(), 0, 4, 0);

    // Probe trace
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_tracePipelineLayout, 0, 1, &m_traceSet, 0, nullptr);

    struct {
        float origin[3], pad0;
        float spacing[3], pad1;
        uint32_t px, py, pz, rpp;
        float rotation, _pad2;
        uint32_t _pad3;
    } pc;
    pc.origin[0] = origin.x; pc.origin[1] = origin.y; pc.origin[2] = origin.z;
    pc.spacing[0] = spacing.x; pc.spacing[1] = spacing.y; pc.spacing[2] = spacing.z;
    pc.px = NdgiResources::kProbesX;
    pc.py = NdgiResources::kProbesY;
    pc.pz = NdgiResources::kProbesZ;
    pc.rpp = NdgiResources::kRaysPerProbe;
    pc.rotation = float((frameIndex % 360) * 0.0174532925);

    vkCmdPushConstants(cmd, m_tracePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(pc), &pc);

    uint32_t totalRays = pc.px * pc.py * pc.pz * pc.rpp;
    vkCmdDispatch(cmd, (totalRays + 63) / 64, 1, 1);
}

void NdgiPass::recordTraining(VkCommandBuffer cmd, NdgiResources& res, uint32_t /*frameIndex*/) {
    if (!m_rtSupported || !m_trainPipeline || m_initSet == VK_NULL_HANDLE) return;

    uint32_t totalSamples = 0;
    // Read sample count from host buffer (mapped)
    auto* cnt = static_cast<uint32_t*>(res.sampleCount().mapped());
    if (cnt) totalSamples = *cnt;
    if (totalSamples == 0) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_trainPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_trainPipelineLayout, 0, 1, &m_initSet, 0, nullptr);

    struct { float lr, ema; uint32_t batch, iters, samples, p0, p1, p2; } pc;
    pc.lr = 0.01f;
    pc.ema = 0.95f;
    pc.batch = 256;
    pc.iters = 4;
    pc.samples = totalSamples;
    vkCmdPushConstants(cmd, m_trainPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, &pc);
    vkCmdDispatch(cmd, 1, 1, 1);
}

} // namespace somegi
