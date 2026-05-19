#include "gbuffer_pass.h"
#include "core/device.h"
#include <array>
#include <cstring>

namespace somegi {

void GBufferPass::init(Device& d,
                       VkFormat rt0Fmt, VkFormat rt1Fmt, VkFormat rt2Fmt,
                       VkFormat depthFmt, uint32_t maxTextures) {
    m_device = &d;
    m_rt0Fmt = rt0Fmt; m_rt1Fmt = rt1Fmt; m_rt2Fmt = rt2Fmt;
    m_depthFmt = depthFmt;
    m_maxTextures = maxTextures;

    // === Set=0 layout (mirrors ForwardPass) ===
    std::array<VkDescriptorSetLayoutBinding, 4> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,   maxTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

    std::array<VkDescriptorBindingFlags, 4> bf{0u, 0u, 0u,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT};
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bfci.bindingCount = (uint32_t)bf.size(); bfci.pBindingFlags = bf.data();

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.pNext = &bfci;
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    std::array<VkDescriptorPoolSize, 4> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, maxTextures},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_set));

    m_frameUbo = Buffer(d, sizeof(FrameUBO),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    buildPipeline();
}

void GBufferPass::buildPipeline() {
    auto& d = *m_device;

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pc.size = 64 + 16;   // mat4 model + int materialIndex + 3 ints pad

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule shader(d, shaderDir() / "gbuffer" / "gbuffer.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = shader.handle(); stages[0].pName = "vs_main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = shader.handle(); stages[1].pName = "ps_main";

    VkVertexInputBindingDescription vib{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription, 4> via{};
    via[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(Vertex,position)};
    via[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(Vertex,normal)};
    via[2] = {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex,tangent)};
    via[3] = {3, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(Vertex,uv0)};

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = (uint32_t)via.size(); vi.pVertexAttributeDescriptions = via.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    // Three identical color blend attachments (no blending, full RGBA write).
    std::array<VkPipelineColorBlendAttachmentState, 3> ba{};
    for (auto& a : ba) a.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = (uint32_t)ba.size(); cb.pAttachments = ba.data();

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyni{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyni.dynamicStateCount = 2; dyni.pDynamicStates = dyn;

    std::array<VkFormat, 3> colorFmts{m_rt0Fmt, m_rt1Fmt, m_rt2Fmt};
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = (uint32_t)colorFmts.size();
    rci.pColorAttachmentFormats = colorFmts.data();
    rci.depthAttachmentFormat = m_depthFmt;

    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.pNext = &rci;
    gpci.stageCount = 2; gpci.pStages = stages;
    gpci.pVertexInputState = &vi; gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp; gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms; gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb; gpci.pDynamicState = &dyni;
    gpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(d.device(), VK_NULL_HANDLE, 1, &gpci, nullptr, &m_pipeline));
}

void GBufferPass::destroyPipeline() {
    if (!m_device) return;
    if (m_pipeline)       vkDestroyPipeline(m_device->device(), m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(m_device->device(), m_pipelineLayout, nullptr);
    m_pipeline = VK_NULL_HANDLE; m_pipelineLayout = VK_NULL_HANDLE;
}

void GBufferPass::destroy() {
    if (!m_device) return;
    destroyPipeline();
    auto dev = m_device->device();
    if (m_pool)      vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout) vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    m_pool = VK_NULL_HANDLE; m_setLayout = VK_NULL_HANDLE;
    m_frameUbo.reset();
    m_device = nullptr;
}

void GBufferPass::bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount) {
    VkDescriptorBufferInfo uboInfo{m_frameUbo.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo matInfo{gpu.materialBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = gpu.linearSampler;

    std::vector<VkDescriptorImageInfo> imgs;
    imgs.reserve(m_maxTextures);
    for (uint32_t i = 0; i < m_maxTextures; ++i) {
        VkDescriptorImageInfo ii{};
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        if (i < textureCount && i < gpu.images.size())
            ii.imageView = gpu.images[i].view();
        else
            ii.imageView = gpu.whiteTex.view();
        imgs.push_back(ii);
    }

    std::array<VkWriteDescriptorSet, 4> w{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_set; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &uboInfo;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[1].dstSet = m_set; w[1].dstBinding = 1; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &matInfo;
    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[2].dstSet = m_set; w[2].dstBinding = 2; w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[2].pImageInfo = &samplerInfo;
    w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[3].dstSet = m_set; w[3].dstBinding = 3; w[3].descriptorCount = m_maxTextures;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[3].pImageInfo = imgs.data();

    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void GBufferPass::updateFrame(const FrameUBO& ubo) {
    std::memcpy(m_frameUbo.mapped(), &ubo, sizeof(FrameUBO));
}

void GBufferPass::record(VkCommandBuffer cmd, const RenderTargets& rt,
                         const SceneCpu& cpu, const SceneGpu& gpu) {
    std::array<VkRenderingAttachmentInfo, 3> color{};
    auto setColor = [](VkRenderingAttachmentInfo& a, VkImageView view) {
        a = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        a.imageView   = view;
        a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        a.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        a.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        a.clearValue.color = {{0, 0, 0, 0}};
    };
    setColor(color[0], rt.gAlbedoMetal.view());
    setColor(color[1], rt.gNormalRough.view());
    setColor(color[2], rt.gEmissiveAO.view());

    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageView   = rt.depth.view();
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    depth.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, rt.extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = (uint32_t)color.size();
    ri.pColorAttachments    = color.data();
    ri.pDepthAttachment     = &depth;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{0, 0, (float)rt.extent.width, (float)rt.extent.height, 0, 1};
    VkRect2D sc{{0, 0}, rt.extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

    VkDeviceSize zero = 0;
    VkBuffer vb = gpu.vertexBuffer.handle();
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
    vkCmdBindIndexBuffer(cmd, gpu.indexBuffer.handle(), 0, VK_INDEX_TYPE_UINT32);

    struct PC {
        glm::mat4 model;
        int materialIndex;
        int p0, p1, p2;
    } pc;
    for (auto& n : cpu.nodes) {
        if (n.meshIndex < 0) continue;
        const Mesh& M = cpu.meshes[n.meshIndex];
        pc.model = n.worldTransform;
        for (auto& p : M.primitives) {
            pc.materialIndex = p.materialIndex >= 0 ? p.materialIndex : 0;
            pc.p0 = pc.p1 = pc.p2 = 0;
            vkCmdPushConstants(cmd, m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(PC), &pc);
            vkCmdDrawIndexed(cmd, p.indexCount, 1, p.firstIndex, p.vertexOffset, 0);
        }
    }

    vkCmdEndRendering(cmd);
}

}
