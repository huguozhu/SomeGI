#include "renderer/core/gbuffer_pass.h"
#include "core/device.h"
#include <array>
#include <cstring>

namespace somegi {

namespace {
struct PC {
    glm::mat4 model;
    int materialIndex;
    int p0, p1, p2;
};
static_assert(sizeof(PC) == 80, "PC must match shader push constant layout");
}

void GBufferPass::init(Device& d,
                       VkFormat rt0Fmt, VkFormat rt1Fmt, VkFormat rt2Fmt,
                       VkFormat depthFmt, uint32_t maxTextures,
                       VkSampleCountFlagBits msaaSamples) {
    m_device = &d;
    m_rt0Fmt = rt0Fmt; m_rt1Fmt = rt1Fmt; m_rt2Fmt = rt2Fmt;
    m_depthFmt = depthFmt;
    m_maxTextures = maxTextures;
    m_msaaSamples = msaaSamples;

    // === Set=0 layout (mirrors ForwardPass) ===
    std::array<VkDescriptorSetLayoutBinding, 5> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,   maxTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[4] = {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};

    std::array<VkDescriptorBindingFlags, 5> bf{0u, 0u, 0u,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, 0u};
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bfci.bindingCount = (uint32_t)bf.size(); bfci.pBindingFlags = bf.data();

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.pNext = &bfci;
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    std::array<VkDescriptorPoolSize, 4> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
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

    // ── Mesh Shader descriptor set layout（set=0，bindings 0-11）────
    {
        // 所有 Task-only 绑定同时声明 MESH stage，保证无 Task Shader 时 layout 仍有效
        const VkShaderStageFlags kTS = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
        std::array<VkDescriptorSetLayoutBinding, 12> mb{};
        mb[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTS, nullptr};
        mb[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTS, nullptr};  // MeshGroup 映射
        for (uint32_t i = 0; i < 4; ++i)
            mb[2+i] = {2+i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, kTS, nullptr};
        mb[6] = {6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, kTS | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        mb[7] = {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTS, nullptr};
        mb[8] = {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kTS, nullptr};
        mb[9] = {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        mb[10]= {10, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        mb[11]= {11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, m_maxTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

        // 纹理数组 binding 设 PARTIALLY_BOUND（与 VS 路径对齐）
        std::array<VkDescriptorBindingFlags, 12> mbf{};
        mbf[11] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo mbfci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        mbfci.bindingCount = (uint32_t)mbf.size(); mbfci.pBindingFlags = mbf.data();

        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.pNext = &mbfci;
        li.bindingCount = (uint32_t)mb.size(); li.pBindings = mb.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_meshSetLayout));

        std::array<VkDescriptorPoolSize, 4> mps{{
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5},  // bindings 0,1,7,8,9
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},  // binding 6
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, m_maxTextures + 4},
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        }};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.maxSets = 1; pci.poolSizeCount = (uint32_t)mps.size(); pci.pPoolSizes = mps.data();
        VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_meshPool));

        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = m_meshPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_meshSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_meshSet));

        m_cullUbo = Buffer(d, sizeof(glm::vec4)*6 + sizeof(glm::vec2) + sizeof(uint32_t)*2,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    // buildMeshPipeline() 延迟到 setMeshShaderEnabled(true) 调用时，避免 Intel IGC 在 init 阶段崩溃
}

void GBufferPass::buildPipeline() {
    auto& d = *m_device;


    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 0; plci.pPushConstantRanges = nullptr;
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
    ms.rasterizationSamples = m_msaaSamples;

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

void GBufferPass::setMsaaSamples(VkSampleCountFlagBits samples) {
    if (m_msaaSamples == samples) return;
    m_msaaSamples = samples;
    destroyPipeline();
    buildPipeline();
}

void GBufferPass::destroyPipeline() {
    if (!m_device) return;
    if (m_pipeline)       vkDestroyPipeline(m_device->device(), m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(m_device->device(), m_pipelineLayout, nullptr);
    if (m_meshPipeline)   vkDestroyPipeline(m_device->device(), m_meshPipeline, nullptr);
    if (m_meshPipelineLayout) vkDestroyPipelineLayout(m_device->device(), m_meshPipelineLayout, nullptr);
    m_pipeline = VK_NULL_HANDLE; m_pipelineLayout = VK_NULL_HANDLE;
    m_meshPipeline = VK_NULL_HANDLE; m_meshPipelineLayout = VK_NULL_HANDLE;
}

void GBufferPass::buildMeshPipeline() {
    auto& d = *m_device;
    m_meshPipelineLayout = VK_NULL_HANDLE;

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_meshSetLayout;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_meshPipelineLayout));

    auto sd = shaderDir();
    // glslang 编译的 GLSL mesh shader（Slang 不生成 OpSetMeshOutputsEXT）
    bool hasTask = false;
    ShaderModule meshMod(d, sd / "gbuffer" / "gbuffer_mesh_no_task_mesh.spv");
    ShaderModule fragMod(d, sd / "gbuffer" / "gbuffer_mesh_no_task_frag.spv");
    ShaderModule taskMod;  // 不使用

    std::vector<VkPipelineShaderStageCreateInfo> si;
    if (hasTask) {
        VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s.stage = VK_SHADER_STAGE_TASK_BIT_EXT; s.module = taskMod.handle(); s.pName = "ts_main";
        si.push_back(s);
    }
    {
        VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s.stage = VK_SHADER_STAGE_MESH_BIT_EXT; s.module = meshMod.handle(); s.pName = "main";
        si.push_back(s);
    }
    {
        VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s.stage = VK_SHADER_STAGE_FRAGMENT_BIT; s.module = fragMod.handle(); s.pName = "main";
        si.push_back(s);
    }

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 0; vi.vertexAttributeDescriptionCount = 0;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = m_msaaSamples;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    std::array<VkPipelineColorBlendAttachmentState, 3> ba{};
    for (auto& a : ba) a.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 3; cb.pAttachments = ba.data();

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyni{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyni.dynamicStateCount = 2; dyni.pDynamicStates = dyn;

    std::array<VkFormat, 3> colorFmts{m_rt0Fmt, m_rt1Fmt, m_rt2Fmt};
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 3; rci.pColorAttachmentFormats = colorFmts.data();
    rci.depthAttachmentFormat = m_depthFmt;

    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.pNext = &rci;
    gpci.stageCount = (uint32_t)si.size(); gpci.pStages = si.data();
    gpci.pVertexInputState = &vi; gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp; gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms; gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb; gpci.pDynamicState = &dyni;
    gpci.layout = m_meshPipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(d.device(), VK_NULL_HANDLE, 1, &gpci, nullptr, &m_meshPipeline));
}

void GBufferPass::setMeshShaderEnabled(bool v) {
    if (v && m_meshPipeline == VK_NULL_HANDLE) {
        buildMeshPipeline();  // 首次启用时才创建（Intel IGC 可能崩溃）
    }
    m_useMeshShader = v;
}

void GBufferPass::bindHiZViews(VkImageView mip1, VkImageView mip2, VkImageView mip3, VkImageView mip4) {
    auto hiZInfo = [](VkImageView v) {
        VkDescriptorImageInfo i{};
        i.imageView = v; i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; return i;
    };
    VkDescriptorImageInfo hz1 = hiZInfo(mip1), hz2 = hiZInfo(mip2), hz3 = hiZInfo(mip3), hz4 = hiZInfo(mip4);
    std::array<VkWriteDescriptorSet, 4> w{};
    for (uint32_t i = 0; i < 4; ++i) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = m_meshSet; w[i].dstBinding = 2 + i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w[i].pImageInfo = (i==0?&hz1:i==1?&hz2:i==2?&hz3:&hz4);
    }
    vkUpdateDescriptorSets(m_device->device(), 4, w.data(), 0, nullptr);
}

void GBufferPass::destroy() {
    if (!m_device) return;
    destroyPipeline();
    auto dev = m_device->device();
    if (m_pool)          vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout)     vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    if (m_meshPool)      vkDestroyDescriptorPool(dev, m_meshPool, nullptr);
    if (m_meshSetLayout) vkDestroyDescriptorSetLayout(dev, m_meshSetLayout, nullptr);
    m_pool = VK_NULL_HANDLE; m_setLayout = VK_NULL_HANDLE;
    m_meshPool = VK_NULL_HANDLE; m_meshSetLayout = VK_NULL_HANDLE;
    m_frameUbo.reset();
    m_cullUbo.reset();
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

    // ── Mesh Shader 的 set=0 描述符（仅在启用 mesh 时写入，避免 Intel 驱动崩溃）──
    if (m_useMeshShader) {
    VkDescriptorBufferInfo vbInfo{gpu.vertexBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo ibInfo{gpu.indexBuffer.handle(), 0, VK_WHOLE_SIZE};
    std::array<VkWriteDescriptorSet, 6> mw{};  // bindings 6-11
    mw[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    mw[0].dstSet=m_meshSet;mw[0].dstBinding=6;mw[0].descriptorCount=1;
    mw[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;mw[0].pBufferInfo=&uboInfo;
    mw[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    mw[1].dstSet=m_meshSet;mw[1].dstBinding=7;mw[1].descriptorCount=1;
    mw[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;mw[1].pBufferInfo=&vbInfo;
    mw[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    mw[2].dstSet=m_meshSet;mw[2].dstBinding=8;mw[2].descriptorCount=1;
    mw[2].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;mw[2].pBufferInfo=&ibInfo;
    mw[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    mw[3].dstSet=m_meshSet;mw[3].dstBinding=9;mw[3].descriptorCount=1;
    mw[3].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;mw[3].pBufferInfo=&matInfo;
    mw[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    mw[4].dstSet=m_meshSet;mw[4].dstBinding=10;mw[4].descriptorCount=1;
    mw[4].descriptorType=VK_DESCRIPTOR_TYPE_SAMPLER;mw[4].pImageInfo=&samplerInfo;
    mw[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    mw[5].dstSet=m_meshSet;mw[5].dstBinding=11;mw[5].descriptorCount=m_maxTextures;
    mw[5].descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;mw[5].pImageInfo=imgs.data();
    // DrawData (binding 0) 在 bindDrawData 中写入
    vkUpdateDescriptorSets(d.device(), (uint32_t)mw.size(), mw.data(), 0, nullptr);
    } // if (m_useMeshShader)
}

void GBufferPass::bindDrawData(Device& d, VkBuffer drawDataBuf) {
    // VS 路径 binding 10
    VkDescriptorBufferInfo dd{drawDataBuf,0,VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet=m_set;w.dstBinding=10;w.descriptorCount=1;
    w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&dd;
    vkUpdateDescriptorSets(d.device(),1,&w,0,nullptr);
    // Mesh 路径 binding 0（同一 buffer，不同 binding）
    if (m_useMeshShader) {
        VkWriteDescriptorSet mw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        mw.dstSet=m_meshSet;mw.dstBinding=0;mw.descriptorCount=1;
        mw.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;mw.pBufferInfo=&dd;
        vkUpdateDescriptorSets(d.device(),1,&mw,0,nullptr);
    }
}
// 更新 Task Shader 的 CullUbo（每帧调用）
void GBufferPass::buildMeshGroups(const std::vector<DrawEntry>& entries) {
    struct MeshGroup { uint32_t drawIndex; uint32_t triOffset; };
    constexpr uint32_t kMaxTris = 85;
    std::vector<MeshGroup> groups;
    for (uint32_t d = 0; d < (uint32_t)entries.size(); ++d) {
        uint32_t totalTris = entries[d].indexCount / 3;
        for (uint32_t offset = 0; offset < totalTris; offset += kMaxTris) {
            groups.push_back({d, offset});
        }
    }
    m_meshGroupCount = (uint32_t)groups.size();
    if (m_meshGroupCount == 0) return;
    m_meshGroupBuf = Buffer(*m_device, m_meshGroupCount * sizeof(MeshGroup),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(m_meshGroupBuf.mapped(), groups.data(), m_meshGroupCount * sizeof(MeshGroup));
    // 写入 mesh descriptor set binding 1
    VkDescriptorBufferInfo gi{m_meshGroupBuf.handle(), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = m_meshSet; w.dstBinding = 1; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &gi;
    vkUpdateDescriptorSets(m_device->device(), 1, &w, 0, nullptr);
}

void GBufferPass::updateCullUbo(const glm::mat4& viewProj, const glm::vec4 frustum[6],
                                 uint32_t drawCount, uint32_t hizMaxMip,
                                 uint32_t screenW, uint32_t screenH) {
    struct {
        glm::mat4 viewProj;
        glm::vec4 frustum[6];
        glm::vec2 screenSize; uint32_t drawCount; uint32_t hizMaxMip;
    } cull{};
    cull.viewProj = viewProj;
    for (int i = 0; i < 6; ++i) cull.frustum[i] = frustum[i];
    cull.screenSize = glm::vec2((float)screenW, (float)screenH);
    cull.drawCount = drawCount;
    cull.hizMaxMip = hizMaxMip;
    std::memcpy(m_cullUbo.mapped(), &cull, sizeof(cull));
}
void GBufferPass::updateFrame(const FrameUBO& ubo) {
    std::memcpy(m_frameUbo.mapped(), &ubo, sizeof(FrameUBO));
}

void GBufferPass::record(VkCommandBuffer cmd, const RenderTargets& rt,
                         VkBuffer indirectBuf, uint32_t drawCount, const SceneGpu& gpu) {
    if (drawCount == 0) return;
    bool useMsaa = m_msaaSamples != VK_SAMPLE_COUNT_1_BIT;

    std::array<VkRenderingAttachmentInfo, 3> color{};
    for (int i = 0; i < 3; ++i) {
        color[i] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color[i].loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color[i].storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        color[i].clearValue.color = {{0, 0, 0, 0}};
    }
    color[0].imageView = useMsaa ? rt.gAlbedoMetalMs.view() : rt.gAlbedoMetal.view();
    color[1].imageView = useMsaa ? rt.gNormalRoughMs.view() : rt.gNormalRough.view();
    color[2].imageView = useMsaa ? rt.gEmissiveAOMs.view() : rt.gEmissiveAO.view();

    if (useMsaa) {
        color[0].resolveImageView   = rt.gAlbedoMetal.view();
        color[0].resolveMode        = VK_RESOLVE_MODE_AVERAGE_BIT;
        color[0].resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color[1].resolveImageView   = rt.gNormalRough.view();
        color[1].resolveMode        = VK_RESOLVE_MODE_AVERAGE_BIT;
        color[1].resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color[2].resolveImageView   = rt.gEmissiveAO.view();
        color[2].resolveMode        = VK_RESOLVE_MODE_AVERAGE_BIT;
        color[2].resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageView   = useMsaa ? rt.depthMs.view() : rt.depth.view();
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    depth.clearValue.depthStencil = {1.0f, 0};
    if (useMsaa) {
        depth.resolveImageView   = rt.depth.view();
        depth.resolveMode        = VK_RESOLVE_MODE_MIN_BIT;
        depth.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    }

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

    if (m_useMeshShader && m_meshPipeline != VK_NULL_HANDLE) {
        // ── Mesh Shader 路径：无需 vertex/index buffer bind ──
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_meshPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_meshPipelineLayout, 0, 1, &m_meshSet, 0, nullptr);
        // Task Shader 可用时按 64-thread group 分配；无 Task 时每 draw 一个 mesh group
        // 当前 pipeline 不含 Task Shader，始终用 workgroup 映射表
        uint32_t groups = m_meshGroupCount;
        m_device->vkCmdDrawMeshTasksEXT(cmd, groups, 1, 1);
    } else {
        // ── VS 路径：传统 vertex/index buffer ──
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout, 0, 1, &m_set, 0, nullptr);

        VkDeviceSize zero = 0;
        VkBuffer vb = gpu.vertexBuffer.handle();
        vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
        vkCmdBindIndexBuffer(cmd, gpu.indexBuffer.handle(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexedIndirectCount(cmd, indirectBuf, 0, indirectBuf, 0, drawCount, sizeof(VkDrawIndexedIndirectCommand));
    }

    vkCmdEndRendering(cmd);
}

}
