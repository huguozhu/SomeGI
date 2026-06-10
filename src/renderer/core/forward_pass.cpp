#include "renderer/core/forward_pass.h"
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

void ForwardPass::init(Device& d, VkFormat colorFmt, VkFormat depthFmt, uint32_t maxTextures) {
    m_device = &d;
    m_colorFmt = colorFmt;
    m_depthFmt = depthFmt;
    m_maxTextures = maxTextures;

    // === Set=0 layout ===
    std::array<VkDescriptorSetLayoutBinding, 11> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,   maxTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    // NDGI MLP 权重 (bindings 4-9)
    for (uint32_t i = 4; i < 10; ++i)
        b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    // GPU-driven DrawData (binding 10)
    b[10] = {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};

    std::array<VkDescriptorBindingFlags, 11> bf{};
    bf[3] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bfci.bindingCount = (uint32_t)bf.size(); bfci.pBindingFlags = bf.data();

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.pNext = &bfci;
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    // === Descriptor pool + Set=0 alloc ===
    std::array<VkDescriptorPoolSize, 4> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8},  // +6 NDGI
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

    // ── Mesh Shader set=0 descriptor layout ──
    {
        std::array<VkDescriptorSetLayoutBinding, 12> mb{};
        mb[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                 VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr};
        mb[1] = {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_TASK_BIT_EXT, nullptr};
        for (uint32_t i = 0; i < 4; ++i)
            mb[2+i] = {2+i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_TASK_BIT_EXT, nullptr};
        mb[6] = {6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                 VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        mb[7] = {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr};
        mb[8] = {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr};
        mb[9] = {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        mb[10]= {10, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        mb[11]= {11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, m_maxTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = (uint32_t)mb.size(); li.pBindings = mb.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_meshSetLayout));

        std::array<VkDescriptorPoolSize, 4> mps{{
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
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
}

void ForwardPass::buildPipeline() {
    auto& d = *m_device;

    // set=0（帧/材质/纹理）+ set=1（IBL cube 贴图）
    std::array<VkDescriptorSetLayout, 2> sets{m_setLayout, m_iblDsl};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 2;
    plci.pSetLayouts = sets.data();
    plci.pushConstantRangeCount = 0; plci.pPushConstantRanges = nullptr;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    auto sd = shaderDir();
    ShaderModule shader(d, sd / "forward" / "forward_ibl.spv");

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

    VkPipelineColorBlendAttachmentState ba{}; ba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &ba;

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyni{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyni.dynamicStateCount = 2; dyni.pDynamicStates = dyn;

    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1; rci.pColorAttachmentFormats = &m_colorFmt;
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

void ForwardPass::destroyPipeline() {
    if (!m_device) return;
    if (m_pipeline)       vkDestroyPipeline(m_device->device(), m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(m_device->device(), m_pipelineLayout, nullptr);
    if (m_meshPipeline)   vkDestroyPipeline(m_device->device(), m_meshPipeline, nullptr);
    if (m_meshPipelineLayout) vkDestroyPipelineLayout(m_device->device(), m_meshPipelineLayout, nullptr);
    m_pipeline = VK_NULL_HANDLE; m_pipelineLayout = VK_NULL_HANDLE;
    m_meshPipeline = VK_NULL_HANDLE; m_meshPipelineLayout = VK_NULL_HANDLE;
}

void ForwardPass::buildMeshPipeline() {
    auto& d = *m_device;

    std::array<VkDescriptorSetLayout, 2> sets{m_meshSetLayout, m_iblDsl};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 2; plci.pSetLayouts = sets.data();
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_meshPipelineLayout));

    auto sd = shaderDir();
    ShaderModule taskMod(d, sd / "forward" / "forward_mesh.spv");
    ShaderModule meshMod(d, sd / "forward" / "forward_mesh.spv");
    ShaderModule fragMod(d, sd / "forward" / "forward_mesh.spv");

    VkPipelineShaderStageCreateInfo stages[3]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_TASK_BIT_EXT; stages[0].module = taskMod.handle(); stages[0].pName = "ts_main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_MESH_BIT_EXT; stages[1].module = meshMod.handle(); stages[1].pName = "ms_main";
    stages[2] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[2].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[2].module = fragMod.handle(); stages[2].pName = "ps_main";

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 0; vi.vertexAttributeDescriptionCount = 0;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_BACK_BIT; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendAttachmentState ba{}; ba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &ba;
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyni{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyni.dynamicStateCount = 2; dyni.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1; rci.pColorAttachmentFormats = &m_colorFmt;
    rci.depthAttachmentFormat = m_depthFmt;
    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.pNext = &rci; gpci.stageCount = 3; gpci.pStages = stages;
    gpci.pVertexInputState = &vi; gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp; gpci.pRasterizationState = &rs; gpci.pMultisampleState = &ms;
    gpci.pDepthStencilState = &ds; gpci.pColorBlendState = &cb; gpci.pDynamicState = &dyni;
    gpci.layout = m_meshPipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(d.device(), VK_NULL_HANDLE, 1, &gpci, nullptr, &m_meshPipeline));
}

void ForwardPass::bindHiZViews(VkImageView mip1, VkImageView mip2, VkImageView mip3, VkImageView mip4) {
    auto hz = [](VkImageView v) { VkDescriptorImageInfo i{}; i.imageView=v; i.imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; return i; };
    VkDescriptorImageInfo h[4] = {hz(mip1),hz(mip2),hz(mip3),hz(mip4)};
    std::array<VkWriteDescriptorSet, 4> w{};
    for (uint32_t i=0;i<4;++i) { w[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[i].dstSet=m_meshSet; w[i].dstBinding=2+i; w[i].descriptorCount=1; w[i].descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[i].pImageInfo=&h[i]; }
    vkUpdateDescriptorSets(m_device->device(),4,w.data(),0,nullptr);
}

void ForwardPass::updateCullUbo(const glm::mat4& viewProj, const glm::vec4 frustum[6],
                                 uint32_t drawCount, uint32_t hizMaxMip,
                                 uint32_t screenW, uint32_t screenH) {
    struct { glm::mat4 viewProj; glm::vec4 frustum[6]; glm::vec2 screenSize; uint32_t drawCount; uint32_t hizMaxMip; } cull{};
    cull.viewProj = viewProj;
    for (int i=0;i<6;++i) cull.frustum[i]=frustum[i];
    cull.screenSize = glm::vec2((float)screenW,(float)screenH);
    cull.drawCount = drawCount; cull.hizMaxMip = hizMaxMip;
    std::memcpy(m_cullUbo.mapped(),&cull,sizeof(cull));
}

void ForwardPass::bindIblResources(Device& d, const IblResources& ibl) {
    // 创建 IBL set=1 描述符布局
    constexpr VkShaderStageFlags kStages =
        VK_SHADER_STAGE_FRAGMENT_BIT;
    std::array<VkDescriptorSetLayoutBinding, 5> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, kStages};  // diffuse cube
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, kStages};  // specular cube
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, kStages};  // brdf lut
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLER,        1, kStages};  // linear sampler
    b[4] = {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, kStages};  // params (intensity)

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_iblDsl));

    // 分配 IBL set=1 descriptor pool + set
    std::array<VkDescriptorPoolSize, 3> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_iblPool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_iblPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_iblDsl;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_iblSet));

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
    // Dummy intensity UBO（前向路径可通过 FrameUBO 传 intensity）
    Buffer dummyUbo(d, 16,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    float one = 1.0f;
    std::memcpy(dummyUbo.mapped(), &one, sizeof(float));
    VkDescriptorBufferInfo uboI{dummyUbo.handle(), 0, VK_WHOLE_SIZE};

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

    // 创建 IBL set=1 后构建 pipeline
    buildPipeline();
    buildMeshPipeline();
}

void ForwardPass::destroy() {
    if (!m_device) return;
    destroyPipeline();
    auto dev = m_device->device();
    if (m_pool)      vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout) vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    if (m_iblPool)   vkDestroyDescriptorPool(dev, m_iblPool, nullptr);
    if (m_iblDsl)    vkDestroyDescriptorSetLayout(dev, m_iblDsl, nullptr);
    if (m_meshPool)  vkDestroyDescriptorPool(dev, m_meshPool, nullptr);
    if (m_meshSetLayout) vkDestroyDescriptorSetLayout(dev, m_meshSetLayout, nullptr);
    m_pool = VK_NULL_HANDLE; m_setLayout = VK_NULL_HANDLE;
    m_iblPool = VK_NULL_HANDLE; m_iblDsl = VK_NULL_HANDLE;
    m_meshPool = VK_NULL_HANDLE; m_meshSetLayout = VK_NULL_HANDLE;
    m_frameUbo.reset();
    m_cullUbo.reset();
    m_device = nullptr;
}

void ForwardPass::bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount) {
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

    // ── 同时写入 Mesh Shader set=0 描述符 ──
    VkDescriptorBufferInfo vbInfo{gpu.vertexBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo ibInfo{gpu.indexBuffer.handle(), 0, VK_WHOLE_SIZE};
    std::array<VkWriteDescriptorSet, 6> mw{};
    mw[0]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};mw[0].dstSet=m_meshSet;mw[0].dstBinding=6;mw[0].descriptorCount=1;
    mw[0].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;mw[0].pBufferInfo=&uboInfo;
    mw[1]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};mw[1].dstSet=m_meshSet;mw[1].dstBinding=7;mw[1].descriptorCount=1;
    mw[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;mw[1].pBufferInfo=&vbInfo;
    mw[2]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};mw[2].dstSet=m_meshSet;mw[2].dstBinding=8;mw[2].descriptorCount=1;
    mw[2].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;mw[2].pBufferInfo=&ibInfo;
    mw[3]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};mw[3].dstSet=m_meshSet;mw[3].dstBinding=9;mw[3].descriptorCount=1;
    mw[3].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;mw[3].pBufferInfo=&matInfo;
    mw[4]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};mw[4].dstSet=m_meshSet;mw[4].dstBinding=10;mw[4].descriptorCount=1;
    mw[4].descriptorType=VK_DESCRIPTOR_TYPE_SAMPLER;mw[4].pImageInfo=&samplerInfo;
    mw[5]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};mw[5].dstSet=m_meshSet;mw[5].dstBinding=11;mw[5].descriptorCount=m_maxTextures;
    mw[5].descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;mw[5].pImageInfo=imgs.data();
    vkUpdateDescriptorSets(d.device(),(uint32_t)mw.size(),mw.data(),0,nullptr);
}

void ForwardPass::bindDrawData(Device& d, VkBuffer drawDataBuf) {
    VkDescriptorBufferInfo dd{drawDataBuf,0,VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet=m_set;w.dstBinding=10;w.descriptorCount=1;
    w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&dd;
    vkUpdateDescriptorSets(d.device(),1,&w,0,nullptr);
    // Mesh 路径 binding 0
    VkWriteDescriptorSet mw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    mw.dstSet=m_meshSet;mw.dstBinding=0;mw.descriptorCount=1;
    mw.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;mw.pBufferInfo=&dd;
    vkUpdateDescriptorSets(d.device(),1,&mw,0,nullptr);
}
void ForwardPass::updateFrame(const FrameUBO& ubo) {
    std::memcpy(m_frameUbo.mapped(), &ubo, sizeof(FrameUBO));
}

void ForwardPass::record(VkCommandBuffer cmd, const RenderTargets& rt,
                         VkBuffer indirectBuf, uint32_t drawCount, const SceneGpu& gpu) {
    if (drawCount == 0) return;
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = rt.hdrColor.view();
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{0.02f, 0.02f, 0.04f, 1.0f}};

    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageView = rt.depth.view();
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0,0}, rt.extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1; ri.pColorAttachments = &color;
    ri.pDepthAttachment = &depth;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{0, 0, (float)rt.extent.width, (float)rt.extent.height, 0, 1};
    VkRect2D sc{{0,0}, rt.extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    if (m_useMeshShader && m_meshPipeline != VK_NULL_HANDLE) {
        // ── Mesh Shader 路径 ──
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_meshPipeline);
        VkDescriptorSet msSets[2] = {m_meshSet, m_iblSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_meshPipelineLayout, 0, 2, msSets, 0, nullptr);
        uint32_t taskGroups = (drawCount + 63) / 64;
        m_device->vkCmdDrawMeshTasksEXT(cmd, taskGroups, 1, 1);
    } else {
        // ── VS 路径 ──
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
        VkDescriptorSet sets[2] = {m_set, m_iblSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout, 0, 2, sets, 0, nullptr);
        VkDeviceSize zero = 0;
        VkBuffer vb = gpu.vertexBuffer.handle();
        vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
        vkCmdBindIndexBuffer(cmd, gpu.indexBuffer.handle(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexedIndirectCount(cmd, indirectBuf, 0, indirectBuf, 0, drawCount, sizeof(VkDrawIndexedIndirectCommand));
    }

    vkCmdEndRendering(cmd);
}

void ForwardPass::setNdgiWeights(Device& d,
    VkBuffer w1, VkBuffer b1, VkBuffer w2, VkBuffer b2,
    VkBuffer w3, VkBuffer b3) {
    if (m_set == VK_NULL_HANDLE) return;
    VkDescriptorBufferInfo infos[6]{};
    infos[0] = {w1, 0, VK_WHOLE_SIZE}; infos[1] = {b1, 0, VK_WHOLE_SIZE};
    infos[2] = {w2, 0, VK_WHOLE_SIZE}; infos[3] = {b2, 0, VK_WHOLE_SIZE};
    infos[4] = {w3, 0, VK_WHOLE_SIZE}; infos[5] = {b3, 0, VK_WHOLE_SIZE};
    std::array<VkWriteDescriptorSet, 6> iw{};
    for (uint32_t i = 0; i < 6; ++i) {
        iw[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        iw[i].dstSet = m_set; iw[i].dstBinding = 4 + i; iw[i].descriptorCount = 1;
        iw[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; iw[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(d.device(), (uint32_t)iw.size(), iw.data(), 0, nullptr);
}

}
