// RsmGeometryPass 实现 —— 仿 GBufferPass 的模板。
// 主要差别：
//   - 渲染目标固定 512²，不跟 swapchain 走。
//   - 3 个 color attachment 而非 GBuffer 的 3 个不同语义；depth 同。
//   - viewProj 来自 sun 视角的 ortho 矩阵（updateLight 在 CPU 端算）。
//   - shader 用 gi/rsm/rsm_geometry.spv。

#include "rsm_geometry_pass.h"
#include "core/device.h"
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <cstring>
#include <limits>

namespace somegi {

namespace {
// 与 shaders/common/shared_types.slang 中 RsmFrameUniforms 严格对齐。
struct RsmFrameUbo {
    glm::mat4 sunViewProj;
    glm::mat4 sunView;
    glm::vec4 sunDir;
    glm::vec4 sunColor_intensity;
};

// barrier helper —— 与 ibl_baker / app.cpp 中的 transitionImage 类似。
void transitionImage2(VkCommandBuffer cmd, VkImage img,
                      VkImageAspectFlags aspect,
                      VkImageLayout oldL, VkImageLayout newL,
                      VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                      VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
    b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
    b.oldLayout = oldL; b.newLayout = newL;
    b.image = img;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}
}

void RsmGeometryPass::init(Device& d, uint32_t maxTextures) {
    m_device = &d;
    m_maxTextures = maxTextures;

    // 1. 分配 4 张 RT。固定 512²，与 swapchain 解耦。
    auto mkColor = [&](VkFormat fmt, Image& img) {
        ImageDesc id{};
        id.format = fmt;
        id.extent = {kRsmSize, kRsmSize, 1};
        id.usage  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                  | VK_IMAGE_USAGE_SAMPLED_BIT;
        img = Image(d, id);
    };
    mkColor(VK_FORMAT_R16G16B16A16_SFLOAT, m_position);
    mkColor(VK_FORMAT_R16G16B16A16_SFLOAT, m_normal);
    mkColor(VK_FORMAT_R16G16B16A16_SFLOAT, m_flux);
    {
        ImageDesc id{};
        id.format = VK_FORMAT_D32_SFLOAT;
        id.extent = {kRsmSize, kRsmSize, 1};
        id.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        id.usage  = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                  | VK_IMAGE_USAGE_SAMPLED_BIT;
        m_depth = Image(d, id);
    }

    // 2. RsmFrameUbo（host-coherent，updateLight 直接 memcpy 写）。
    m_rsmFrameUbo = Buffer(d, sizeof(RsmFrameUbo),
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // 3. set=0 layout —— 与 GBufferPass 同：UBO + materials SSBO + sampler
    //    + 贴图数组。RsmFrameUbo 占 binding 0（GBufferPass 那是 FrameUbo，
    //    本 pass 是 RsmFrameUbo，shader 端通过 import 引用名字一致）。
    std::array<VkDescriptorSetLayoutBinding, 4> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLER,        1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  maxTextures, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

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

    // 把 RsmFrameUbo 写一次 binding 0；scene 相关的 binding 1/2/3 由
    // bindScene 写。
    VkDescriptorBufferInfo uboInfo{m_rsmFrameUbo.handle(), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w0{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w0.dstSet = m_set; w0.dstBinding = 0; w0.descriptorCount = 1;
    w0.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w0.pBufferInfo = &uboInfo;
    vkUpdateDescriptorSets(d.device(), 1, &w0, 0, nullptr);

    buildPipeline();
}

void RsmGeometryPass::buildPipeline() {
    auto& d = *m_device;

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pc.size = 64 + 16;   // mat4 model + materialIndex + 3 ints

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    ShaderModule shader(d, shaderDir() / "gi" / "rsm" / "rsm_geometry.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = shader.handle(); stages[0].pName = "vs_main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = shader.handle(); stages[1].pName = "ps_main";

    VkVertexInputBindingDescription vib{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    // RSM VS 只读 pos/normal/uv0；tangent (location=2) 不消费，不在 pipeline
    // 声明里列出可消除"attribute not consumed"性能 warning。注意 location
    // 编号要与 shader 对应：UV 在 shader 端是 TEXCOORD0 → location=3。
    std::array<VkVertexInputAttributeDescription, 3> via{};
    via[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex,position)};
    via[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex,normal)};
    via[2] = {3, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex,uv0)};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = (uint32_t)via.size(); vi.pVertexAttributeDescriptions = via.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    // sun ortho 视角下，front face 朝向反过来跟主 camera 一样还是 CCW。
    // 但因为光从背面照不到（NdotL=0 被 flux=0 抹掉），可以保持 BACK cull。
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    // 3 个 color attach（位置 / 法线 / flux），每个标准 RGBA write，无 blend。
    std::array<VkPipelineColorBlendAttachmentState, 3> ba{};
    for (auto& a : ba) a.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = (uint32_t)ba.size(); cb.pAttachments = ba.data();

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyni{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyni.dynamicStateCount = 2; dyni.pDynamicStates = dyn;

    std::array<VkFormat, 3> colorFmts{
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT,
    };
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = (uint32_t)colorFmts.size();
    rci.pColorAttachmentFormats = colorFmts.data();
    rci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

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

void RsmGeometryPass::destroyPipeline() {
    if (!m_device) return;
    if (m_pipeline)       vkDestroyPipeline(m_device->device(), m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(m_device->device(), m_pipelineLayout, nullptr);
    m_pipeline = VK_NULL_HANDLE; m_pipelineLayout = VK_NULL_HANDLE;
}

void RsmGeometryPass::destroy() {
    if (!m_device) return;
    destroyPipeline();
    auto dev = m_device->device();
    if (m_pool)      vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout) vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    m_pool = VK_NULL_HANDLE; m_setLayout = VK_NULL_HANDLE;
    m_rsmFrameUbo.reset();
    m_position.reset();
    m_normal.reset();
    m_flux.reset();
    m_depth.reset();
    m_device = nullptr;
}

void RsmGeometryPass::bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount) {
    // 与 GBufferPass.bindScene 同：写 binding 1 (materials) / 2 (sampler) /
    // 3 (textures)。binding 0 = RsmFrameUbo 在 init 里已经写过且不变。
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

    std::array<VkWriteDescriptorSet, 3> w{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_set; w[0].dstBinding = 1; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &matInfo;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[1].dstSet = m_set; w[1].dstBinding = 2; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[1].pImageInfo = &samplerInfo;
    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[2].dstSet = m_set; w[2].dstBinding = 3; w[2].descriptorCount = m_maxTextures;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[2].pImageInfo = imgs.data();

    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void RsmGeometryPass::updateLight(const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                                  const glm::vec3& sunDir,
                                  const glm::vec3& sunColor, float sunIntensity) {
    // 1. 约定：sunDir = 光传播方向（从 sun 指向 surface，与主 FrameUBO 一致）。
    //    sun camera 应放在场景外、朝光传播方向"上游"的位置。
    glm::vec3 lightDir = glm::normalize(sunDir);   // sun → surface
    glm::vec3 toSun    = -lightDir;                // surface → sun
    glm::vec3 sceneCenter = (aabbMin + aabbMax) * 0.5f;
    glm::vec3 sceneSize   = aabbMax - aabbMin;
    float diag = glm::length(sceneSize);

    // 2. 把"sun camera"放到 scene 朝太阳方向之外足够远的位置，让 ortho
    //    frustum 能覆盖整个 scene。
    glm::vec3 sunPos = sceneCenter + toSun * diag;

    // 3. lookAt：从 sunPos 看向 sceneCenter（视线方向 = lightDir）。
    //    up 选 (0,1,0)，但当 toSun 接近 ±Y 时退化用 (1,0,0) 防 cross 0。
    glm::vec3 up = (std::abs(toSun.y) < 0.999f) ? glm::vec3(0,1,0) : glm::vec3(1,0,0);
    glm::mat4 view = glm::lookAt(sunPos, sceneCenter, up);

    // 4. 把 8 个 AABB 角点投到 view 空间，得到 sun-space 包围盒，再加少
    //    许 padding 防边界裁剪。
    glm::vec3 mn{ std::numeric_limits<float>::max()};
    glm::vec3 mx{-std::numeric_limits<float>::max()};
    for (int i = 0; i < 8; ++i) {
        glm::vec3 corner(
            (i & 1) ? aabbMax.x : aabbMin.x,
            (i & 2) ? aabbMax.y : aabbMin.y,
            (i & 4) ? aabbMax.z : aabbMin.z);
        glm::vec3 v = glm::vec3(view * glm::vec4(corner, 1.0f));
        mn = glm::min(mn, v);
        mx = glm::max(mx, v);
    }
    float padding = diag * 0.05f;
    mn -= glm::vec3(padding);
    mx += glm::vec3(padding);

    // 5. ortho 投影。注意 view space z：相机看向 -Z，所以远 = -mx.z，近 = -mn.z。
    glm::mat4 proj = glm::ortho(mn.x, mx.x, mn.y, mx.y, -mx.z, -mn.z);
    proj[1][1] *= -1.0f;   // Vulkan Y-flip（与主相机一致）

    glm::mat4 viewProj = proj * view;

    RsmFrameUbo u{};
    u.sunViewProj = viewProj;
    u.sunView     = view;
    u.sunDir      = glm::vec4(lightDir, 0.0f);   // 与主 FrameUBO 同：光传播方向
    u.sunColor_intensity = glm::vec4(sunColor, sunIntensity);
    std::memcpy(m_rsmFrameUbo.mapped(), &u, sizeof(u));
}

void RsmGeometryPass::record(VkCommandBuffer cmd, const SceneCpu& cpu, const SceneGpu& gpu) {
    // 1. layout 转换：4 张 RT 都到合适的 attachment layout。
    //    第一帧 / 切场景后是 UNDEFINED；后续帧来自上一次结尾的 SHADER_READ_ONLY，
    //    用 UNDEFINED 起始（discard 上一帧内容）就够 —— 因为本 pass 整张
    //    覆盖写。
    transitionImage2(cmd, m_position.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    transitionImage2(cmd, m_normal.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    transitionImage2(cmd, m_flux.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    transitionImage2(cmd, m_depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    // 2. dynamic rendering 配置 3 color + 1 depth。
    std::array<VkRenderingAttachmentInfo, 3> color{};
    auto setColor = [](VkRenderingAttachmentInfo& a, VkImageView v) {
        a = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        a.imageView   = v;
        a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        a.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        a.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        a.clearValue.color = {{0, 0, 0, 0}};
    };
    setColor(color[0], m_position.view());
    setColor(color[1], m_normal.view());
    setColor(color[2], m_flux.view());

    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageView   = m_depth.view();
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    depth.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {kRsmSize, kRsmSize}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = (uint32_t)color.size();
    ri.pColorAttachments    = color.data();
    ri.pDepthAttachment     = &depth;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{0, 0, (float)kRsmSize, (float)kRsmSize, 0, 1};
    VkRect2D sc{{0, 0}, {kRsmSize, kRsmSize}};
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

    // 3. 4 张 RT 转 SHADER_READ_ONLY，方便后续 RsmSamplePass 立即采样。
    transitionImage2(cmd, m_position.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    transitionImage2(cmd, m_normal.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    transitionImage2(cmd, m_flux.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    transitionImage2(cmd, m_depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

}
