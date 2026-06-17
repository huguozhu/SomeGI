// rhi/vulkan/vk_pso.cpp
#include "vk_pso.h"
#include "vk_shader.h"
#include "vk_descriptor.h"
#include <cstring>

namespace somegi {
namespace rhi {

static VkFormat toVkFormatRHI(Format f) {
    switch (f) {
        case Format::R8_UNORM:          return VK_FORMAT_R8_UNORM;
        case Format::R8G8B8A8_UNORM:    return VK_FORMAT_R8G8B8A8_UNORM;
        case Format::R16G16B16A16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case Format::R32_UINT:          return VK_FORMAT_R32_UINT;
        case Format::R32_SFLOAT:        return VK_FORMAT_R32_SFLOAT;
        case Format::D32_SFLOAT:        return VK_FORMAT_D32_SFLOAT;
        case Format::B8G8R8A8_UNORM:    return VK_FORMAT_B8G8R8A8_UNORM;
        case Format::B8G8R8A8_SRGB:     return VK_FORMAT_B8G8R8A8_SRGB;
        default: return VK_FORMAT_R8G8B8A8_UNORM;
    }
}

static VkShaderStageFlags toVkShaderStage(ShaderStage s) {
    VkShaderStageFlags f = 0;
    if ((uint32_t)s & (uint32_t)ShaderStage::Vertex)   f |= VK_SHADER_STAGE_VERTEX_BIT;
    if ((uint32_t)s & (uint32_t)ShaderStage::Fragment) f |= VK_SHADER_STAGE_FRAGMENT_BIT;
    if ((uint32_t)s & (uint32_t)ShaderStage::Compute)  f |= VK_SHADER_STAGE_COMPUTE_BIT;
    if ((uint32_t)s & (uint32_t)ShaderStage::Mesh)     f |= VK_SHADER_STAGE_MESH_BIT_EXT;
    if ((uint32_t)s & (uint32_t)ShaderStage::Task)     f |= VK_SHADER_STAGE_TASK_BIT_EXT;
    return f;
}

// ════════════════════════════════════════════════════════════════
// Graphics PSO
// ════════════════════════════════════════════════════════════════
std::unique_ptr<RHIPipelineState> VkRHIPipelineState::createGraphics(VkRHIDevice& device, const GraphicsPSODesc& desc) {
    auto pso = std::unique_ptr<VkRHIPipelineState>(new VkRHIPipelineState(device));
    VkDevice vkd = device.vkDevice();

    // Shader stages
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    auto addStage = [&](const RHIShader* shader, VkShaderStageFlagBits stage) {
        if (!shader) return;
        VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        si.stage = stage;
        si.module = (VkShaderModule)(uintptr_t)shader->nativeHandle();
        si.pName = shader->entryPoint();  // 使用 ShaderDesc 中指定的入口函数名
        stages.push_back(si);
    };
    addStage(desc.vertexShader, VK_SHADER_STAGE_VERTEX_BIT);
    addStage(desc.fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT);
    addStage(desc.meshShader, VK_SHADER_STAGE_MESH_BIT_EXT);
    addStage(desc.taskShader, VK_SHADER_STAGE_TASK_BIT_EXT);

    // Vertex input
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attrs;
    for (auto& b : desc.vertexInput.bindings)
        bindings.push_back({b.binding, b.stride, b.perInstance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
    for (auto& a : desc.vertexInput.attributes)
        attrs.push_back({a.location, a.binding, VK_FORMAT_R32G32B32_SFLOAT, a.offset}); // simplified
    vi.vertexBindingDescriptionCount = (uint32_t)bindings.size();
    vi.pVertexBindingDescriptions = bindings.data();
    vi.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
    vi.pVertexAttributeDescriptions = attrs.data();

    // Input assembly
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = (desc.topology == PrimitiveTopology::TriangleList) ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

    // Rasterization
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = (desc.rasterization.fill == FillMode::Solid) ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_LINE;
    rs.cullMode = (desc.rasterization.cull == CullMode::Back) ? VK_CULL_MODE_BACK_BIT : (desc.rasterization.cull == CullMode::None ? VK_CULL_MODE_NONE : VK_CULL_MODE_FRONT_BIT);
    rs.frontFace = desc.rasterization.frontCCW ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;

    // Depth stencil
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = desc.depthStencil.depthTest;
    ds.depthWriteEnable = desc.depthStencil.depthWrite;
    switch (desc.depthStencil.depthCompare) {
        case CompareFunc::Never:        ds.depthCompareOp = VK_COMPARE_OP_NEVER; break;
        case CompareFunc::Less:         ds.depthCompareOp = VK_COMPARE_OP_LESS; break;
        case CompareFunc::Equal:        ds.depthCompareOp = VK_COMPARE_OP_EQUAL; break;
        case CompareFunc::LessEqual:    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL; break;
        case CompareFunc::Greater:      ds.depthCompareOp = VK_COMPARE_OP_GREATER; break;
        case CompareFunc::NotEqual:     ds.depthCompareOp = VK_COMPARE_OP_NOT_EQUAL; break;
        case CompareFunc::GreaterEqual: ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; break;
        case CompareFunc::Always:       ds.depthCompareOp = VK_COMPARE_OP_ALWAYS; break;
    }

    // Blend
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    std::vector<VkPipelineColorBlendAttachmentState> blendAtts;
    for (auto& a : desc.blend.attachments) {
        VkPipelineColorBlendAttachmentState ba{};
        ba.blendEnable = a.blendEnable;
        ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        ba.colorBlendOp = VK_BLEND_OP_ADD;
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blendAtts.push_back(ba);
    }
    if (blendAtts.empty()) {
        blendAtts.resize(desc.renderTargets.colorFormats.empty() ? 1 : desc.renderTargets.colorFormats.size());
        // 默认：关闭混合、写入所有通道（与 Vulkan 默认行为一致）
        for (auto& ba : blendAtts)
            ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }
    cb.attachmentCount = (uint32_t)blendAtts.size();
    cb.pAttachments = blendAtts.data();

    // Dynamic rendering
    std::vector<VkFormat> colorFmts;
    for (auto& f : desc.renderTargets.colorFormats) colorFmts.push_back(toVkFormatRHI(f));
    VkPipelineRenderingCreateInfo dr{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    dr.colorAttachmentCount = (uint32_t)colorFmts.size();
    dr.pColorAttachmentFormats = colorFmts.data();
    dr.depthAttachmentFormat = desc.renderTargets.depthFormat != Format::Unknown ? toVkFormatRHI(desc.renderTargets.depthFormat) : VK_FORMAT_UNDEFINED;

    // Multisample
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = (VkSampleCountFlagBits)desc.renderTargets.sampleCount;

    // Viewport (dynamic)
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    // Pipeline layout
    std::vector<VkDescriptorSetLayout> setLayouts;
    for (auto* l : desc.descriptorSetLayouts) {
        setLayouts.push_back((VkDescriptorSetLayout)(uintptr_t)l->nativeHandle());
    }
    std::vector<VkPushConstantRange> pcRanges;
    for (auto& pc : desc.pushConstants) {
        pcRanges.push_back({toVkShaderStage(pc.stages), pc.offset, pc.size});
    }
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = (uint32_t)setLayouts.size();
    pl.pSetLayouts = setLayouts.data();
    pl.pushConstantRangeCount = (uint32_t)pcRanges.size();
    pl.pPushConstantRanges = pcRanges.data();
    vkCreatePipelineLayout(vkd, &pl, nullptr, &pso->m_pipelineLayout);

    // Create pipeline
    VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gci.stageCount = (uint32_t)stages.size();
    gci.pStages = stages.data();
    gci.pVertexInputState = &vi;
    gci.pInputAssemblyState = &ia;
    gci.pRasterizationState = &rs;
    gci.pDepthStencilState = &ds;
    gci.pColorBlendState = &cb;
    gci.pMultisampleState = &ms;
    gci.pViewportState = &vp;
    gci.pDynamicState = &dyn;
    gci.layout = pso->m_pipelineLayout;
    gci.pNext = &dr;
    gci.renderPass = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(vkd, VK_NULL_HANDLE, 1, &gci, nullptr, &pso->m_pipeline);
    pso->m_bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

    return pso;
}

// ════════════════════════════════════════════════════════════════
// Compute PSO
// ════════════════════════════════════════════════════════════════
std::unique_ptr<RHIPipelineState> VkRHIPipelineState::createCompute(VkRHIDevice& device, const ComputePSODesc& desc) {
    auto pso = std::unique_ptr<VkRHIPipelineState>(new VkRHIPipelineState(device));
    VkDevice vkd = device.vkDevice();

    // Pipeline layout
    std::vector<VkDescriptorSetLayout> setLayouts;
    for (auto* l : desc.descriptorSetLayouts)
        setLayouts.push_back((VkDescriptorSetLayout)(uintptr_t)l->nativeHandle());
    std::vector<VkPushConstantRange> pcRanges;
    for (auto& pc : desc.pushConstants)
        pcRanges.push_back({toVkShaderStage(pc.stages), pc.offset, pc.size});

    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = (uint32_t)setLayouts.size();
    pl.pSetLayouts = setLayouts.data();
    pl.pushConstantRangeCount = (uint32_t)pcRanges.size();
    pl.pPushConstantRanges = pcRanges.data();
    vkCreatePipelineLayout(vkd, &pl, nullptr, &pso->m_pipelineLayout);

    // Compute pipeline
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = (VkShaderModule)(uintptr_t)desc.computeShader->nativeHandle();
    ci.stage.pName = desc.computeShader->entryPoint();
    ci.layout = pso->m_pipelineLayout;
    vkCreateComputePipelines(vkd, VK_NULL_HANDLE, 1, &ci, nullptr, &pso->m_pipeline);
    pso->m_bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;

    return pso;
}

// ════════════════════════════════════════════════════════════════
// Ray Tracing PSO
// ════════════════════════════════════════════════════════════════
std::unique_ptr<RHIPipelineState> VkRHIPipelineState::createRayTracing(VkRHIDevice& device, const RayTracingPSODesc& desc) {
    auto pso = std::unique_ptr<VkRHIPipelineState>(new VkRHIPipelineState(device));
    VkDevice vkd = device.vkDevice();

    // 光线追踪 GPU 不支持 → 返回空 PSO
    if (!device.limits().rayTracingSupported) return pso;

    // ── Shader stages：将 RHI shader 数组转换为 Vulkan 管线阶段 ──
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    stages.reserve(desc.shaders.size());
    for (auto* shader : desc.shaders) {
        if (!shader) continue;
        VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        si.module = (VkShaderModule)(uintptr_t)shader->nativeHandle();
        si.pName = shader->entryPoint();
        switch (shader->stage()) {
            case ShaderStage::RayGen:        si.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR; break;
            case ShaderStage::RayMiss:       si.stage = VK_SHADER_STAGE_MISS_BIT_KHR; break;
            case ShaderStage::RayClosestHit: si.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR; break;
            case ShaderStage::RayAnyHit:     si.stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR; break;
            default: continue;  // 跳过非 RT 阶段（如顶点/计算着色器）
        }
        stages.push_back(si);
    }

    // ── Shader groups：构造 SBT 着色器组 ──
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;
    groups.reserve(desc.groups.size());
    for (auto& g : desc.groups) {
        VkRayTracingShaderGroupCreateInfoKHR gi{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        // 所有字段默认标记为"未使用"
        gi.generalShader = VK_SHADER_UNUSED_KHR;
        gi.closestHitShader = VK_SHADER_UNUSED_KHR;
        gi.anyHitShader = VK_SHADER_UNUSED_KHR;
        gi.intersectionShader = VK_SHADER_UNUSED_KHR;

        switch (g.type) {
            case ShaderGroupType::RayGen:
            case ShaderGroupType::Miss:
            case ShaderGroupType::Callable:
                gi.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
                gi.generalShader = g.generalShader;
                break;
            case ShaderGroupType::Hit:
                gi.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
                gi.closestHitShader = g.closestHitShader;
                if (g.anyHitShader != UINT32_MAX) gi.anyHitShader = g.anyHitShader;
                if (g.intersectionShader != UINT32_MAX) gi.intersectionShader = g.intersectionShader;
                break;
        }
        groups.push_back(gi);
    }

    // ── Pipeline layout ──
    std::vector<VkDescriptorSetLayout> setLayouts;
    for (auto* l : desc.descriptorSetLayouts)
        setLayouts.push_back((VkDescriptorSetLayout)(uintptr_t)l->nativeHandle());
    std::vector<VkPushConstantRange> pcRanges;
    for (auto& pc : desc.pushConstants)
        pcRanges.push_back({toVkShaderStage(pc.stages), pc.offset, pc.size});

    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = (uint32_t)setLayouts.size();
    pl.pSetLayouts = setLayouts.data();
    pl.pushConstantRangeCount = (uint32_t)pcRanges.size();
    pl.pPushConstantRanges = pcRanges.data();
    vkCreatePipelineLayout(vkd, &pl, nullptr, &pso->m_pipelineLayout);

    // ── 创建 RT 管线 ──
    VkRayTracingPipelineCreateInfoKHR ci{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ci.stageCount = (uint32_t)stages.size();
    ci.pStages = stages.data();
    ci.groupCount = (uint32_t)groups.size();
    ci.pGroups = groups.data();
    ci.maxPipelineRayRecursionDepth = desc.maxRecursionDepth;
    ci.layout = pso->m_pipelineLayout;

    // 使用 vkb dispatch 表调用（支持按需加载的扩展函数）
    device.dispatch().createRayTracingPipelinesKHR(
        VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &ci, nullptr, &pso->m_pipeline);
    pso->m_bindPoint = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;

    return pso;
}

VkRHIPipelineState::~VkRHIPipelineState() {
    if (m_pipeline) vkDestroyPipeline(m_device.vkDevice(), m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(m_device.vkDevice(), m_pipelineLayout, nullptr);
}

} // namespace rhi
} // namespace somegi
