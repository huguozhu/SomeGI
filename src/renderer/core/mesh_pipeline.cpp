#include "renderer/core/mesh_pipeline.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>
#include <stdexcept>

namespace somegi {

MeshPipeline createMeshPipeline(Device& d, const MeshPipelineDesc& desc) {
    // ── Shader stages ──────────────────────────────────────────
    std::vector<VkPipelineShaderStageCreateInfo> stages;

    // Task Shader（可选）
    ShaderModule taskModule;
    if (!desc.taskSpv.empty()) {
        taskModule = ShaderModule(d, desc.taskSpv);
        VkPipelineShaderStageCreateInfo taskStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        taskStage.stage  = VK_SHADER_STAGE_TASK_BIT_EXT;
        taskStage.module = taskModule.handle();
        taskStage.pName  = desc.taskEntry;
        stages.push_back(taskStage);
    }

    // Mesh Shader（必需）
    ShaderModule meshModule(d, desc.meshSpv);
    VkPipelineShaderStageCreateInfo meshStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    meshStage.stage  = VK_SHADER_STAGE_MESH_BIT_EXT;
    meshStage.module = meshModule.handle();
    meshStage.pName  = desc.meshEntry;
    stages.push_back(meshStage);

    // Fragment Shader（必需）
    ShaderModule fragModule(d, desc.fragSpv);
    VkPipelineShaderStageCreateInfo fragStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule.handle();
    fragStage.pName  = desc.fragEntry;
    stages.push_back(fragStage);

    // ── Pipeline layout ────────────────────────────────────────
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = (uint32_t)desc.setLayouts.size();
    plci.pSetLayouts    = desc.setLayouts.data();
    plci.pushConstantRangeCount = 0;
    VkPipelineLayout pipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &pipelineLayout));

    // ── Fixed-function state ───────────────────────────────────
    // Mesh Shader 不读 vertex input → 设 0
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount   = 0;
    vi.vertexAttributeDescriptionCount = 0;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode  = desc.cullMode;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = desc.samples;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = desc.depthTest  ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = desc.depthWrite ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    // ── Color blend ────────────────────────────────────────────
    std::vector<VkPipelineColorBlendAttachmentState> blendAttachments;
    for (size_t i = 0; i < desc.colorFormats.size(); ++i) {
        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = 0xF;
        blendAttachments.push_back(ba);
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = (uint32_t)blendAttachments.size();
    cb.pAttachments    = blendAttachments.data();

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyni{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyni.dynamicStateCount = 2; dyni.pDynamicStates = dyn;

    // ── Dynamic rendering ──────────────────────────────────────
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount    = (uint32_t)desc.colorFormats.size();
    rci.pColorAttachmentFormats = desc.colorFormats.data();
    rci.depthAttachmentFormat   = desc.depthFormat;

    // ── Create pipeline ────────────────────────────────────────
    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.pNext               = &rci;
    gpci.stageCount          = (uint32_t)stages.size();
    gpci.pStages             = stages.data();
    gpci.pVertexInputState   = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState      = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState   = &ms;
    gpci.pDepthStencilState  = &ds;
    gpci.pColorBlendState    = &cb;
    gpci.pDynamicState       = &dyni;
    gpci.layout              = pipelineLayout;

    VkPipeline pipeline;
    VK_CHECK(vkCreateGraphicsPipelines(d.device(), VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline));

    return {pipeline, pipelineLayout};
}

} // namespace somegi
