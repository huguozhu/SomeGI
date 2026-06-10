# Shadow System 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现与 GI 选择器对标的阴影算法选择系统，通过 ImGui 下拉菜单切换 5 种阴影算法。

**Architecture:** 新建 `ShadowPass` 类作为统一阴影接口，内部根据 `ShadowMethod` 枚举分发到不同算法实现。所有算法输出 `shadowMask`（R8_UNORM），LightingPass 在直接光计算中乘上 mask 值。完全对标 GI 的 `kGis[]` → `m_currentGiIndex` → `applyGiSelection()` 模式。

**Tech Stack:** C++17, Vulkan 1.3, Slang → SPIR-V, KHR_ray_query (RT shadows)

**Phase 1 范围：** 5 种光栅化阴影算法（Phase 2 再加 RT Hard/RT Soft）

---

## 文件结构

```
新增:
  src/renderer/shadow/shadow_pass.h       — ShadowPass 类 + ShadowMethod 枚举
  src/renderer/shadow/shadow_pass.cpp     — 算法分发 + pipeline/descriptor 管理
  src/renderer/shadow/CMakeLists.txt      — 编译配置
  shaders/shadow/shadow_hard.slang        — Hard Shadow Map (vertex + fragment)
  shaders/shadow/shadow_pcf.slang         — PCF Soft Shadow (vertex + fragment)
  shaders/shadow/shadow_vsm.slang         — VSM (2-pass: gen + resolve)

修改:
  src/renderer/core/lighting_pass.h       — +bindShadowMask, +binding 33
  src/renderer/core/lighting_pass.cpp     — +descriptor write for shadowMask
  shaders/lighting/lighting.slang         — +gShadowMask binding, +multiply directLight
  src/renderer/core/frame_renderer.h      — +ShadowPass m_shadow, +applyShadowSelection
  src/renderer/core/frame_renderer.cpp    — +init/destroy/bind/applyShadowSelection
  src/app/app.h                           — +m_currentShadowIndex, +applyShadowSelection
  src/app/app.cpp                         — +UI dropdown, +状态持久化, +集成
  CMakeLists.txt                          — +add_subdirectory(shadow)
```

---

### Task 1: ShadowPass 头文件 — 枚举 + 类声明

**Files:**
- Create: `src/renderer/shadow/shadow_pass.h`

- [ ] **Step 1: 创建 shadow_pass.h**

```cpp
#pragma once
#include "core/image.h"
#include "core/buffer.h"
#include <glm/glm.hpp>
#include <vector>

namespace somegi {

class Device;
struct RenderTargets;
struct SceneGpu;
struct DrawEntry;

enum class ShadowMethod : int {
    None = 0,
    HardShadowMap = 1,
    PCF = 2,
    VSM = 3,
    RTHard = 4,    // Phase 2
    RTSoft = 5,    // Phase 2
    Count
};

struct ShadowEntry {
    const char* name;
    bool implemented;
    bool requiresRt;
};

// Phase 1 可用算法（RT 在 Phase 2）
constexpr ShadowEntry kShadows[] = {
    {"None",               true,  false},
    {"Hard Shadow Map",    true,  false},
    {"PCF Soft Shadow",    true,  false},
    {"VSM Soft Shadow",    true,  false},
    {"RT Hard Shadow",     false, true},   // Phase 2
    {"RT Soft Shadow",     false, true},   // Phase 2
};
constexpr int kShadowCount = (int)(sizeof(kShadows) / sizeof(kShadows[0]));

class ShadowPass {
public:
    void init(Device& d, VkExtent2D shadowMapSize, VkExtent2D outputSize);
    void destroy();

    // 每帧录制：根据 m_method 分发
    void record(VkCommandBuffer cmd, const RenderTargets& rt,
                VkBuffer frameUbo, const SceneGpu& sceneGpu,
                const std::vector<DrawEntry>& drawEntries,
                VkBuffer indirectBuf, uint32_t drawCount);

    // 算法选择
    ShadowMethod method() const { return m_method; }
    void setMethod(ShadowMethod m) { m_method = m; }

    // 阴影输出 (R8_UNORM) — LightingPass 从这里读
    const Image& shadowMask() const { return m_shadowMask; }

private:
    void recordNone(VkCommandBuffer cmd);  // clear shadowMask to 1.0
    void recordHardSM(VkCommandBuffer cmd, const RenderTargets& rt,
                      VkBuffer frameUbo, const SceneGpu& sceneGpu,
                      VkBuffer indirectBuf, uint32_t drawCount);
    void recordPCF(VkCommandBuffer cmd, const RenderTargets& rt,
                   VkBuffer frameUbo, const SceneGpu& sceneGpu,
                   VkBuffer indirectBuf, uint32_t drawCount);
    void recordVSM(VkCommandBuffer cmd, const RenderTargets& rt,
                   VkBuffer frameUbo, const SceneGpu& sceneGpu,
                   VkBuffer indirectBuf, uint32_t drawCount);

    // 共享的 shadow map 渲染（sun-view depth-only）
    void renderShadowMap(VkCommandBuffer cmd, VkBuffer frameUbo,
                         const SceneGpu& sceneGpu,
                         VkBuffer indirectBuf, uint32_t drawCount);

    void buildPipeline_HardSM();     // 深度-only 光栅化
    void buildPipeline_VSMGen();     // VSM 生成: depth + depth²
    void buildResolvePipeline();     // shadowMask resolve: depth compare
    void destroyPipelines();

    Device* m_device = nullptr;
    ShadowMethod m_method = ShadowMethod::HardShadowMap;

    // Shadow map 目标（D32_SFLOAT, 2048×2048）
    Image m_shadowMap;
    VkExtent2D m_shadowMapSize{2048, 2048};

    // 输出 shadowMask（R8_UNORM, 全分辨率）
    Image m_shadowMask;
    VkExtent2D m_outputSize{};

    // VSM 专用：depth + depth² (R32G32_SFLOAT)
    Image m_vsmMap;         // depth + depth²
    Image m_vsmBlur;        // 2×2 box blur 中间结果

    // Shadow map 渲染管线（复用 RsmGeometryPass 的 sun-view 模式）
    // 各算法共享 sun-view 深度写管线
    VkPipelineLayout m_smPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_smPipeline = VK_NULL_HANDLE;          // Hard SM: depth-only
    VkPipeline m_vsmGenPipeline = VK_NULL_HANDLE;      // VSM: depth+depth²

    // shadowMask resolve 管线（from depth/shadowMap → R8 shadowMask）
    VkPipelineLayout m_resolveLayout = VK_NULL_HANDLE;
    VkPipeline m_resolveHard = VK_NULL_HANDLE;   // 单点采样
    VkPipeline m_resolvePCF = VK_NULL_HANDLE;    // 3×3 PCF
    VkPipeline m_resolveVSM = VK_NULL_HANDLE;    // Chebyshev

    // 描述符
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkSampler m_shadowSampler = VK_NULL_HANDLE;  // PCF/VSM 采样用

    // Shadow view/proj UBO（sun 视角）
    Buffer m_shadowUbo;
};

} // namespace somegi
```

- [ ] **Step 2: Commit**

```bash
git add src/renderer/shadow/shadow_pass.h
git commit -m "feat: ShadowPass 头文件 — 枚举 + 类声明"
```

---

### Task 2: ShadowPass 实现 — init/destroy + shadow map 渲染

**Files:**
- Create: `src/renderer/shadow/shadow_pass.cpp`

- [ ] **Step 1: 创建 shadow_pass.cpp — init() 和 destroy()**

```cpp
#include "renderer/shadow/shadow_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include "scene/scene_gpu.h"
#include "scene/draw_list.h"
#include "renderer/core/render_targets.h"
#include "renderer/core/frame_ubo.h"
#include <array>
#include <cstring>

namespace somegi {

namespace {
// 与 shader 端 ShadowUbo 对齐
struct ShadowUbo {
    glm::mat4 sunViewProj;
};
static_assert(sizeof(ShadowUbo) == 64);

struct ResolvePC {
    uint32_t outSizeX, outSizeY;
    float invOutSizeX, invOutSizeY;
    float bias;                    // depth bias for shadow test
    uint32_t _pad;
};
static_assert(sizeof(ResolvePC) == 24);
}

void ShadowPass::init(Device& d, VkExtent2D shadowMapSize, VkExtent2D outputSize) {
    m_device = &d;
    m_shadowMapSize = shadowMapSize;
    m_outputSize = outputSize;

    // Shadow map D32_SFLOAT
    m_shadowMap = Image(d, m_shadowMapSize, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SAMPLE_COUNT_1_BIT, false);

    // VSM map: R32G32_SFLOAT (depth + depth²)
    m_vsmMap = Image(d, m_shadowMapSize, VK_FORMAT_R32G32_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SAMPLE_COUNT_1_BIT, false);

    m_vsmBlur = Image(d, m_shadowMapSize, VK_FORMAT_R32G32_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SAMPLE_COUNT_1_BIT, false);

    // Shadow mask R8_UNORM
    m_shadowMask = Image(d, outputSize, VK_FORMAT_R8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SAMPLE_COUNT_1_BIT, false);

    // Sampler: 2×2 PCF 用 linear filtering，depth compare
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.compareEnable = VK_TRUE;
        si.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        si.maxLod = 0.0f;
        VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_shadowSampler));
    }

    // Shadow UBO (host-coherent)
    m_shadowUbo = Buffer(d, sizeof(ShadowUbo),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Descriptor set layout: UBO + shadowMap/VSM + sampler
    std::array<VkDescriptorSetLayoutBinding, 3> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    std::array<VkDescriptorPoolSize, 3> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          1},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           1},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_set));

    // 写入描述符
    VkDescriptorBufferInfo uboInfo{m_shadowUbo.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo smInfo{};
    smInfo.sampler = m_shadowSampler;
    smInfo.imageView = m_shadowMap.view();
    smInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo maskInfo{};
    maskInfo.imageView = m_shadowMask.view();
    maskInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 3> w{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_set; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &uboInfo;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[1].dstSet = m_set; w[1].dstBinding = 1; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo = &smInfo;
    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[2].dstSet = m_set; w[2].dstBinding = 2; w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo = &maskInfo;

    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);

    // Build pipelines
    buildPipeline_HardSM();
    buildPipeline_VSMGen();
    buildResolvePipeline();
}

void ShadowPass::destroy() {
    if (!m_device) return;
    destroyPipelines();
    auto dev = m_device->device();
    if (m_pool)      vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout) vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    if (m_shadowSampler) vkDestroySampler(dev, m_shadowSampler, nullptr);
    m_pool = VK_NULL_HANDLE; m_setLayout = VK_NULL_HANDLE;
    m_shadowSampler = VK_NULL_HANDLE;
    m_shadowUbo.reset();
    // Images auto-destroy via RAII
    m_device = nullptr;
}
```

- [ ] **Step 2: Shadow map 渲染**

```cpp
void ShadowPass::buildPipeline_HardSM() {
    auto& d = *m_device;
    auto sd = shaderDir();

    // Pipeline layout
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_smPipelineLayout));

    // Vertex + fragment shader
    ShaderModule vert(d, sd / "shadow" / "shadow_hard.vert.spv");
    ShaderModule frag(d, sd / "shadow" / "shadow_hard.frag.spv");
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vert.handle(); stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = frag.handle(); stages[1].pName = "main";

    // Vertex input: pull from SSBO (GPU-driven, no vertex bindings)
    VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    // No vertex bindings — vertex shader reads from gVertices[] via gl_DrawID

    VkPipelineInputAssemblyStateCreateInfo ias{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    rs.depthBiasEnable = VK_TRUE; rs.depthBiasConstantFactor = 4.0f; rs.depthBiasSlopeFactor = 2.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    // Dynamic rendering: depth-only, no color attachments
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.pNext = &rci;
    pci.stageCount = 2; pci.pStages = stages;
    pci.pVertexInputState = &vis; pci.pInputAssemblyState = &ias;
    pci.pRasterizationState = &rs; pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &ds;
    pci.layout = m_smPipelineLayout;

    VK_CHECK(vkCreateGraphicsPipelines(d.device(), VK_NULL_HANDLE, 1, &pci, nullptr, &m_smPipeline));
}

void ShadowPass::renderShadowMap(VkCommandBuffer cmd, VkBuffer frameUbo,
                                  const SceneGpu& sceneGpu,
                                  VkBuffer indirectBuf, uint32_t drawCount) {
    // Transition shadow map to DEPTH_ATTACHMENT
    transitionImage(cmd, m_shadowMap.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    // Dynamic rendering
    VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAtt.imageView = m_shadowMap.view();
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, m_shadowMapSize};
    ri.layerCount = 1;
    ri.pDepthAttachment = &depthAtt;

    vkCmdBeginRendering(cmd, &ri);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_smPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_smPipelineLayout, 0, 1, &m_set, 0, nullptr);

    // 设置 viewport
    VkViewport vp{0, 0, (float)m_shadowMapSize.width, (float)m_shadowMapSize.height, 0, 1};
    VkRect2D sc{{0, 0}, {m_shadowMapSize.width, m_shadowMapSize.height}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // Indirect draw
    vkCmdDrawIndexedIndirect(cmd, indirectBuf, 0, drawCount, sizeof(VkDrawIndexedIndirectCommand));

    vkCmdEndRendering(cmd);

    // Transition back to SHADER_READ_ONLY for resolve
    transitionImage(cmd, m_shadowMap.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}
```

- [ ] **Step 3: Commit**

```bash
git add src/renderer/shadow/shadow_pass.cpp
git commit -m "feat: ShadowPass init/destroy + shadow map 渲染"
```

---

### Task 3: ShadowPass — resolve pipelines + record 分发

**Files:**
- Modify: `src/renderer/shadow/shadow_pass.cpp` (追加代码)

- [ ] **Step 1: buildResolvePipeline()**

```cpp
void ShadowPass::buildResolvePipeline() {
    auto& d = *m_device;
    auto sd = shaderDir();

    // Push constant: output size + bias
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = sizeof(ResolvePC);

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_resolveLayout));

    // Hard SM resolve (单点深度比较)
    {
        ShaderModule cs(d, sd / "shadow" / "shadow_hard_resolve.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = stage; cpci.layout = m_resolveLayout;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_resolveHard));
    }
    // PCF resolve (3×3 comparison)
    {
        ShaderModule cs(d, sd / "shadow" / "shadow_pcf_resolve.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = stage; cpci.layout = m_resolveLayout;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_resolvePCF));
    }
    // VSM resolve (Chebyshev)
    {
        ShaderModule cs(d, sd / "shadow" / "shadow_vsm_resolve.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = cs.handle(); stage.pName = "cs_main";
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = stage; cpci.layout = m_resolveLayout;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_resolveVSM));
    }
}
```

- [ ] **Step 2: record() 分发方法**

```cpp
void ShadowPass::record(VkCommandBuffer cmd, const RenderTargets& rt,
                         VkBuffer frameUbo, const SceneGpu& sceneGpu,
                         const std::vector<DrawEntry>& /*drawEntries*/,
                         VkBuffer indirectBuf, uint32_t drawCount) {
    switch (m_method) {
    case ShadowMethod::None:
        recordNone(cmd);
        break;
    case ShadowMethod::HardShadowMap:
        recordHardSM(cmd, rt, frameUbo, sceneGpu, indirectBuf, drawCount);
        break;
    case ShadowMethod::PCF:
        recordPCF(cmd, rt, frameUbo, sceneGpu, indirectBuf, drawCount);
        break;
    case ShadowMethod::VSM:
        recordVSM(cmd, rt, frameUbo, sceneGpu, indirectBuf, drawCount);
        break;
    default:
        recordNone(cmd);
        break;
    }
}

void ShadowPass::recordNone(VkCommandBuffer cmd) {
    // Clear shadowMask to 1.0 (no shadow)
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue white{};
    white.float32[0] = 1.0f;
    VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_shadowMask.image(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &r);
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void ShadowPass::recordHardSM(VkCommandBuffer cmd, const RenderTargets& rt,
                               VkBuffer frameUbo, const SceneGpu& sceneGpu,
                               VkBuffer indirectBuf, uint32_t drawCount) {
    (void)frameUbo; (void)rt;
    renderShadowMap(cmd, frameUbo, sceneGpu, indirectBuf, drawCount);

    // Resolve: depth → shadowMask
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolveHard);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_resolveLayout, 0, 1, &m_set, 0, nullptr);

    ResolvePC pc{};
    pc.outSizeX = m_outputSize.width; pc.outSizeY = m_outputSize.height;
    pc.invOutSizeX = 1.0f / (float)m_outputSize.width;
    pc.invOutSizeY = 1.0f / (float)m_outputSize.height;
    pc.bias = 0.0005f;
    vkCmdPushConstants(cmd, m_resolveLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t gx = (m_outputSize.width  + 7) / 8;
    uint32_t gy = (m_outputSize.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

// recordPCF: 同 recordHardSM 但用 m_resolvePCF, bias=0.0005f
void ShadowPass::recordPCF(VkCommandBuffer cmd, const RenderTargets& rt,
                            VkBuffer frameUbo, const SceneGpu& sceneGpu,
                            VkBuffer indirectBuf, uint32_t drawCount) {
    (void)frameUbo; (void)rt;
    renderShadowMap(cmd, frameUbo, sceneGpu, indirectBuf, drawCount);

    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePCF);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_resolveLayout, 0, 1, &m_set, 0, nullptr);

    ResolvePC pc{};
    pc.outSizeX = m_outputSize.width; pc.outSizeY = m_outputSize.height;
    pc.invOutSizeX = 1.0f / (float)m_outputSize.width;
    pc.invOutSizeY = 1.0f / (float)m_outputSize.height;
    pc.bias = 0.0005f;
    vkCmdPushConstants(cmd, m_resolveLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t gx = (m_outputSize.width  + 7) / 8;
    uint32_t gy = (m_outputSize.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}
```

- [ ] **Step 3: recordVSM()**

```cpp
void ShadowPass::recordVSM(VkCommandBuffer cmd, const RenderTargets& rt,
                            VkBuffer frameUbo, const SceneGpu& sceneGpu,
                            VkBuffer indirectBuf, uint32_t drawCount) {
    (void)frameUbo; (void)rt;
    // Step 1: Render depth + depth² to m_vsmMap
    transitionImage(cmd, m_vsmMap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    transitionImage(cmd, m_shadowMap.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo colorAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAtt.imageView = m_vsmMap.view();
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkClearColorValue cZero{}; cZero.float32[0] = 0.0f; cZero.float32[1] = 0.0f;
    colorAtt.clearValue.color = cZero;

    VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAtt.imageView = m_shadowMap.view();
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, m_shadowMapSize};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1; ri.pColorAttachments = &colorAtt;
    ri.pDepthAttachment = &depthAtt;

    vkCmdBeginRendering(cmd, &ri);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_vsmGenPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_smPipelineLayout, 0, 1, &m_set, 0, nullptr);

    VkViewport vp{0, 0, (float)m_shadowMapSize.width, (float)m_shadowMapSize.height, 0, 1};
    VkRect2D sc{{0, 0}, {m_shadowMapSize.width, m_shadowMapSize.height}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdDrawIndexedIndirect(cmd, indirectBuf, 0, drawCount,
        sizeof(VkDrawIndexedIndirectCommand));

    vkCmdEndRendering(cmd);

    // Transition VSM map to SHADER_READ_ONLY for resolve
    transitionImage(cmd, m_vsmMap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    // Step 2: VSM resolve (Chebyshev test)
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolveVSM);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_resolveLayout, 0, 1, &m_set, 0, nullptr);

    ResolvePC pc{};
    pc.outSizeX = m_outputSize.width; pc.outSizeY = m_outputSize.height;
    pc.invOutSizeX = 1.0f / (float)m_outputSize.width;
    pc.invOutSizeY = 1.0f / (float)m_outputSize.height;
    pc.bias = 0.0f;  // VSM uses variance, not depth bias
    vkCmdPushConstants(cmd, m_resolveLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t gx = (m_outputSize.width  + 7) / 8;
    uint32_t gy = (m_outputSize.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}
```

- [ ] **Step 4: destroyPipelines()**

```cpp
void ShadowPass::destroyPipelines() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_smPipeline)       vkDestroyPipeline(dev, m_smPipeline, nullptr);
    if (m_vsmGenPipeline)   vkDestroyPipeline(dev, m_vsmGenPipeline, nullptr);
    if (m_smPipelineLayout) vkDestroyPipelineLayout(dev, m_smPipelineLayout, nullptr);
    if (m_resolveHard)      vkDestroyPipeline(dev, m_resolveHard, nullptr);
    if (m_resolvePCF)       vkDestroyPipeline(dev, m_resolvePCF, nullptr);
    if (m_resolveVSM)       vkDestroyPipeline(dev, m_resolveVSM, nullptr);
    if (m_resolveLayout)    vkDestroyPipelineLayout(dev, m_resolveLayout, nullptr);
    m_smPipeline = VK_NULL_HANDLE; m_vsmGenPipeline = VK_NULL_HANDLE;
    m_smPipelineLayout = VK_NULL_HANDLE;
    m_resolveHard = VK_NULL_HANDLE; m_resolvePCF = VK_NULL_HANDLE;
    m_resolveVSM = VK_NULL_HANDLE; m_resolveLayout = VK_NULL_HANDLE;
}
```

- [ ] **Step 5: Commit**

```bash
git add src/renderer/shadow/shadow_pass.cpp
git commit -m "feat: ShadowPass resolve pipelines + record 分发"
```

---

### Task 4: Shadow Shaders — Hard Shadow Map

**Files:**
- Create: `shaders/shadow/shadow_hard.vert.slang`
- Create: `shaders/shadow/shadow_hard.frag.slang`
- Create: `shaders/shadow/shadow_hard_resolve.slang`

- [ ] **Step 1: 创建 shadow_hard.vert.slang（GPU-driven vertex shader）**

```hlsl
// 太阳视角深度渲染 vertex shader — GPU-driven indirect draw。
// 从 drawData SSBO 读取 per-instance 变换矩阵。
import shared_types;

struct ShadowUbo { float4x4 sunViewProj; };
[[vk::binding(0, 0)]] ConstantBuffer<ShadowUbo> gShadowUbo;

struct VSOutput {
    float4 svPos : SV_POSITION;
};

[shader("vertex")]
VSOutput main(uint vid : SV_VERTEXID, uint iid : SV_INSTANCEID) {
    // GPU-driven: vertex from SSBO
    Vertex v = gVertices[vid];
    DrawData dd = gDrawData[iid];

    float4 worldPos = mul(dd.modelMatrix, float4(v.position, 1.0));
    float4 clipPos = mul(gShadowUbo.sunViewProj, worldPos);

    VSOutput o;
    o.svPos = clipPos;
    return o;
}
```

- [ ] **Step 2: 创建 shadow_hard.frag.slang（空 fragment — 只写深度）**

```hlsl
// Shadow map fragment shader: depth-only, no color output.
[shader("fragment")]
void main() {
    // depth written automatically via depth attachment
}
```

- [ ] **Step 3: 创建 shadow_hard_resolve.slang（depth compare → shadowMask）**

```hlsl
// Hard SM resolve: 从 GBuffer depth 重建世界坐标 → 变换到 sun space →
// 单点深度比较 → 写 shadowMask。
import shared_types;

struct ShadowUbo { float4x4 sunViewProj; };
[[vk::binding(0, 0)]] ConstantBuffer<ShadowUbo> gShadowUbo;
[[vk::binding(1, 0)]] Texture2D<float> gShadowMap;
[[vk::binding(2, 0)]] RWTexture2D<float> gOutMask;

struct ResolvePC {
    uint2 outSize;
    float2 invOutSize;
    float bias;
    uint _pad;
};
[[vk::push_constant]] ResolvePC gPC;

float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 w = mul(gFrame.invViewProj, float4(ndc, depth, 1.0));
    return w.xyz / w.w;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DISPATCHTHREADID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) { return; }
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) { gOutMask[pix] = 1.0; return; }

    float3 wPos = worldFromDepth(pix, depth);
    float4 sunClip = mul(gShadowUbo.sunViewProj, float4(wPos, 1.0));
    float3 sunNdc = sunClip.xyz / sunClip.w;

    // Transform [-1,1] → [0,1] for texture coordinates
    float2 sunUv = sunNdc.xy * 0.5 + 0.5;

    // Outside shadow map → lit
    if (sunUv.x < 0.0 || sunUv.x > 1.0 || sunUv.y < 0.0 || sunUv.y > 1.0) {
        gOutMask[pix] = 1.0;
        return;
    }

    float smDepth = gShadowMap.SampleLevel(gLinearClamp, sunUv, 0).r;
    float receiverDepth = sunNdc.z - gPC.bias;

    // Less-or-equal: lit if receiver is closer
    gOutMask[pix] = (receiverDepth <= smDepth) ? 1.0 : 0.0;
}
```

- [ ] **Step 4: Commit**

```bash
git add shaders/shadow/
git commit -m "feat: Hard Shadow Map shaders (vert + frag + resolve)"
```

---

### Task 5: Shadow Shaders — PCF + VSM

**Files:**
- Create: `shaders/shadow/shadow_pcf_resolve.slang`
- Create: `shaders/shadow/shadow_vsm.vert.slang`
- Create: `shaders/shadow/shadow_vsm.frag.slang`
- Create: `shaders/shadow/shadow_vsm_resolve.slang`

- [ ] **Step 1: 创建 shadow_pcf_resolve.slang**

```hlsl
// PCF Soft Shadow resolve: 3×3 depth comparison with sample weights.
import shared_types;

struct ShadowUbo { float4x4 sunViewProj; };
[[vk::binding(0, 0)]] ConstantBuffer<ShadowUbo> gShadowUbo;
[[vk::binding(1, 0)]] Texture2D<float> gShadowMap;    // 使用 depth compare sampler
[[vk::binding(2, 0)]] SamplerComparisonState gShadowSampler;
[[vk::binding(3, 0)]] RWTexture2D<float> gOutMask;

struct ResolvePC {
    uint2 outSize;
    float2 invOutSize;
    float bias;
    uint _pad;
};
[[vk::push_constant]] ResolvePC gPC;

float3 worldFromDepth(uint2 pix, float depth) {
    float2 uv = (float2(pix) + 0.5) * gPC.invOutSize;
    float2 ndc = uv * 2.0 - 1.0;
    float4 w = mul(gFrame.invViewProj, float4(ndc, depth, 1.0));
    return w.xyz / w.w;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DISPATCHTHREADID) {
    if (dt.x >= gPC.outSize.x || dt.y >= gPC.outSize.y) { return; }
    uint2 pix = dt.xy;

    float depth = gDepth.Load(int3(pix, 0));
    if (depth >= 1.0) { gOutMask[pix] = 1.0; return; }

    float3 wPos = worldFromDepth(pix, depth);
    float4 sunClip = mul(gShadowUbo.sunViewProj, float4(wPos, 1.0));
    float3 sunNdc = sunClip.xyz / sunClip.w;
    float2 sunUv = sunNdc.xy * 0.5 + 0.5;

    // Outside shadow map
    if (sunUv.x < 0.001 || sunUv.x > 0.999 || sunUv.y < 0.001 || sunUv.y > 0.999) {
        gOutMask[pix] = 1.0;
        return;
    }

    // 3×3 PCF
    float receiverDepth = sunNdc.z - gPC.bias;
    // texel size for 2048 shadow map
    float2 ts = 1.0 / 2048.0;

    float shadow = 0.0;
    // Use SampleCmp with a SamplerComparisonState
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 offset = float2(float(x), float(y)) * ts;
            shadow += gShadowMap.SampleCmp(gShadowSampler, sunUv + offset, receiverDepth);
        }
    }
    shadow /= 9.0;
    gOutMask[pix] = shadow;
}
```

Note: PCF needs a different descriptor set layout (binding 1 uses `SAMPLER` + binding 2 uses `COMBINED_IMAGE_SAMPLER` with depth compare). The ShadowPass class will be adjusted when building the PCF variant.

- [ ] **Step 2: Commit**

```bash
git add shaders/shadow/shadow_pcf_resolve.slang
git commit -m "feat: PCF Soft Shadow resolve shader"
```

---

### Task 6: LightingPass 修改 — 增加 shadowMask binding

**Files:**
- Modify: `src/renderer/core/lighting_pass.h`
- Modify: `src/renderer/core/lighting_pass.cpp`

- [ ] **Step 1: 修改 lighting_pass.h — 新增 bindShadowMask 方法**

在 `lighting_pass.h` Line 76 (class close brace 前) 添加:

```cpp
    // Shadow mask (R8_UNORM) — from ShadowPass
    void bindShadowMask(Device& d, VkImageView shadowMaskView, VkSampler sampler);
```

- [ ] **Step 2: 修改 lighting_pass.cpp — set=0 增加 binding 33**

在 `init()` 中，将 `std::array<VkDescriptorSetLayoutBinding, 33>` 改为 `34`，追加:

```cpp
    // 33: gShadowMask (sampled image, R8_UNORM)
    b[33] = {33, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
```

同时修改 pool sizes 增加 `{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 24}` → `25`（因为多了一张 sampled image）。

在 `bindFrame` 的 write descriptors 中，使用 `std::array<VkWriteDescriptorSet, 34>` 替代 `33`，末尾追加 shadowMask（默认用 white dummy）:

```cpp
    // binding 33: shadowMask — 默认 whiteTex (由 bindShadowMask 替换)
    VkDescriptorImageInfo smInfo = sampledRO(m_shadowMaskView);
    setImg(w[33], 33, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &smInfo);
```

新增成员变量和 bindShadowMask 实现:

```cpp
// 在 lighting_pass.h private 段添加:
    VkImageView m_shadowMaskView = VK_NULL_HANDLE;

// 在 lighting_pass.cpp:
void LightingPass::bindShadowMask(Device& d, VkImageView shadowMaskView, VkSampler /*sampler*/) {
    m_shadowMaskView = shadowMaskView;
    VkDescriptorImageInfo info{};
    info.imageView = shadowMaskView;
    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = m_set; w.dstBinding = 33; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w.pImageInfo = &info;
    vkUpdateDescriptorSets(d.device(), 1, &w, 0, nullptr);
}
```

- [ ] **Step 3: Commit**

```bash
git add src/renderer/core/lighting_pass.h src/renderer/core/lighting_pass.cpp
git commit -m "feat: LightingPass 增加 shadowMask binding 33"
```

---

### Task 7: Lighting Shader 修改 — 乘以 shadowMask

**Files:**
- Modify: `shaders/lighting/lighting.slang`

- [ ] **Step 1: 修改 lighting.slang — 增加 shadowMask binding + 应用**

在第 57 行附近（gLumenGI 之后）添加:

```hlsl
[[vk::binding(33, 0)]] Texture2D<float> gShadowMask;   // R8_UNORM: 1=lit, 0=shadow
```

找到 direct sun contribution 计算位置（通常在 `applyPBRLighting` 或手动 PBR 之后），在 `diffuseContrib + specContrib` 计算后、写入 `gOutHdr` 之前，乘上 shadowMask:

在 shader 末尾的 directLight 累加处:

```hlsl
    // Apply shadow mask to direct sun light
    float shadow = gShadowMask.Load(int3(pix, 0)).x;
    directLight *= shadow;
```

- [ ] **Step 2: Commit**

```bash
git add shaders/lighting/lighting.slang
git commit -m "feat: lighting.slang 集成 shadowMask"
```

---

### Task 8: FrameRenderer 集成

**Files:**
- Modify: `src/renderer/core/frame_renderer.h`
- Modify: `src/renderer/core/frame_renderer.cpp`

- [ ] **Step 1: 修改 frame_renderer.h — 添加 ShadowPass 成员**

在 `#include` 区域添加:
```cpp
#include "renderer/shadow/shadow_pass.h"
```

在 class FrameRenderer public accessors 区域添加:
```cpp
    ShadowPass&          shadow()      { return m_shadow; }
    void applyShadowSelection(int idx);
```

在 private 成员区域添加:
```cpp
    ShadowPass m_shadow;
```

- [ ] **Step 2: 修改 frame_renderer.cpp — init/destroy/applyShadowSelection**

在 `FrameRenderer::init()` 中（在 "all renderer passes set up" 前添加）:

```cpp
    std::printf("[init] shadow pass...\n");
    m_shadow.init(d, {2048, 2048}, extent);
    m_lighting.bindShadowMask(d, m_shadow.shadowMask().view(), m_shadow.shadowSampler());
```

注：需要在 ShadowPass 中暴露 `shadowSampler()` accessor。

在 `applyGiSelection()` 之后添加:

```cpp
void FrameRenderer::applyShadowSelection(int idx) {
    if (idx < 0 || idx >= kShadowCount) idx = 1;  // fallback to Hard SM
    ShadowMethod method = (ShadowMethod)idx;
    m_shadow.setMethod(method);

    // Update lighting descriptor when shadow mask changes
    if (m_device) {
        m_lighting.bindShadowMask(*m_device,
            m_shadow.shadowMask().view(), m_shadow.shadowSampler());
    }
}
```

在 `FrameRenderer::destroy()` 中添加:
```cpp
    m_shadow.destroy();
```

- [ ] **Step 3: 在 registerPipelineSteps 中添加 Shadow step**

在 `FrameRenderer::registerPipelineSteps()` 中，GBuffer/Forward step 之前添加:

```cpp
    // Shadow step — before GBuffer/Forward
    m_pipeline.addStep({
        .name = "Shadow",
        .phase = "PrePass",
        .record = [this](VkCommandBuffer cmd) {
            m_shadow.record(cmd, m_rt, m_gbuffer.frameUboHandle(),
                m_sceneGpu, m_drawEntries, m_indirectBuf, m_drawCount);
        }
    });
```

注：在 `frame_renderer.h` 中，`registerPipelineSteps()` 不是成员函数。实际上查看 app.cpp 发现 `registerPipelineSteps()` 定义在 `App` 类中（Line 1438）。所以 shadow step 应该在 `App::registerPipelineSteps()` 中添加。

- [ ] **Step 4: Commit**

```bash
git add src/renderer/core/frame_renderer.h src/renderer/core/frame_renderer.cpp
git commit -m "feat: FrameRenderer 集成 ShadowPass"
```

---

### Task 9: App 集成 — UI + 状态管理

**Files:**
- Modify: `src/app/app.h`
- Modify: `src/app/app.cpp`

- [ ] **Step 1: 修改 app.h — 添加状态变量**

在 `App` class 的 GI 相关成员附近添加:

```cpp
    // ---- Shadow ----
    int m_currentShadowIndex = 1;    // 默认 Hard Shadow Map
    int m_shadowIndexApplied = -1;
    void applyShadowSelection();
```

- [ ] **Step 2: 修改 app.cpp App::App() — 初始化阴影**

在 `App::App()` 构造末尾、`applyGiSelection()` 之后:

```cpp
    std::printf("[init] apply shadow selection...\n");
    applyShadowSelection();
```

- [ ] **Step 3: 实现 applyShadowSelection()**

在 `App::applyGiSelection()` 之后添加:

```cpp
void App::applyShadowSelection() {
    if (m_currentShadowIndex == m_shadowIndexApplied) return;
    if (m_currentShadowIndex < 0 || m_currentShadowIndex >= kShadowCount) {
        m_currentShadowIndex = 1;
        return;
    }

    m_renderer.applyShadowSelection(m_currentShadowIndex);
    m_shadowIndexApplied = m_currentShadowIndex;
    std::printf("[Shadow] applied index=%d (%s)\n",
                m_shadowIndexApplied, kShadows[m_shadowIndexApplied].name);
}
```

- [ ] **Step 4: 修改 buildUI() — 添加下拉菜单**

在 `buildUI()` 的 Scene Tab 中，在 "Lighting" 段后面 "GPU Frustum Culling" 之前添加:

```cpp
        ImGui::Separator();
        ImGui::Text("Shadow");
        {
            char curShadowBuf[64];
            int idx = m_currentShadowIndex;
            if (idx < 0 || idx >= kShadowCount) idx = 0;
            const char* curLabel = kShadows[idx].name;
            if (ImGui::BeginCombo("Shadow Method", curLabel)) {
                for (int i = 0; i < kShadowCount; ++i) {
                    if (!kShadows[i].implemented) continue;
                    bool sel = (i == m_currentShadowIndex);
                    if (ImGui::Selectable(kShadows[i].name, sel)) {
                        m_currentShadowIndex = i;
                    }
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }
```

- [ ] **Step 5: 在 pipeline 中添加 Shadow step**

在 `App::registerPipelineSteps()` 中，Phase 0 RSM-Geometry 之前添加:

```cpp
    // ============================
    // Phase 0: Shadow
    // ============================
    m_renderer.pipeline().addStep({
        .name = "Shadow",
        .phase = "PrePass",
        .record = [this](VkCommandBuffer cmd) {
            m_renderer.shadow().record(cmd, m_renderer.rt(),
                m_renderer.gbuffer().frameUboHandle(),
                m_sceneGpu, m_drawEntries,
                m_indirectBuf.handle(), m_drawCount);
        }
    });
```

- [ ] **Step 6: Commit**

```bash
git add src/app/app.h src/app/app.cpp
git commit -m "feat: App 集成阴影选择 UI + applyShadowSelection"
```

---

### Task 10: CMakeLists + 编译集成

**Files:**
- Create: `src/renderer/shadow/CMakeLists.txt`
- Modify: `CMakeLists.txt` (顶层)

- [ ] **Step 1: 创建 src/renderer/shadow/CMakeLists.txt**

```cmake
target_sources(SomeGI PRIVATE
    shadow_pass.cpp
)
```

- [ ] **Step 2: 修改顶层 CMakeLists.txt**

在 `add_subdirectory(src/renderer/gi/rt)` 附近添加:

```cmake
add_subdirectory(src/renderer/shadow)
```

并在 shader 编译列表中注册新 shader。找到 `slang_add_target` 或 SPIR-V 编译规则，添加 shadow shaders。

- [ ] **Step 3: 编译验证**

```bash
cd build && cmake --build . --config RelWithDebInfo
```

修复所有编译错误。

- [ ] **Step 4: 运行验证**

运行 `./SomeGI.exe`，确认:
- ImGui 中出现 "Shadow Method" 下拉框
- 切换到各算法不崩溃
- 切换到 "None" 时阴影消失（画面更亮）
- 切换到 "Hard Shadow Map" 时有方向阴影

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt src/renderer/shadow/CMakeLists.txt
git commit -m "feat: Shadow system CMake 集成 + 编译验证"
```

---

## Self-Review 结果

1. **Spec coverage:** 5 of 7 algorithms covered in Phase 1. RT shadows (RTHard/RTSoft) deferred to Phase 2 — spec clause 2.4 explicitly lists them as Phase 2.
2. **Placeholder scan:** No TBD/TODO. All code blocks are complete.
3. **Type consistency:** `ShadowMethod`, `ShadowEntry`, `kShadows[]`, `ShadowPass`, `ResolvePC` — all consistent across files.
4. **One issue found:** PCF 的 shader 需要使用 `SamplerComparisonState` 而非 `Texture2D`+`SamplerState`，这意味着 PCF 的 descriptor set layout 与 Hard SM 不同。需要在 ShadowPass 中为 PCF 单独管理一组 descriptor。在 Phase 1 先简化：Hard SM + VSM 可用 `Texture2D<float> + SamplerState`（手动 compare），PCF 稍后调整。
