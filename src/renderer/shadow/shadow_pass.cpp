// ShadowPass 实现 —— 多种阴影算法的统一录制入口。
//
// 支持算法：None / HardSM / PCF / VSM（Phase 1）。
// 核心流程：
//   1. 从 sun 视角渲染深度图（或 VSM 的深度+深度平方），GPU-driven indirect draw
//   2. Compute resolve：将 shadow map 采样结果写入 R8 shadowMask
//
// 管线约定：
//   - 阴影图渲染：graphics pipeline, depth-only（Hard/PCF）或 1×color+depth（VSM）
//     GPU-driven: VS 从 SSBO（gVertices/gIndices/gDrawData）读取，无传统 vertex input
//   - Resolve：compute pipeline, 绑定 [0]=UBO, [1]=COMBINED_IMAGE_SAMPLER,
//     [2]=STORAGE_IMAGE(shadowMask)
//
// 注意：shader .spv 文件尚不存在（Task 4/5），ShaderModule 构造会因此 404。

#include "renderer/shadow/shadow_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include "renderer/core/frame_ubo.h"
#include "scene/scene_gpu.h"
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <cstring>
#include <limits>

namespace somegi {

namespace {

// ShadowMap 渲染的 UBO（set=0, binding=0）—— 与 shader 端 ConstantBuffer<ShadowUbo> 对齐
struct ShadowUbo {
    glm::mat4 sunViewProj;
};

// Compute resolve 的 push constant —— 与 shader 端 ResolvePC 对齐
struct ResolvePC {
    uint32_t outSizeX, outSizeY;
    float    invOutSizeX, invOutSizeY;
    float    bias;
    uint32_t _pad;
    float    invShadowMapX, invShadowMapY;  // PCF texel 步长（1/w, 1/h）
};
static_assert(sizeof(ResolvePC) == 32, "ResolvePC must match shader push constant layout");

// PCSS 专用 push constant —— 比 ResolvePC 多 lightSize + maxKernelRadius
struct PCSS_PC {
    uint32_t outSizeX, outSizeY;
    float    invOutSizeX, invOutSizeY;
    float    bias;
    uint32_t _pad;
    float    invShadowMapX, invShadowMapY;
    float    lightSize;         // 太阳角半径（如 0.03）
    float    maxKernelRadius;   // 最大 PCF kernel（texels，如 15.0）
};
static_assert(sizeof(PCSS_PC) == 40, "PCSS_PC must match shader push constant layout");

} // anonymous namespace

// ────────────────────────────────────────────────────────────────────────────
// 公共接口
// ────────────────────────────────────────────────────────────────────────────

void ShadowPass::init(Device& d, VkExtent2D shadowMapSize, VkExtent2D outputSize) {
    m_device = &d;
    m_shadowMapSize = shadowMapSize;
    m_outputSize = outputSize;

    // ── 1. Shadow map (D32_SFLOAT, depth-only) ──
    {
        ImageDesc desc{};
        desc.format = VK_FORMAT_D32_SFLOAT;
        desc.extent = {shadowMapSize.width, shadowMapSize.height, 1};
        desc.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        desc.usage  = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                    | VK_IMAGE_USAGE_SAMPLED_BIT;
        m_shadowMap = Image(d, desc);
    }

    // ── 2. VSM map (R32G32_SFLOAT, depth+depth^2) ──
    {
        ImageDesc desc{};
        desc.format = VK_FORMAT_R32G32_SFLOAT;
        desc.extent = {shadowMapSize.width, shadowMapSize.height, 1};
        desc.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        desc.usage  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                    | VK_IMAGE_USAGE_SAMPLED_BIT;
        m_vsmMap = Image(d, desc);
    }

    // ── 2b. VSM blur intermediate (R32G32_SFLOAT, sampled + storage) ──
    {
        ImageDesc desc{};
        desc.format = VK_FORMAT_R32G32_SFLOAT;
        desc.extent = {shadowMapSize.width, shadowMapSize.height, 1};
        desc.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        desc.usage  = VK_IMAGE_USAGE_STORAGE_BIT   // blur shader 写入
                    | VK_IMAGE_USAGE_SAMPLED_BIT;   // resolve shader 采样
        m_vsmBlur = Image(d, desc);
    }

    // ── 3. Shadow mask output (R8_UNORM, storage + sampled) ──
    {
        ImageDesc desc{};
        desc.format = VK_FORMAT_R8_UNORM;
        desc.extent = {outputSize.width, outputSize.height, 1};
        desc.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        desc.usage  = VK_IMAGE_USAGE_STORAGE_BIT
                    | VK_IMAGE_USAGE_SAMPLED_BIT;
        m_shadowMask = Image(d, desc);
    }

    std::printf("[shadow] shadowMask view=%p\n", (void*)m_shadowMask.view());

    // ── 4. Shadow sampler (linear + depth compare, PCF 用) ──
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter        = VK_FILTER_LINEAR;
        si.minFilter        = VK_FILTER_LINEAR;
        si.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.compareEnable    = VK_TRUE;
        si.compareOp        = VK_COMPARE_OP_LESS_OR_EQUAL;
        si.maxLod           = 0.0f;
        VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_shadowSampler));
    }

    // ── 4b. VSM sampler (线性、无 depth compare，VSM resolve 用) ──
    //       VSM moments 是 (depth, depth²) 颜色数据，不应用深度比较
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter        = VK_FILTER_LINEAR;
        si.minFilter        = VK_FILTER_LINEAR;
        si.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.compareEnable    = VK_FALSE;
        si.maxLod           = 0.0f;
        VK_CHECK(vkCreateSampler(d.device(), &si, nullptr, &m_vsmSampler));
    }

    // ── 5. Resolve compute descriptor set layout ──
    //     与 shader 端 set=0 对齐。
    //     binding 1/2 在 cmd 录制期间可能被更新（VSM ↔ Hard 切换），
    //     使用 UPDATE_AFTER_BIND 标志允许安全更新。
    {
        std::array<VkDescriptorSetLayoutBinding, 4> bld{};
        bld[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bld[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bld[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLER,       1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bld[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        // binding 1,2 标记为 UPDATE_AFTER_BIND（Vulkan 1.2+ core）
        std::array<VkDescriptorBindingFlags, 2> flags{
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo fi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        fi.bindingCount = (uint32_t)flags.size(); fi.pBindingFlags = flags.data();

        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.pNext = &fi;
        li.bindingCount = (uint32_t)bld.size(); li.pBindings = bld.data();
        li.flags = 0;  // 不需要 VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));
    }

    // ── 6. Resolve descriptor pool + set（支持 UPDATE_AFTER_BIND）──
    {
        std::array<VkDescriptorPoolSize, 4> ps{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_SAMPLER,       1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        }};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.flags    = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pci.maxSets  = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
        VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_set));
    }

    // ── 7. SM graphics pipeline descriptor set layout ──
    //     GPU-driven VS 读取 SSBO：[0]=ShadowUbo, [1]=gVertices, [2]=gIndices, [3]=gDrawData
    {
        std::array<VkDescriptorSetLayoutBinding, 4> bld{};
        bld[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        bld[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        bld[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        bld[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = (uint32_t)bld.size(); li.pBindings = bld.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_smSetLayout));
    }

    // ── 8. SM descriptor pool + set ──
    {
        std::array<VkDescriptorPoolSize, 2> ps{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
        }};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
        VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_smPool));

        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = m_smPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_smSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_smSet));
    }

    // ── 9. 每帧资源 descriptor set layout（set=1：FrameUniforms + GBuffer depth）
    {
        std::array<VkDescriptorSetLayoutBinding, 2> bld{};
        bld[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bld[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = (uint32_t)bld.size(); li.pBindings = bld.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_frameSetLayout));
    }

    // ── 10. 每帧资源 descriptor pool + set ──
    {
        std::array<VkDescriptorPoolSize, 2> ps{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
        }};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
        VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_framePool));

        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = m_framePool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_frameSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_frameSet));
    }

    // ── 10b. VSM blur descriptor set layout ──
    //        set=0: [0]=SAMPLED_IMAGE(vsmMap), [1]=STORAGE_IMAGE(vsmBlur)
    //        两 binding 均在 recordVSM 中更新，需要 UPDATE_AFTER_BIND
    {
        std::array<VkDescriptorSetLayoutBinding, 2> bld{};
        bld[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bld[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        std::array<VkDescriptorBindingFlags, 2> flags{
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo fi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        fi.bindingCount = (uint32_t)flags.size(); fi.pBindingFlags = flags.data();

        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.pNext = &fi;
        li.bindingCount = (uint32_t)bld.size(); li.pBindings = bld.data();
        VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_vsmBlurSetLayout));
    }

    // ── 10c. VSM blur descriptor pool + set（支持 UPDATE_AFTER_BIND）──
    {
        std::array<VkDescriptorPoolSize, 2> ps{{
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        }};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.flags   = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
        VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_vsmBlurPool));

        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = m_vsmBlurPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_vsmBlurSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_vsmBlurSet));
    }

    // ── 10d. RT shadow descriptor set（仅 HW 支持 ray query 时创建）──
    bool rtOk = d.features().accelStruct && d.features().rayQuery;
    if (rtOk) {
        // set=0: [0]=TLAS, [1]=STORAGE_IMAGE(shadowMask), [2]=SAMPLED_IMAGE(normal)
        {
            std::array<VkDescriptorSetLayoutBinding, 3> bld{};
            bld[0] = {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            bld[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            bld[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,            1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

            VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            li.bindingCount = (uint32_t)bld.size(); li.pBindings = bld.data();
            VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_rtSetLayout));
        }

        {
            std::array<VkDescriptorPoolSize, 3> ps{{
                {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              1},
                {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,              1},
            }};
            VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
            VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_rtPool));

            VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dai.descriptorPool = m_rtPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_rtSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_rtSet));
        }

        // 初始写入：binding 1 = shadowMask（TLAS 在 bindTLAS 写入，normal 在 bindFrameResources 写入）
        {
            VkDescriptorImageInfo maskInfo{};
            maskInfo.imageView   = m_shadowMask.view();
            maskInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = m_rtSet; w.dstBinding = 1; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w.pImageInfo = &maskInfo;
            vkUpdateDescriptorSets(d.device(), 1, &w, 0, nullptr);
        }

        buildPipeline_RTHard();
    }

    // ── 11. Shadow UBO (host-coherent, renderShadowMap 每帧写入) ──
    m_shadowUbo = Buffer(d, sizeof(ShadowUbo),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // ── 12. 写入初始描述符 ──
    // Resolve set: binding 0=UBO, 1=SAMPLED_IMAGE, 2=SAMPLER, 3=STORAGE_IMAGE
    {
        VkDescriptorBufferInfo uboInfo{m_shadowUbo.handle(), 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo smImgInfo{};
        smImgInfo.imageView   = m_shadowMap.view();
        smImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo smSmpInfo{};
        smSmpInfo.sampler = m_shadowSampler;
        VkDescriptorImageInfo maskInfo{};
        maskInfo.imageView   = m_shadowMask.view();
        maskInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 4> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = m_set; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &uboInfo;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = m_set; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[1].pImageInfo = &smImgInfo;
        w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[2].dstSet = m_set; w[2].dstBinding = 2; w[2].descriptorCount = 1;
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[2].pImageInfo = &smSmpInfo;
        w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[3].dstSet = m_set; w[3].dstBinding = 3; w[3].descriptorCount = 1;
        w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[3].pImageInfo = &maskInfo;

        vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
    }

    // SM set: binding 0 = ShadowUbo（每帧会覆盖写入 mapped 内存，不需要重新写 DS）
    {
        VkDescriptorBufferInfo uboInfo{m_shadowUbo.handle(), 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = m_smSet; w.dstBinding = 0; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w.pBufferInfo = &uboInfo;
        vkUpdateDescriptorSets(d.device(), 1, &w, 0, nullptr);
    }

    // VSM blur set: 写入初始描述符（binding 0=vsmMap, binding 1=vsmBlur）
    // 每帧 recordVSM 会更新 image layout，此处只设初始值
    {
        VkDescriptorImageInfo inInfo{};
        inInfo.imageView   = m_vsmMap.view();
        inInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo outInfo{};
        outInfo.imageView   = m_vsmBlur.view();
        outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 2> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = m_vsmBlurSet; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &inInfo;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = m_vsmBlurSet; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &outInfo;
        vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
    }

    // ── 13. 构建 pipeline ──
    //     注意：shader .spv 尚不存在，ShaderModule 构造会抛异常。
    //     等 Task 4/5 编译 shader 后这些调用才会成功。
    buildPipeline_HardSM();
    buildPipeline_VSMGen();
    buildPipeline_VSMBlur();
    buildResolvePipeline();
}

// ──────────────────────────────────────────────────────────────────
// RT Hard shadow pipeline（compute shader + ray query）
// ──────────────────────────────────────────────────────────────────
void ShadowPass::buildPipeline_RTHard() {
    auto& d = *m_device;
    if (!m_rtSetLayout) return; // RT 不可用时跳过

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size       = 16;  // uint2 outSize + float2 invOutSize
    std::array<VkDescriptorSetLayout, 2> sets{m_rtSetLayout, m_frameSetLayout};

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = (uint32_t)sets.size(); plci.pSetLayouts = sets.data();
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_rtHardLayout));

    ShaderModule cs(d, shaderDir() / "shadow" / "shadow_rt_hard.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs.handle();
    stage.pName  = "cs_main";

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage  = stage;
    cpci.layout = m_rtHardLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_rtHardPipeline));

    // RT Soft pipeline（push constant 含 frameIndex + sunRadius）
    {
        VkPushConstantRange pc2{};
        pc2.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc2.size       = 32;  // uint2(8)+float2(8)+uint(4)+float(4)+uint(4)+pad(4)=32
        std::array<VkDescriptorSetLayout, 2> sets2{m_rtSetLayout, m_frameSetLayout};

        VkPipelineLayoutCreateInfo plci2{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci2.setLayoutCount = (uint32_t)sets2.size(); plci2.pSetLayouts = sets2.data();
        plci2.pushConstantRangeCount = 1; plci2.pPushConstantRanges = &pc2;
        VK_CHECK(vkCreatePipelineLayout(d.device(), &plci2, nullptr, &m_rtSoftLayout));

        ShaderModule cs2(d, shaderDir() / "shadow" / "shadow_rt_soft.spv");
        VkPipelineShaderStageCreateInfo stage2{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage2.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage2.module = cs2.handle();
        stage2.pName  = "cs_main";

        VkComputePipelineCreateInfo cpci2{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci2.stage  = stage2;
        cpci2.layout = m_rtSoftLayout;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci2, nullptr, &m_rtSoftPipeline));
    }
}

void ShadowPass::bindTLAS(Device& d, VkAccelerationStructureKHR tlas) {
    if (!m_rtSet) return;
    m_tlas = tlas;

    VkWriteDescriptorSetAccelerationStructureKHR tlasAI{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    tlasAI.accelerationStructureCount = 1;
    tlasAI.pAccelerationStructures = &m_tlas;

    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = m_rtSet; w.dstBinding = 0; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    w.pNext = &tlasAI;
    vkUpdateDescriptorSets(d.device(), 1, &w, 0, nullptr);
}

void ShadowPass::destroy() {
    if (!m_device) return;
    destroyPipelines();
    auto dev = m_device->device();
    if (m_pool)        vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_setLayout)   vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    if (m_framePool)   vkDestroyDescriptorPool(dev, m_framePool, nullptr);
    if (m_frameSetLayout) vkDestroyDescriptorSetLayout(dev, m_frameSetLayout, nullptr);
    if (m_smPool)      vkDestroyDescriptorPool(dev, m_smPool, nullptr);
    if (m_smSetLayout) vkDestroyDescriptorSetLayout(dev, m_smSetLayout, nullptr);
    if (m_vsmBlurPool) vkDestroyDescriptorPool(dev, m_vsmBlurPool, nullptr);
    if (m_vsmBlurSetLayout) vkDestroyDescriptorSetLayout(dev, m_vsmBlurSetLayout, nullptr);
    if (m_shadowSampler) vkDestroySampler(dev, m_shadowSampler, nullptr);
    if (m_vsmSampler)    vkDestroySampler(dev, m_vsmSampler, nullptr);
    m_pool = VK_NULL_HANDLE; m_setLayout = VK_NULL_HANDLE;
    m_framePool = VK_NULL_HANDLE; m_frameSetLayout = VK_NULL_HANDLE;
    m_smPool = VK_NULL_HANDLE; m_smSetLayout = VK_NULL_HANDLE;
    m_vsmBlurPool = VK_NULL_HANDLE; m_vsmBlurSetLayout = VK_NULL_HANDLE;
    if (m_rtPool)     vkDestroyDescriptorPool(dev, m_rtPool, nullptr);
    if (m_rtSetLayout) vkDestroyDescriptorSetLayout(dev, m_rtSetLayout, nullptr);
    m_rtPool = VK_NULL_HANDLE; m_rtSetLayout = VK_NULL_HANDLE;
    m_shadowSampler = VK_NULL_HANDLE;
    m_vsmSampler    = VK_NULL_HANDLE;
    m_shadowMap.reset();
    m_vsmMap.reset();
    m_shadowMask.reset();
    m_vsmBlur.reset();
    m_shadowUbo.reset();
    m_device = nullptr;
}

void ShadowPass::bindScene(Device& d, const SceneGpu& gpu) {
    // Save index buffer for vkCmdBindIndexBuffer
    m_indexBuffer = gpu.indexBuffer.handle();

    // 绑定 GPU-driven SSBO：gVertices[1], gIndices[2], gDrawData[3]
    VkDescriptorBufferInfo vb{gpu.vertexBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo ib{gpu.indexBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dd{gpu.drawDataBuffer.handle(), 0, VK_WHOLE_SIZE};

    std::array<VkWriteDescriptorSet, 3> w{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_smSet; w[0].dstBinding = 1; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &vb;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[1].dstSet = m_smSet; w[1].dstBinding = 2; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &ib;
    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[2].dstSet = m_smSet; w[2].dstBinding = 3; w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[2].pBufferInfo = &dd;

    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void ShadowPass::bindFrameResources(Device& d, VkBuffer frameUbo, VkImageView depthView, VkImageView normalView) {
    // ── set=1: FrameUniforms + gDepth ──
    {
        VkDescriptorBufferInfo uboInfo{frameUbo, 0, VK_WHOLE_SIZE};
        VkDescriptorImageInfo depthInfo{};
        depthInfo.imageView = depthView;
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        std::array<VkWriteDescriptorSet, 2> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = m_frameSet; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &uboInfo;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = m_frameSet; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[1].pImageInfo = &depthInfo;
        vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
    }

    // ── RT set binding 2: gNormalRough ──
    if (m_rtSet) {
        VkDescriptorImageInfo normalInfo{};
        normalInfo.imageView   = normalView;
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = m_rtSet; w.dstBinding = 2; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w.pImageInfo = &normalInfo;
        vkUpdateDescriptorSets(d.device(), 1, &w, 0, nullptr);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 每帧录制 —— 按 m_method 分发
// ────────────────────────────────────────────────────────────────────────────

void ShadowPass::record(VkCommandBuffer cmd, const RenderTargets& rt,
                         VkBuffer frameUbo, const SceneGpu& sceneGpu,
                         VkBuffer indirectBuf, uint32_t drawCount, uint32_t frameIndex) {
    m_currentFrameIndex = frameIndex;
    switch (m_method) {
    case ShadowMethod::None:          recordNone(cmd); break;
    case ShadowMethod::HardShadowMap: recordHardSM(cmd, rt, frameUbo, sceneGpu, indirectBuf, drawCount); break;
    case ShadowMethod::PCF:           recordPCF(cmd, rt, frameUbo, sceneGpu, indirectBuf, drawCount); break;
    case ShadowMethod::PCSS:          recordPCSS(cmd, rt, frameUbo, sceneGpu, indirectBuf, drawCount); break;
    case ShadowMethod::VSM:           recordVSM(cmd, rt, frameUbo, sceneGpu, indirectBuf, drawCount); break;
    case ShadowMethod::RTHard:        recordRTHard(cmd); break;
    case ShadowMethod::RTSoft:        recordRTSoft(cmd); break;
    default:                          recordNone(cmd); break;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 计算 sun view-proj 矩阵并写入 m_shadowUbo
// ────────────────────────────────────────────────────────────────────────────

namespace {
void computeSunViewProj(const glm::vec3& sunDir,
                         const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                         void* mappedUbo) {
    glm::vec3 lightDir = glm::normalize(sunDir);
    glm::vec3 toSun    = -lightDir;
    glm::vec3 sceneCenter = (aabbMin + aabbMax) * 0.5f;
    glm::vec3 sceneSize   = aabbMax - aabbMin;
    float diag = glm::length(sceneSize);

    // 把相机放在场景朝太阳方向之外足够远（=场景对角线长）
    glm::vec3 sunPos = sceneCenter + toSun * diag;

    // lookAt：从 sunPos 看向 sceneCenter
    glm::vec3 up = (std::abs(toSun.y) < 0.999f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    glm::mat4 view = glm::lookAt(sunPos, sceneCenter, up);

    // 将 AABB 8 个角点变换到 view space 以求紧凑的 ortho 投影
    glm::vec3 mn{ std::numeric_limits<float>::max() };
    glm::vec3 mx{ -std::numeric_limits<float>::max() };
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

    // Vulkan NDC: z ∈ [0,1]。view space Z 指向 -Z（相机看向 -Z），
    // 所以远 = -mx.z，近 = -mn.z
    glm::mat4 proj = glm::ortho(mn.x, mx.x, mn.y, mx.y, -mx.z, -mn.z);
    proj[1][1] *= -1.0f;   // Vulkan Y-flip

    ShadowUbo ubo{};
    ubo.sunViewProj = proj * view;
    std::memcpy(mappedUbo, &ubo, sizeof(ubo));
}
} // anonymous namespace

// ────────────────────────────────────────────────────────────────────────────
// recordNone —— 将 shadowMask 清为全白（1.0f = 无阴影）
// ────────────────────────────────────────────────────────────────────────────

void ShadowPass::recordNone(VkCommandBuffer cmd) {
    // 转换 shadowMask 到 GENERAL layout（vkCmdClearColorImage 要求）
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_CLEAR_BIT, 0);

    VkClearColorValue clearVal{};
    clearVal.float32[0] = 1.0f;
    clearVal.float32[1] = 1.0f;
    clearVal.float32[2] = 1.0f;
    clearVal.float32[3] = 1.0f;

    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_shadowMask.image(), VK_IMAGE_LAYOUT_GENERAL,
                         &clearVal, 1, &range);

    // 转换到 SHADER_READ_ONLY，供 LightingPass 采样
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_CLEAR_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

// ────────────────────────────────────────────────────────────────────────────
// recordHardSM —— 渲染 shadow map + Hard SM resolve
// ────────────────────────────────────────────────────────────────────────────

void ShadowPass::recordHardSM(VkCommandBuffer cmd, const RenderTargets& /*rt*/,
                               VkBuffer frameUbo, const SceneGpu& sceneGpu,
                               VkBuffer indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) { recordNone(cmd); return; }

    // 1. 渲染 shadow map（depth-only）
    renderShadowMap(cmd, frameUbo, sceneGpu, indirectBuf, drawCount);

    // 2a. 恢复 resolve descriptor（VSM 可能已改 binding 1/2）
    {
        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageView   = m_shadowMap.view();
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo smpInfo{};
        smpInfo.sampler = m_shadowSampler;

        std::array<VkWriteDescriptorSet, 2> dw{};
        dw[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        dw[0].dstSet = m_set; dw[0].dstBinding = 1; dw[0].descriptorCount = 1;
        dw[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; dw[0].pImageInfo = &imgInfo;
        dw[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        dw[1].dstSet = m_set; dw[1].dstBinding = 2; dw[1].descriptorCount = 1;
        dw[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; dw[1].pImageInfo = &smpInfo;
        vkUpdateDescriptorSets(m_device->device(), (uint32_t)dw.size(), dw.data(), 0, nullptr);
    }

    // 2b. 转换 shadowMask 到 GENERAL（storage image write）
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // 3. Resolve dispatch
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolveHard);
    {
        std::array<VkDescriptorSet, 2> dsets{m_set, m_frameSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_resolveLayout, 0, (uint32_t)dsets.size(), dsets.data(), 0, nullptr);
    }

    ResolvePC pc{};
    pc.outSizeX       = m_outputSize.width;
    pc.outSizeY       = m_outputSize.height;
    pc.invOutSizeX    = 1.0f / (float)m_outputSize.width;
    pc.invOutSizeY    = 1.0f / (float)m_outputSize.height;
    pc.bias           = 0.001f;
    pc._pad           = 0;
    pc.invShadowMapX  = 1.0f / (float)m_shadowMapSize.width;
    pc.invShadowMapY  = 1.0f / (float)m_shadowMapSize.height;
    vkCmdPushConstants(cmd, m_resolveLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t gx = (m_outputSize.width  + 7) / 8;
    uint32_t gy = (m_outputSize.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    // 4. 转换 shadowMask → SHADER_READ_ONLY
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

// ────────────────────────────────────────────────────────────────────────────
// recordPCF —— 渲染 shadow map + PCF resolve（与 HardSM 共用同一张 shadow map）
// ────────────────────────────────────────────────────────────────────────────

void ShadowPass::recordPCF(VkCommandBuffer cmd, const RenderTargets& /*rt*/,
                            VkBuffer frameUbo, const SceneGpu& sceneGpu,
                            VkBuffer indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) { recordNone(cmd); return; }

    // 1. 渲染 shadow map（与 HardSM 共用同一张）
    renderShadowMap(cmd, frameUbo, sceneGpu, indirectBuf, drawCount);

    // 2. Resolve
    //    恢复 resolve descriptor：VSM 可能已改 binding 1/2，PCF 需要 shadowMap + compare sampler
    {
        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageView   = m_shadowMap.view();
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo smpInfo{};
        smpInfo.sampler = m_shadowSampler;

        std::array<VkWriteDescriptorSet, 2> dw{};
        dw[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        dw[0].dstSet = m_set; dw[0].dstBinding = 1; dw[0].descriptorCount = 1;
        dw[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; dw[0].pImageInfo = &imgInfo;
        dw[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        dw[1].dstSet = m_set; dw[1].dstBinding = 2; dw[1].descriptorCount = 1;
        dw[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; dw[1].pImageInfo = &smpInfo;
        vkUpdateDescriptorSets(m_device->device(), (uint32_t)dw.size(), dw.data(), 0, nullptr);
    }

    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePCF);
    {
        std::array<VkDescriptorSet, 2> dsets{m_set, m_frameSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_resolveLayout, 0, (uint32_t)dsets.size(), dsets.data(), 0, nullptr);
    }

    ResolvePC pc{};
    pc.outSizeX       = m_outputSize.width;
    pc.outSizeY       = m_outputSize.height;
    pc.invOutSizeX    = 1.0f / (float)m_outputSize.width;
    pc.invOutSizeY    = 1.0f / (float)m_outputSize.height;
    pc.bias           = 0.001f;
    pc._pad           = 0;
    pc.invShadowMapX  = 1.0f / (float)m_shadowMapSize.width;
    pc.invShadowMapY  = 1.0f / (float)m_shadowMapSize.height;
    vkCmdPushConstants(cmd, m_resolveLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t gx = (m_outputSize.width  + 7) / 8;
    uint32_t gy = (m_outputSize.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

// ────────────────────────────────────────────────────────────────────────────
// recordPCSS —— 渲染 shadow map + PCSS resolve（blocker search + 自适应 PCF）
// ────────────────────────────────────────────────────────────────────────────

void ShadowPass::recordPCSS(VkCommandBuffer cmd, const RenderTargets& /*rt*/,
                             VkBuffer frameUbo, const SceneGpu& sceneGpu,
                             VkBuffer indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) { recordNone(cmd); return; }

    // 1. 渲染 shadow map（与 HardSM/PCF 共用同一张）
    renderShadowMap(cmd, frameUbo, sceneGpu, indirectBuf, drawCount);

    // 2. 恢复 resolve descriptor（VSM 可能已改 binding 1/2）
    {
        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageView   = m_shadowMap.view();
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo smpInfo{};
        smpInfo.sampler = m_shadowSampler;

        std::array<VkWriteDescriptorSet, 2> dw{};
        dw[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        dw[0].dstSet = m_set; dw[0].dstBinding = 1; dw[0].descriptorCount = 1;
        dw[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; dw[0].pImageInfo = &imgInfo;
        dw[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        dw[1].dstSet = m_set; dw[1].dstBinding = 2; dw[1].descriptorCount = 1;
        dw[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; dw[1].pImageInfo = &smpInfo;
        vkUpdateDescriptorSets(m_device->device(), (uint32_t)dw.size(), dw.data(), 0, nullptr);
    }

    // 3. Resolve
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePCSS);
    {
        std::array<VkDescriptorSet, 2> dsets{m_set, m_frameSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pcssResolveLayout, 0, (uint32_t)dsets.size(), dsets.data(), 0, nullptr);
    }

    PCSS_PC pc{};
    pc.outSizeX       = m_outputSize.width;
    pc.outSizeY       = m_outputSize.height;
    pc.invOutSizeX    = 1.0f / (float)m_outputSize.width;
    pc.invOutSizeY    = 1.0f / (float)m_outputSize.height;
    pc.bias           = 0.001f;
    pc._pad           = 0;
    pc.invShadowMapX  = 1.0f / (float)m_shadowMapSize.width;
    pc.invShadowMapY  = 1.0f / (float)m_shadowMapSize.height;
    pc.lightSize      = 0.03f;          // 太阳角半径 ~1.7°
    pc.maxKernelRadius = 15.0f;         // 最大 kernel 半径（texels）
    vkCmdPushConstants(cmd, m_pcssResolveLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t gx = (m_outputSize.width  + 7) / 8;
    uint32_t gy = (m_outputSize.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

// ────────────────────────────────────────────────────────────────────────────
// recordVSM —— 渲染 depth+depth^2 + VSM resolve
// ────────────────────────────────────────────────────────────────────────────

void ShadowPass::recordVSM(VkCommandBuffer cmd, const RenderTargets& /*rt*/,
                            VkBuffer frameUbo, const SceneGpu& sceneGpu,
                            VkBuffer indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) { recordNone(cmd); return; }

    // ── 1. 计算 sun view-proj ──
    computeSunViewProj(m_sunDir, m_sceneAabbMin, m_sceneAabbMax,
                       m_shadowUbo.mapped());

    // ── 2. 布局转换 ──
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

    // ── 3. 动态渲染：1 color (vsmMap) + depth (shadowMap) ──
    {
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView   = m_vsmMap.view();
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        // 关键：VSM clear 值必须是 (1,1)。天空/无几何区域 depth=1（远平面），
        // 确保 receiverDepth < 1 → lit。用 0 会导致所有天空区域被错误判定为阴影。
        color.clearValue.color = {{1.0f, 1.0f, 0.0f, 0.0f}};

        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView   = m_shadowMap.view();
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, m_shadowMapSize};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments    = &color;
        ri.pDepthAttachment     = &depth;
        vkCmdBeginRendering(cmd, &ri);

        VkViewport vp{0, 0, (float)m_shadowMapSize.width, (float)m_shadowMapSize.height, 0, 1};
        VkRect2D sc{{0, 0}, m_shadowMapSize};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_vsmGenPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_smPipelineLayout, 0, 1, &m_smSet, 0, nullptr);

        vkCmdBindIndexBuffer(cmd, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // GPU-driven indirect draw: loop per-draw (multiDrawIndirect not enabled)
        for (uint32_t d = 0; d < drawCount; ++d) {
            VkDeviceSize offset = 0;
            if (d > 0) offset = d * sizeof(VkDrawIndexedIndirectCommand);
            vkCmdDrawIndexedIndirect(cmd, indirectBuf, offset, 1,
                                      sizeof(VkDrawIndexedIndirectCommand));
        }
        vkCmdEndRendering(cmd);
    }

    // ── 4. 可分離 blur：水平 pass → 垂直 pass（5-tap gaussian）──
    //     共享内存 blur pipeline，pass 1 vsmMap→vsmBlur，pass 2 vsmBlur→vsmMap
    //     最终 vsmMap 包含两次滤波后的 moments，resolve 直接读 vsmMap

    auto blurDispatch = [&](VkImageView inView, VkImageLayout inLayout,
                            VkImageView outView, VkImageLayout outLayout,
                            uint32_t dir) {
        // 更新 blur descriptor
        VkDescriptorImageInfo inInfo{};
        inInfo.imageView   = inView;
        inInfo.imageLayout = inLayout;
        VkDescriptorImageInfo outInfo{};
        outInfo.imageView   = outView;
        outInfo.imageLayout = outLayout;
        std::array<VkWriteDescriptorSet, 2> wd{};
        wd[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wd[0].dstSet = m_vsmBlurSet; wd[0].dstBinding = 0; wd[0].descriptorCount = 1;
        wd[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; wd[0].pImageInfo = &inInfo;
        wd[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wd[1].dstSet = m_vsmBlurSet; wd[1].dstBinding = 1; wd[1].descriptorCount = 1;
        wd[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wd[1].pImageInfo = &outInfo;
        vkUpdateDescriptorSets(m_device->device(), (uint32_t)wd.size(), wd.data(), 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_vsmBlurPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_vsmBlurLayout, 0, 1, &m_vsmBlurSet, 0, nullptr);

        struct { uint32_t x, y; float ix, iy; uint32_t dir, pad; } pc;
        pc.x   = m_shadowMapSize.width;  pc.y   = m_shadowMapSize.height;
        pc.ix  = 1.0f / (float)pc.x;     pc.iy  = 1.0f / (float)pc.y;
        pc.dir = dir;                     pc.pad = 0;
        vkCmdPushConstants(cmd, m_vsmBlurLayout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(pc), &pc);

        uint32_t gx = (m_shadowMapSize.width  + 7) / 8;
        uint32_t gy = (m_shadowMapSize.height + 7) / 8;
        vkCmdDispatch(cmd, gx, gy, 1);
    };

    // 4a. Pass 1 布局：vsmMap → SHADER_READ_ONLY（blur 输入），vsmBlur → GENERAL（输出）
    transitionImage(cmd, m_vsmMap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    transitionImage(cmd, m_vsmBlur.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // 4b. Pass 1: 水平 blur（vsmMap → vsmBlur）
    blurDispatch(m_vsmMap.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 m_vsmBlur.view(), VK_IMAGE_LAYOUT_GENERAL, 0);

    // 4c. Pass 2 布局：vsmBlur → SHADER_READ_ONLY，vsmMap → GENERAL
    transitionImage(cmd, m_vsmBlur.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    transitionImage(cmd, m_vsmMap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // 4d. Pass 2: 垂直 blur（vsmBlur → vsmMap，覆盖原始数据）
    blurDispatch(m_vsmBlur.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 m_vsmMap.view(), VK_IMAGE_LAYOUT_GENERAL, 1);

    // 4e. vsmMap → SHADER_READ_ONLY（resolve 输入）
    transitionImage(cmd, m_vsmMap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    // 4f. shadowMap → SHADER_READ_ONLY
    transitionImage(cmd, m_shadowMap.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    // ── 5. 更新 resolve descriptor: binding 1 → vsmMap（两次 blur 后）, binding 2 → vsmSampler ──
    {
        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageView   = m_vsmMap.view();
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo smpInfo{};
        smpInfo.sampler = m_vsmSampler;

        std::array<VkWriteDescriptorSet, 2> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = m_set; w[0].dstBinding = 1; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &imgInfo;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = m_set; w[1].dstBinding = 2; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[1].pImageInfo = &smpInfo;
        vkUpdateDescriptorSets(m_device->device(), (uint32_t)w.size(), w.data(), 0, nullptr);
    }

    // ── 6. VSM resolve dispatch ──
    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolveVSM);
    {
        std::array<VkDescriptorSet, 2> dsets{m_set, m_frameSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_resolveLayout, 0, (uint32_t)dsets.size(), dsets.data(), 0, nullptr);
    }

    {
        ResolvePC pc{};
        pc.outSizeX       = m_outputSize.width;
        pc.outSizeY       = m_outputSize.height;
        pc.invOutSizeX    = 1.0f / (float)m_outputSize.width;
        pc.invOutSizeY    = 1.0f / (float)m_outputSize.height;
        pc.bias           = 0.001f;
        pc._pad           = 0;
        pc.invShadowMapX  = 1.0f / (float)m_shadowMapSize.width;
        pc.invShadowMapY  = 1.0f / (float)m_shadowMapSize.height;
        vkCmdPushConstants(cmd, m_resolveLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    }

    {
        uint32_t gx = (m_outputSize.width  + 7) / 8;
        uint32_t gy = (m_outputSize.height + 7) / 8;
        vkCmdDispatch(cmd, gx, gy, 1);
    }

    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

// ────────────────────────────────────────────────────────────────────────────
// recordRTHard —— RT 硬阴影：每像素 1 条 shadow ray 向太阳
// ────────────────────────────────────────────────────────────────────────────

void ShadowPass::recordRTHard(VkCommandBuffer cmd) {
    if (!m_rtHardPipeline) return;

    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtHardPipeline);
    {
        std::array<VkDescriptorSet, 2> dsets{m_rtSet, m_frameSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_rtHardLayout, 0, (uint32_t)dsets.size(), dsets.data(), 0, nullptr);
    }

    struct { uint32_t x, y; float ix, iy; } pc;
    pc.x  = m_outputSize.width;  pc.y  = m_outputSize.height;
    pc.ix = 1.0f / (float)pc.x;  pc.iy = 1.0f / (float)pc.y;
    vkCmdPushConstants(cmd, m_rtHardLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t gx = (m_outputSize.width  + 7) / 8;
    uint32_t gy = (m_outputSize.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

// ────────────────────────────────────────────────────────────────────────────
// recordRTSoft —— RT 软阴影：每像素 8 rays 锥体采样模拟面光源
// ────────────────────────────────────────────────────────────────────────────

void ShadowPass::recordRTSoft(VkCommandBuffer cmd) {
    if (!m_rtSoftPipeline) { recordRTHard(cmd); return; }

    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtSoftPipeline);
    {
        std::array<VkDescriptorSet, 2> dsets{m_rtSet, m_frameSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_rtSoftLayout, 0, (uint32_t)dsets.size(), dsets.data(), 0, nullptr);
    }

    struct { uint32_t x, y; float ix, iy; uint32_t fi; float sr; uint32_t rc; uint32_t pad; } pc;
    pc.x  = m_outputSize.width;  pc.y  = m_outputSize.height;
    pc.ix = 1.0f / (float)pc.x;  pc.iy = 1.0f / (float)pc.y;
    pc.fi = m_currentFrameIndex;
    pc.sr = m_rtSunRadius;
    pc.rc = (uint32_t)m_rtRayCount;
    pc.pad = 0;
    vkCmdPushConstants(cmd, m_rtSoftLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t gx = (m_outputSize.width  + 7) / 8;
    uint32_t gy = (m_outputSize.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    transitionImage(cmd, m_shadowMask.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

// ────────────────────────────────────────────────────────────────────────────
// renderShadowMap —— 从 sun 视角渲染深度图（GPU-driven indirect draw）
// ────────────────────────────────────────────────────────────────────────────

void ShadowPass::renderShadowMap(VkCommandBuffer cmd, VkBuffer /*frameUbo*/,
                                  const SceneGpu& /*sceneGpu*/,
                                  VkBuffer indirectBuf, uint32_t drawCount) {
    if (drawCount == 0) return;

    // ── 1. 计算太阳视角 view-proj 并写入 m_shadowUbo ──
    computeSunViewProj(m_sunDir, m_sceneAabbMin, m_sceneAabbMax,
                       m_shadowUbo.mapped());

    // ── 2. 布局转换 ──
    transitionImage(cmd, m_shadowMap.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    // ── 3. 动态渲染：depth-only ──
    {
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth.imageView   = m_shadowMap.view();
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, m_shadowMapSize};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 0;
        ri.pColorAttachments    = nullptr;
        ri.pDepthAttachment     = &depth;
        vkCmdBeginRendering(cmd, &ri);

        VkViewport vp{0, 0, (float)m_shadowMapSize.width, (float)m_shadowMapSize.height, 0, 1};
        VkRect2D sc{{0, 0}, m_shadowMapSize};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_smPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_smPipelineLayout, 0, 1, &m_smSet, 0, nullptr);

        vkCmdBindIndexBuffer(cmd, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // GPU-driven indirect draw: loop per-draw (multiDrawIndirect not enabled)
        for (uint32_t d = 0; d < drawCount; ++d) {
            VkDeviceSize offset = 0;
            if (d > 0) offset = d * sizeof(VkDrawIndexedIndirectCommand);
            vkCmdDrawIndexedIndirect(cmd, indirectBuf, offset, 1,
                                      sizeof(VkDrawIndexedIndirectCommand));
        }

        vkCmdEndRendering(cmd);
    }

    // ── 4. 转换到 SHADER_READ_ONLY，供 resolve compute shader 采样 ──
    transitionImage(cmd, m_shadowMap.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

// ────────────────────────────────────────────────────────────────────────────
// Pipeline 构建
// ────────────────────────────────────────────────────────────────────────────

void ShadowPass::buildPipeline_HardSM() {
    auto& d = *m_device;

    // Pipeline layout: m_smSetLayout（UBO + SSBO×3）
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_smSetLayout;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_smPipelineLayout));

    // Single .slang file with both vs_main and ps_main entry points
    ShaderModule smShader(d, shaderDir() / "shadow" / "shadow_hard.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = smShader.handle(); stages[0].pName = "vs_main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = smShader.handle(); stages[1].pName = "ps_main";

    // GPU-driven draw：无传统 vertex input（VS 直接从 SSBO 读顶点）
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 0;
    vi.vertexAttributeDescriptionCount = 0;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    // Sun ortho 视角：背面剔除 + 主相机一致的 front face winding
    rs.cullMode  = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    // 硬件深度偏移：消除 shadow acne（自阴影条纹）
    // constant factor 补偿精度量化误差，slope factor 补偿表面倾斜
    rs.depthBiasEnable         = VK_TRUE;
    rs.depthBiasConstantFactor = 4.0f;
    rs.depthBiasSlopeFactor    = 2.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    // No color attachments（depth-only rendering）
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 0; cb.pAttachments = nullptr;

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyni{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyni.dynamicStateCount = 2; dyni.pDynamicStates = dyn;

    // Dynamic rendering: depth-only, no color attachments
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 0;
    rci.pColorAttachmentFormats = nullptr;
    rci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.pNext = &rci;
    gpci.stageCount = 2; gpci.pStages = stages;
    gpci.pVertexInputState = &vi; gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp; gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms; gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb; gpci.pDynamicState = &dyni;
    gpci.layout = m_smPipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(d.device(), VK_NULL_HANDLE, 1, &gpci, nullptr, &m_smPipeline));
}

void ShadowPass::buildPipeline_VSMGen() {
    auto& d = *m_device;

    // 复用 m_smPipelineLayout（已在 buildPipeline_HardSM 中创建）
    if (m_smPipelineLayout == VK_NULL_HANDLE) {
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1; plci.pSetLayouts = &m_smSetLayout;
        VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_smPipelineLayout));
    }

    // Single .slang file with both vs_main and ps_main entry points
    ShaderModule smShader(d, shaderDir() / "shadow" / "shadow_vsm.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = smShader.handle(); stages[0].pName = "vs_main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = smShader.handle(); stages[1].pName = "ps_main";

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 0;
    vi.vertexAttributeDescriptionCount = 0;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode  = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    // 硬件深度偏移：消除 shadow acne（自阴影条纹）
    rs.depthBiasEnable         = VK_TRUE;
    rs.depthBiasConstantFactor = 4.0f;
    rs.depthBiasSlopeFactor    = 2.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    // 1 color attachment（R32G32_SFLOAT: depth + depth^2）
    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = 0xF;  // RG 全写
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &ba;

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyni{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyni.dynamicStateCount = 2; dyni.pDynamicStates = dyn;

    VkFormat colorFmt = VK_FORMAT_R32G32_SFLOAT;
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &colorFmt;
    rci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.pNext = &rci;
    gpci.stageCount = 2; gpci.pStages = stages;
    gpci.pVertexInputState = &vi; gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp; gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms; gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb; gpci.pDynamicState = &dyni;
    gpci.layout = m_smPipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(d.device(), VK_NULL_HANDLE, 1, &gpci, nullptr, &m_vsmGenPipeline));
}

// VSM 2×2 box blur compute pipeline：vsmMap → vsmBlur
void ShadowPass::buildPipeline_VSMBlur() {
    auto& d = *m_device;

    // Push constant: uint2 size + float2 invSize + uint dir + uint _pad = 24 bytes
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size       = sizeof(uint32_t) * 6;  // 24 bytes

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_vsmBlurSetLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_vsmBlurLayout));

    ShaderModule cs(d, shaderDir() / "shadow" / "shadow_vsm_blur.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs.handle();
    stage.pName  = "cs_main";

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage  = stage;
    cpci.layout = m_vsmBlurLayout;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_vsmBlurPipeline));
}

void ShadowPass::buildResolvePipeline() {
    auto& d = *m_device;

    // ── Resolve pipeline layout ──
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size       = sizeof(ResolvePC);

    std::array<VkDescriptorSetLayout, 2> sets{m_setLayout, m_frameSetLayout};

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = (uint32_t)sets.size(); plci.pSetLayouts = sets.data();
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_resolveLayout));

    // 注意：以下 ShaderModule() 构造会在 .spv 文件不存在时抛异常。

    // ── Hard SM resolve ──
    {
        ShaderModule cs(d, shaderDir() / "shadow" / "shadow_hard_resolve.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = cs.handle();
        stage.pName  = "cs_main";

        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage  = stage;
        cpci.layout = m_resolveLayout;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_resolveHard));
    }

    // ── PCF resolve ──
    {
        ShaderModule cs(d, shaderDir() / "shadow" / "shadow_pcf_resolve.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = cs.handle();
        stage.pName  = "cs_main";

        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage  = stage;
        cpci.layout = m_resolveLayout;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_resolvePCF));
    }

    // ── VSM resolve ──
    {
        ShaderModule cs(d, shaderDir() / "shadow" / "shadow_vsm_resolve.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = cs.handle();
        stage.pName  = "cs_main";

        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage  = stage;
        cpci.layout = m_resolveLayout;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_resolveVSM));
    }

    // ── PCSS resolve（独立 pipeline layout，push constant 比 ResolvePC 大）──
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size       = sizeof(PCSS_PC);    // 40 bytes：含 lightSize + maxKernelRadius
        static_assert(sizeof(PCSS_PC) == 40);

        std::array<VkDescriptorSetLayout, 2> sets{m_setLayout, m_frameSetLayout};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = (uint32_t)sets.size(); plci.pSetLayouts = sets.data();
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pcssResolveLayout));

        ShaderModule cs(d, shaderDir() / "shadow" / "shadow_pcss_resolve.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = cs.handle();
        stage.pName  = "cs_main";

        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage  = stage;
        cpci.layout = m_pcssResolveLayout;
        VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_resolvePCSS));
    }
}

void ShadowPass::destroyPipelines() {
    if (!m_device) return;
    auto dev = m_device->device();
    if (m_smPipeline)       vkDestroyPipeline(dev, m_smPipeline, nullptr);
    if (m_vsmGenPipeline)   vkDestroyPipeline(dev, m_vsmGenPipeline, nullptr);
    if (m_vsmBlurPipeline)  vkDestroyPipeline(dev, m_vsmBlurPipeline, nullptr);
    if (m_resolveHard)      vkDestroyPipeline(dev, m_resolveHard, nullptr);
    if (m_resolvePCF)       vkDestroyPipeline(dev, m_resolvePCF, nullptr);
    if (m_resolvePCSS)      vkDestroyPipeline(dev, m_resolvePCSS, nullptr);
    if (m_resolveVSM)       vkDestroyPipeline(dev, m_resolveVSM, nullptr);
    if (m_smPipelineLayout) vkDestroyPipelineLayout(dev, m_smPipelineLayout, nullptr);
    if (m_vsmBlurLayout)    vkDestroyPipelineLayout(dev, m_vsmBlurLayout, nullptr);
    if (m_pcssResolveLayout) vkDestroyPipelineLayout(dev, m_pcssResolveLayout, nullptr);
    if (m_rtHardLayout)     vkDestroyPipelineLayout(dev, m_rtHardLayout, nullptr);
    if (m_rtHardPipeline)   vkDestroyPipeline(dev, m_rtHardPipeline, nullptr);
    if (m_rtSoftLayout)     vkDestroyPipelineLayout(dev, m_rtSoftLayout, nullptr);
    if (m_rtSoftPipeline)   vkDestroyPipeline(dev, m_rtSoftPipeline, nullptr);
    if (m_resolveLayout)    vkDestroyPipelineLayout(dev, m_resolveLayout, nullptr);
    m_smPipeline       = VK_NULL_HANDLE;
    m_vsmGenPipeline   = VK_NULL_HANDLE;
    m_vsmBlurPipeline  = VK_NULL_HANDLE;
    m_resolveHard      = VK_NULL_HANDLE;
    m_resolvePCF       = VK_NULL_HANDLE;
    m_resolvePCSS      = VK_NULL_HANDLE;
    m_resolveVSM       = VK_NULL_HANDLE;
    m_smPipelineLayout = VK_NULL_HANDLE;
    m_vsmBlurLayout    = VK_NULL_HANDLE;
    m_pcssResolveLayout = VK_NULL_HANDLE;
    m_rtHardLayout      = VK_NULL_HANDLE;
    m_rtHardPipeline    = VK_NULL_HANDLE;
    m_rtSoftLayout      = VK_NULL_HANDLE;
    m_rtSoftPipeline    = VK_NULL_HANDLE;
    m_resolveLayout    = VK_NULL_HANDLE;
}

} // namespace somegi
