// IBLTechnique 实现 —— 封装 IBL 资源到 set=1 描述符。
// 资源本身的烘焙在 IblBaker 里完成；本类不持有 IblResources 的所有权，
// 只是借用 + 装到 descriptor set。

#include "ibl_technique.h"
#include "core/device.h"
#include <imgui.h>
#include <array>
#include <cstring>
#include <stdexcept>

namespace somegi {

namespace {
// set=1 binding 4 的 UBO 数据布局。slang 里同名 struct 必须保持一致。
// std140 要求 vec4 对齐 → intensity 单 float 后面要补 3 个 float padding。
struct IblParamsUbo {
    float intensity;
    float _pad0, _pad1, _pad2;
};
}

IBLTechnique::~IBLTechnique() { onDetach(); }

void IBLTechnique::onAttach(const GIContext& ctx) {
    m_device = ctx.device;
    if (!ctx.iblBaked) {
        throw std::runtime_error("IBLTechnique requires GIContext::iblBaked (pre-baked env)");
    }
    m_res = ctx.iblBaked;

    // set=1 描述符布局：3 张 sampled image + 1 个 sampler + 1 个 UBO。
    // 5 个 binding 对应 ibl_eval.slang 中的：
    //   binding 0: gIblDiffuse  (TextureCube)
    //   binding 1: gIblSpecular (TextureCube)
    //   binding 2: gIblBrdfLut  (Texture2D)
    //   binding 3: gIblSampler  (SamplerState, 共享给上面三张)
    //   binding 4: gIblParams   (ConstantBuffer<IblParams> 携带 intensity)
    //
    // stage flags 同时包含 FRAGMENT + COMPUTE：M3 forward_ibl.slang
    // (fragment) 与 M4 lighting.slang (compute) 都要绑这个 set，所以
    // 两个阶段都得允许。
    constexpr VkShaderStageFlags kIblStages =
        VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    std::array<VkDescriptorSetLayoutBinding, 5> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, kIblStages};  // diffuse cube
    b[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, kIblStages};  // specular cube
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  1, kIblStages};  // brdf lut
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLER,        1, kIblStages};  // shared linear
    b[4] = {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, kIblStages};  // params (intensity)

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(ctx.device->device(), &li, nullptr, &m_dsl));

    std::array<VkDescriptorPoolSize, 3> ps{{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  3},
        {VK_DESCRIPTOR_TYPE_SAMPLER,        1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(ctx.device->device(), &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_dsl;
    VK_CHECK(vkAllocateDescriptorSets(ctx.device->device(), &dai, &m_set));

    m_paramsUbo = Buffer(*ctx.device, sizeof(IblParamsUbo),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    buildSet();
    writeParams();
}

// 把 IblResources 中的 image view 和自己的 paramsUbo 写到 m_set。
// 调用时机：onAttach 末尾（一次）。set=1 在整个 IBLTechnique 生命周期
// 不变，无需逐帧重写。
void IBLTechnique::buildSet() {
    auto sample = [&](VkImageView v) {
        VkDescriptorImageInfo i{}; i.imageView = v;
        i.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; return i;
    };
    VkDescriptorImageInfo diffI = sample(m_res->diffuseCube.view());
    VkDescriptorImageInfo specI = sample(m_res->specularCube.view());
    VkDescriptorImageInfo lutI  = sample(m_res->brdfLut.view());
    VkDescriptorImageInfo smpI{}; smpI.sampler = m_res->linear;
    VkDescriptorBufferInfo uboI{m_paramsUbo.handle(), 0, VK_WHOLE_SIZE};

    std::array<VkWriteDescriptorSet, 5> w{};
    auto setImg = [&](VkWriteDescriptorSet& W, uint32_t bi, VkDescriptorType t, const VkDescriptorImageInfo* p) {
        W = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        W.dstSet = m_set; W.dstBinding = bi; W.descriptorCount = 1;
        W.descriptorType = t; W.pImageInfo = p;
    };
    setImg(w[0], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &diffI);
    setImg(w[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &specI);
    setImg(w[2], 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &lutI);
    setImg(w[3], 3, VK_DESCRIPTOR_TYPE_SAMPLER,       &smpI);
    w[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[4].dstSet = m_set; w[4].dstBinding = 4; w[4].descriptorCount = 1;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[4].pBufferInfo = &uboI;
    vkUpdateDescriptorSets(m_device->device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

// 把当前 m_intensity 写到 paramsUbo 的 mapped 内存。host-coherent 内存
// 不需要额外 flush。drawUI 里 slider 改变时调用一次（不是每帧调）。
void IBLTechnique::writeParams() {
    if (!m_paramsUbo.mapped()) return;
    IblParamsUbo u{};
    u.intensity = m_intensity;
    std::memcpy(m_paramsUbo.mapped(), &u, sizeof(u));
}

// 释放本 technique 拥有的资源；m_res 是借的不动它。
// 调用时机：析构 / GI 切换时 App 反射性地调用。
void IBLTechnique::onDetach() {
    if (!m_device) return;
    if (m_pool) vkDestroyDescriptorPool(m_device->device(), m_pool, nullptr);
    if (m_dsl)  vkDestroyDescriptorSetLayout(m_device->device(), m_dsl, nullptr);
    m_pool = VK_NULL_HANDLE; m_dsl = VK_NULL_HANDLE; m_set = VK_NULL_HANDLE;
    m_paramsUbo.reset();
    // m_res 是借用，所有权在 App，这里不释放。
    m_res = nullptr;
    m_device = nullptr;
}

// 在 App 的 GI 子面板里被调用。SliderFloat 在用户拖动时返回 true，
// 此时才把新值刷到 GPU；保持稳定时不写，避免无谓 memcpy。
void IBLTechnique::drawUI() {
    if (ImGui::SliderFloat("IBL intensity", &m_intensity, 0.0f, 4.0f)) {
        writeParams();
    }
    if (m_res) ImGui::Text("specular mips: %u", m_res->specularMipCount);
}

}
