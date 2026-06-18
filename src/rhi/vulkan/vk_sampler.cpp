// rhi/vulkan/vk_sampler.cpp
#include "vk_sampler.h"

namespace somegi {
namespace rhi {

static VkFilter toVkFilter(Filter f) { return f == Filter::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR; }
static VkSamplerMipmapMode toVkMip(SamplerMipmapMode m) { return m == SamplerMipmapMode::Nearest ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR; }
static VkSamplerAddressMode toVkAddr(SamplerAddressMode a) {
    switch (a) {
        case SamplerAddressMode::Repeat:          return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case SamplerAddressMode::MirroredRepeat:  return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case SamplerAddressMode::ClampToBorder:   return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}
static VkCompareOp toVkCompare(CompareFunc c) {
    switch (c) {
        case CompareFunc::Never: return VK_COMPARE_OP_NEVER;
        case CompareFunc::Less: return VK_COMPARE_OP_LESS;
        case CompareFunc::Equal: return VK_COMPARE_OP_EQUAL;
        case CompareFunc::LessEqual: return VK_COMPARE_OP_LESS_OR_EQUAL;
        case CompareFunc::Greater: return VK_COMPARE_OP_GREATER;
        case CompareFunc::NotEqual: return VK_COMPARE_OP_NOT_EQUAL;
        case CompareFunc::GreaterEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case CompareFunc::Always: return VK_COMPARE_OP_ALWAYS;
    }
    return VK_COMPARE_OP_LESS_OR_EQUAL;
}

std::unique_ptr<RHISampler> VkRHISampler::create(VkRHIDevice& device, const SamplerDesc& desc) {
    auto s = std::unique_ptr<VkRHISampler>(new VkRHISampler(device));
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = toVkFilter(desc.magFilter);
    si.minFilter = toVkFilter(desc.minFilter);
    si.mipmapMode = toVkMip(desc.mipmapMode);
    si.addressModeU = toVkAddr(desc.addressU);
    si.addressModeV = toVkAddr(desc.addressV);
    si.addressModeW = toVkAddr(desc.addressW);
    si.maxLod = desc.maxLod > 0.f ? desc.maxLod : VK_LOD_CLAMP_NONE;
    si.compareEnable = desc.compareEnable ? VK_TRUE : VK_FALSE;
    si.compareOp = toVkCompare(desc.compareOp);
    vkCreateSampler(device.vkDevice(), &si, nullptr, &s->m_sampler);
    return s;
}

VkRHISampler::~VkRHISampler() {
    if (m_sampler && m_ownsSampler) vkDestroySampler(m_device.vkDevice(), m_sampler, nullptr);
}

} // namespace rhi
} // namespace somegi
