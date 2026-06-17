// rhi/vulkan/vk_descriptor.cpp
#include "vk_descriptor.h"
#include "vk_buffer.h"
#include "vk_texture.h"

namespace somegi {
namespace rhi {

static VkDescriptorType toVkDescType(DescriptorType t) {
    switch (t) {
        case DescriptorType::SampledImage:          return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case DescriptorType::StorageImage:          return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case DescriptorType::UniformBuffer:         return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case DescriptorType::StorageBuffer:         return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        case DescriptorType::Sampler:               return VK_DESCRIPTOR_TYPE_SAMPLER;
        case DescriptorType::AccelerationStructure: return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        default: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    }
}

// ════════════════════════════════════════════════════════════════
// Descriptor Set Layout
// ════════════════════════════════════════════════════════════════
std::unique_ptr<RHIDescriptorSetLayout> VkRHIDescSetLayout::create(VkRHIDevice& device, const DescSetLayoutDesc& desc) {
    auto l = std::unique_ptr<VkRHIDescSetLayout>(new VkRHIDescSetLayout(device));
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (auto& b : desc.bindings) {
        VkDescriptorSetLayoutBinding lb{};
        lb.binding = b.binding;
        lb.descriptorType = toVkDescType(b.type);
        lb.descriptorCount = b.count;
        lb.stageFlags = VK_SHADER_STAGE_ALL;
        bindings.push_back(lb);
    }
    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = (uint32_t)bindings.size();
    ci.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device.vkDevice(), &ci, nullptr, &l->m_layout);
    return l;
}

VkRHIDescSetLayout::~VkRHIDescSetLayout() {
    if (m_layout) vkDestroyDescriptorSetLayout(m_device.vkDevice(), m_layout, nullptr);
}

// ════════════════════════════════════════════════════════════════
// Descriptor Set
// ════════════════════════════════════════════════════════════════
std::unique_ptr<RHIDescriptorSet> VkRHIDescSet::create(VkRHIDevice& device, const RHIDescriptorSetLayout& layout) {
    auto s = std::unique_ptr<VkRHIDescSet>(new VkRHIDescSet(device));

    // 创建 descriptor pool（涵盖所有可能的描述符类型）
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 16},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 4},
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4},
    };
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1;
    pci.poolSizeCount = 6;
    pci.pPoolSizes = poolSizes;
    vkCreateDescriptorPool(device.vkDevice(), &pci, nullptr, &s->m_pool);

    // 分配 descriptor set
    VkDescriptorSetLayout vkLayout = (VkDescriptorSetLayout)(uintptr_t)layout.nativeHandle();
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = s->m_pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &vkLayout;
    vkAllocateDescriptorSets(device.vkDevice(), &ai, &s->m_set);

    return s;
}

VkRHIDescSet::~VkRHIDescSet() {
    if (m_pool) vkDestroyDescriptorPool(m_device.vkDevice(), m_pool, nullptr);
}

void VkRHIDescSet::write(const std::vector<DescriptorWrite>& writes) {
    std::vector<VkWriteDescriptorSet> vkWrites;
    std::vector<VkDescriptorImageInfo> imageInfos;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkWriteDescriptorSetAccelerationStructureKHR> asInfos;

    // 预分配容量，防止 push_back 导致 reallocate 使已存入 vkWrites 的指针悬空
    vkWrites.reserve(writes.size());
    imageInfos.reserve(writes.size());
    bufferInfos.reserve(writes.size());
    asInfos.reserve(writes.size());

    for (auto& w : writes) {
        VkWriteDescriptorSet vw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        vw.dstSet = m_set;
        vw.dstBinding = w.binding;
        vw.descriptorCount = 1;
        vw.descriptorType = toVkDescType(w.type);

        if (w.textureView) {
            imageInfos.push_back({VK_NULL_HANDLE, (VkImageView)(uintptr_t)w.textureView->nativeHandle(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            vw.pImageInfo = &imageInfos.back();
        }
        if (w.sampler) {
            // 独立 Sampler 绑定（VK_DESCRIPTOR_TYPE_SAMPLER）
            imageInfos.push_back({(VkSampler)(uintptr_t)w.sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
            vw.pImageInfo = &imageInfos.back();
        }
        if (w.buffer) {
            if (w.type == DescriptorType::AccelerationStructure) {
                // TLAS 通过 buffer 的 nativeHandle 获取底层 VkAccelerationStructureKHR
                asInfos.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR});
                asInfos.back().accelerationStructureCount = 1;
                asInfos.back().pAccelerationStructures = (const VkAccelerationStructureKHR*)(uintptr_t)w.buffer->nativeHandle();
                vw.pNext = &asInfos.back();
            } else {
                bufferInfos.push_back({(VkBuffer)(uintptr_t)w.buffer->nativeHandle(), w.bufferOffset, w.bufferRange ? w.bufferRange : VK_WHOLE_SIZE});
                vw.pBufferInfo = &bufferInfos.back();
            }
        }
        vkWrites.push_back(vw);
    }
    vkUpdateDescriptorSets(m_device.vkDevice(), (uint32_t)vkWrites.size(), vkWrites.data(), 0, nullptr);
}

} // namespace rhi
} // namespace somegi
