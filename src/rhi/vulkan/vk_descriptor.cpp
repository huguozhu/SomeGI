// rhi/vulkan/vk_descriptor.cpp
#include "vk_descriptor.h"
#include "vk_buffer.h"
#include "vk_texture.h"
#include <map>

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
    auto l = std::unique_ptr<VkRHIDescSetLayout>(new VkRHIDescSetLayout(device, desc));
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<VkDescriptorBindingFlags> bindingFlags;
    for (auto& b : desc.bindings) {
        VkDescriptorSetLayoutBinding lb{};
        lb.binding = b.binding;
        lb.descriptorType = toVkDescType(b.type);
        lb.descriptorCount = b.count;
        lb.stageFlags = VK_SHADER_STAGE_ALL;
        bindings.push_back(lb);
        VkDescriptorBindingFlags f = 0;
        if (b.partiallyBound) f |= VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        bindingFlags.push_back(f);
    }

    // UPDATE_AFTER_BIND 标志
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    for (auto bi : desc.updateAfterBindBindings)
        if (bi < bindingFlags.size()) bindingFlags[bi] |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    bool hasFlags = false;
    for (auto f : bindingFlags) if (f) { hasFlags = true; break; }
    if (hasFlags) {
        bfci.bindingCount = (uint32_t)bindingFlags.size();
        bfci.pBindingFlags = bindingFlags.data();
    }

    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.pNext = hasFlags ? &bfci : nullptr;
    ci.bindingCount = (uint32_t)bindings.size();
    ci.pBindings = bindings.data();
    if (desc.updateAfterBind)
        ci.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
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

    // 根据 layout 的 binding 描述动态计算 descriptor pool 容量
    auto& layoutDesc = static_cast<const VkRHIDescSetLayout&>(layout).desc();
    std::map<VkDescriptorType, uint32_t> typeCounts;
    for (auto& b : layoutDesc.bindings) {
        VkDescriptorType vt = toVkDescType(b.type);
        typeCounts[vt] += b.count;  // 累加同类型的所有 binding count
    }
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto& [vt, cnt] : typeCounts)
        poolSizes.push_back({vt, cnt});

    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1;
    pci.poolSizeCount = (uint32_t)poolSizes.size();
    pci.pPoolSizes = poolSizes.data();
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

    // 预计算总容量（含纹理数组），防止 resize/reallocate 导致指针悬空
    size_t totalImages = 0, totalBuffers = 0, totalAS = 0;
    for (auto& w : writes) {
        if (w.textureArrayCount > 0) totalImages += w.textureArrayCount;
        else if (w.textureView || w.sampler) totalImages += 1;
        if (w.buffer) totalBuffers += 1;
        if (w.accelerationStructure) totalAS += 1;
    }
    vkWrites.reserve(writes.size());
    imageInfos.reserve(totalImages);
    bufferInfos.reserve(totalBuffers);
    asInfos.reserve(totalAS);

    for (auto& w : writes) {
        VkWriteDescriptorSet vw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        vw.dstSet = m_set;
        vw.dstBinding = w.binding;
        vw.descriptorCount = 1;
        vw.descriptorType = toVkDescType(w.type);

        if (w.textureArrayCount > 0 && w.textureViewArray) {
            // 纹理数组绑定（如 GBuffer/Voxelize 的 texture array）
            vw.descriptorCount = w.textureArrayCount;
            size_t baseIdx = imageInfos.size();
            imageInfos.resize(baseIdx + w.textureArrayCount);
            for (uint32_t ai = 0; ai < w.textureArrayCount; ++ai) {
                VkImageView v = w.textureViewArray[ai]
                    ? (VkImageView)(uintptr_t)w.textureViewArray[ai]->nativeHandle()
                    : VK_NULL_HANDLE;
                imageInfos[baseIdx + ai] = {VK_NULL_HANDLE, v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            }
            vw.pImageInfo = &imageInfos[baseIdx];
        } else if (w.textureView) {
            imageInfos.push_back({VK_NULL_HANDLE, (VkImageView)(uintptr_t)w.textureView->nativeHandle(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            vw.pImageInfo = &imageInfos.back();
        }
        if (w.sampler) {
            // 独立 Sampler 绑定（VK_DESCRIPTOR_TYPE_SAMPLER）
            imageInfos.push_back({(VkSampler)(uintptr_t)w.sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
            vw.pImageInfo = &imageInfos.back();
        }
        if (w.accelerationStructure) {
            // TLAS 绑定（VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR）
            asInfos.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR});
            asInfos.back().accelerationStructureCount = 1;
            asInfos.back().pAccelerationStructures = (const VkAccelerationStructureKHR*)w.accelerationStructure;
            vw.pNext = &asInfos.back();
        }
        if (w.buffer) {
            bufferInfos.push_back({(VkBuffer)(uintptr_t)w.buffer->nativeHandle(), w.bufferOffset, w.bufferRange ? w.bufferRange : VK_WHOLE_SIZE});
            vw.pBufferInfo = &bufferInfos.back();
        }
        vkWrites.push_back(vw);
    }
    vkUpdateDescriptorSets(m_device.vkDevice(), (uint32_t)vkWrites.size(), vkWrites.data(), 0, nullptr);
}

} // namespace rhi
} // namespace somegi
