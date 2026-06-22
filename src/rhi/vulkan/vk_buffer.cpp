// rhi/vulkan/vk_buffer.cpp
#include "vk_buffer.h"
#include "vk_common.h"
#include <VulkanMemoryAllocator/vk_mem_alloc.h>

namespace somegi {
namespace rhi {

static VkBufferUsageFlags toVkUsage(BufferUsage u) {
    VkBufferUsageFlags f = 0;
    if ((uint32_t)u & (uint32_t)BufferUsage::Vertex)   f |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if ((uint32_t)u & (uint32_t)BufferUsage::Index)    f |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if ((uint32_t)u & (uint32_t)BufferUsage::Uniform)  f |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if ((uint32_t)u & (uint32_t)BufferUsage::Storage)  f |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if ((uint32_t)u & (uint32_t)BufferUsage::Indirect) f |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if ((uint32_t)u & (uint32_t)BufferUsage::TransferSrc) f |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if ((uint32_t)u & (uint32_t)BufferUsage::TransferDst) f |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if ((uint32_t)u & (uint32_t)BufferUsage::AccelStruct) f |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    return f;
}

std::unique_ptr<RHIBuffer> VkRHIBuffer::create(VkRHIDevice& device, const BufferDesc& desc) {
    auto buf = std::make_unique<VkRHIBuffer>(device);
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = desc.size;
    ci.usage = toVkUsage(desc.usage) | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (desc.alignment > 0) {
        // 手动对齐分配路径（用于 AS scratch buffer 等需要大对齐的场景）
        VK_CHECK(vkCreateBuffer(device.vkDevice(), &ci, nullptr, &buf->m_buffer));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(device.vkDevice(), buf->m_buffer, &mr);
        if (desc.alignment > mr.alignment) mr.alignment = desc.alignment;

        VkMemoryAllocateFlagsInfo maiFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        maiFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &maiFlags;
        mai.allocationSize = mr.size;

        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(device.vkPhysicalDevice(), &mp);
        uint32_t typeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((mr.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                typeIndex = i; break;
            }
        }
        mai.memoryTypeIndex = typeIndex;
        VkDeviceMemory mem;
        VK_CHECK(vkAllocateMemory(device.vkDevice(), &mai, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(device.vkDevice(), buf->m_buffer, mem, 0));
        buf->m_allocation = nullptr;   // 非 VMA 管理
        buf->m_manualMem = mem;        // 析构时手动释放
        buf->m_mapped = nullptr;
        buf->m_ownsBuffer = true;
    } else {
        VmaAllocationCreateInfo ai{};
        if (desc.memory == MemoryType::HostVisible) {
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        } else if (desc.memory == MemoryType::HostCached) {
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        } else {
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        }
        VmaAllocationInfo allocInfo{};
        vmaCreateBuffer(device.vma(), &ci, &ai, &buf->m_buffer, &buf->m_allocation, &allocInfo);
        // VMA_ALLOCATION_CREATE_MAPPED_BIT 已使 VMA 持久映射
        if (ai.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) {
            buf->m_mapped = allocInfo.pMappedData;
        }
    }

    buf->m_size = desc.size;
    VkBufferDeviceAddressInfo ai2{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, buf->m_buffer};
    buf->m_address = vkGetBufferDeviceAddress(device.vkDevice(), &ai2);
    return buf;
}

VkRHIBuffer::~VkRHIBuffer() {
    if (m_buffer && m_ownsBuffer) {
        if (m_manualMem) {
            vkDestroyBuffer(m_device.vkDevice(), m_buffer, nullptr);
            vkFreeMemory(m_device.vkDevice(), m_manualMem, nullptr);
        } else {
            vmaDestroyBuffer(m_device.vma(), m_buffer, m_allocation);
        }
    }
}
void* VkRHIBuffer::map() { return m_mapped; }
void VkRHIBuffer::unmap() { vmaFlushAllocation(m_device.vma(), m_allocation, 0, m_size); }
} // namespace rhi
} // namespace somegi
