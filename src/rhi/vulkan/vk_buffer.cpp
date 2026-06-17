// rhi/vulkan/vk_buffer.cpp
#include "vk_buffer.h"
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
    ci.usage = toVkUsage(desc.usage);
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
    vmaCreateBuffer(device.vma(), &ci, &ai, &buf->m_buffer, &buf->m_allocation, nullptr);
    buf->m_size = desc.size;
    if (desc.usage == BufferUsage::Storage || desc.usage == BufferUsage::Indirect || (uint32_t)desc.usage & (uint32_t)BufferUsage::AccelStruct) {
        VkBufferDeviceAddressInfo ai2{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, buf->m_buffer};
        buf->m_address = vkGetBufferDeviceAddress(device.vkDevice(), &ai2);
    }
    if (ai.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) {
        vmaMapMemory(device.vma(), buf->m_allocation, &buf->m_mapped);
    }
    return buf;
}

VkRHIBuffer::~VkRHIBuffer() {
    if (m_buffer && m_ownsBuffer) vmaDestroyBuffer(m_device.vma(), m_buffer, m_allocation);
}
void* VkRHIBuffer::map() { return m_mapped; }
void VkRHIBuffer::unmap() { vmaFlushAllocation(m_device.vma(), m_allocation, 0, m_size); }
} // namespace rhi
} // namespace somegi
