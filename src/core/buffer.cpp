#include "buffer.h"
#include "device.h"
#include "rhi/vulkan/vk_device.h"
#include <utility>

namespace somegi {

Buffer::Buffer(rhi::VkRHIDevice& vkDev, VkDeviceSize size, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags memProps, VkDeviceSize alignment)
               : m_size(size) {
    m_vkDev = vkDev.vkDevice();
    m_vma = vkDev.vma();
    VkPhysicalDevice physDev = vkDev.vkPhysicalDevice();

    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.requiredFlags = memProps;
    if (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    if (alignment > 0) {
        VK_CHECK(vkCreateBuffer(m_vkDev, &ci, nullptr, &m_buffer));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(m_vkDev, m_buffer, &mr);
        if (alignment > mr.alignment) mr.alignment = alignment;
        VkMemoryAllocateFlagsInfo maiFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        maiFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &maiFlags;
        mai.allocationSize = mr.size;
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(physDev, &mp);
        uint32_t typeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((mr.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                typeIndex = i; break;
            }
        }
        mai.memoryTypeIndex = typeIndex;
        VkDeviceMemory mem;
        VK_CHECK(vkAllocateMemory(m_vkDev, &mai, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(m_vkDev, m_buffer, mem, 0));
        m_allocation = nullptr;
        m_manualMem = mem;
        m_mapped = nullptr;
    } else {
        VmaAllocationInfo ai{};
        VK_CHECK(vmaCreateBuffer(m_vma, &ci, &aci, &m_buffer, &m_allocation, &ai));
        m_mapped = ai.pMappedData;
    }

    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bdai.buffer = m_buffer;
    m_address = vkGetBufferDeviceAddress(m_vkDev, &bdai);
}

Buffer::Buffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags memProps, VkDeviceSize alignment)
               : m_size(size) {
    m_vkDev = d.device();
    m_vma = d.allocator();
    VkPhysicalDevice physDev = d.physicalDevice();

    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.requiredFlags = memProps;
    if (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    if (alignment > 0) {
        VK_CHECK(vkCreateBuffer(m_vkDev, &ci, nullptr, &m_buffer));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(m_vkDev, m_buffer, &mr);
        if (alignment > mr.alignment) mr.alignment = alignment;
        VkMemoryAllocateFlagsInfo maiFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        maiFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &maiFlags;
        mai.allocationSize = mr.size;
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(physDev, &mp);
        uint32_t typeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((mr.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                typeIndex = i; break;
            }
        }
        mai.memoryTypeIndex = typeIndex;
        VkDeviceMemory mem;
        VK_CHECK(vkAllocateMemory(m_vkDev, &mai, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(m_vkDev, m_buffer, mem, 0));
        m_allocation = nullptr;
        m_manualMem = mem;
        m_mapped = nullptr;
    } else {
        VmaAllocationInfo ai{};
        VK_CHECK(vmaCreateBuffer(m_vma, &ci, &aci, &m_buffer, &m_allocation, &ai));
        m_mapped = ai.pMappedData;
    }

    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bdai.buffer = m_buffer;
    m_address = vkGetBufferDeviceAddress(m_vkDev, &bdai);
}

Buffer::~Buffer() { reset(); }

void Buffer::swap(Buffer& o) noexcept {
    std::swap(m_vkDev, o.m_vkDev);
    std::swap(m_vma, o.m_vma);
    std::swap(m_buffer, o.m_buffer);
    std::swap(m_allocation, o.m_allocation);
    std::swap(m_size, o.m_size);
    std::swap(m_address, o.m_address);
    std::swap(m_mapped, o.m_mapped);
    std::swap(m_manualMem, o.m_manualMem);
}

void Buffer::reset() {
    if (m_vkDev && m_buffer) {
        if (m_manualMem) {
            vkDestroyBuffer(m_vkDev, m_buffer, nullptr);
            vkFreeMemory(m_vkDev, m_manualMem, nullptr);
            m_manualMem = VK_NULL_HANDLE;
        } else {
            vmaDestroyBuffer(m_vma, m_buffer, m_allocation);
        }
    }
    m_vkDev = VK_NULL_HANDLE;
    m_vma = VK_NULL_HANDLE;
    m_buffer = VK_NULL_HANDLE;
    m_allocation = VK_NULL_HANDLE;
    m_size = 0; m_address = 0; m_mapped = nullptr;
}

}
