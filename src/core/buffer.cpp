#include "buffer.h"
#include "device.h"
#include "allocator.h"
#include <utility>

namespace somegi {

Buffer::Buffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags memProps) : m_device(&d), m_size(size) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(d.device(), &ci, nullptr, &m_buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(d.device(), m_buffer, &req);

    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flags;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(d.physicalDevice(), req.memoryTypeBits, memProps);
    VK_CHECK(vkAllocateMemory(d.device(), &ai, nullptr, &m_memory));
    VK_CHECK(vkBindBufferMemory(d.device(), m_buffer, m_memory, 0));

    if (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        VK_CHECK(vkMapMemory(d.device(), m_memory, 0, size, 0, &m_mapped));
    }
    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bdai.buffer = m_buffer;
    m_address = vkGetBufferDeviceAddress(d.device(), &bdai);
}

Buffer::~Buffer() { reset(); }

void Buffer::swap(Buffer& o) noexcept {
    std::swap(m_device, o.m_device);
    std::swap(m_buffer, o.m_buffer);
    std::swap(m_memory, o.m_memory);
    std::swap(m_size, o.m_size);
    std::swap(m_address, o.m_address);
    std::swap(m_mapped, o.m_mapped);
}

void Buffer::reset() {
    if (m_device) {
        if (m_mapped) vkUnmapMemory(m_device->device(), m_memory);
        if (m_buffer) vkDestroyBuffer(m_device->device(), m_buffer, nullptr);
        if (m_memory) vkFreeMemory(m_device->device(), m_memory, nullptr);
    }
    m_device = nullptr;
    m_buffer = VK_NULL_HANDLE;
    m_memory = VK_NULL_HANDLE;
    m_size = 0; m_address = 0; m_mapped = nullptr;
}

}
