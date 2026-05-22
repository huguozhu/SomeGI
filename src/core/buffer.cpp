#include "buffer.h"
#include "device.h"
#include <utility>

namespace somegi {

Buffer::Buffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags memProps) : m_device(&d), m_size(size) {
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

    VmaAllocationInfo ai{};
    VK_CHECK(vmaCreateBuffer(d.allocator(), &ci, &aci, &m_buffer, &m_allocation, &ai));
    m_mapped = ai.pMappedData;

    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bdai.buffer = m_buffer;
    m_address = vkGetBufferDeviceAddress(d.device(), &bdai);
}

Buffer::~Buffer() { reset(); }

void Buffer::swap(Buffer& o) noexcept {
    std::swap(m_device, o.m_device);
    std::swap(m_buffer, o.m_buffer);
    std::swap(m_allocation, o.m_allocation);
    std::swap(m_size, o.m_size);
    std::swap(m_address, o.m_address);
    std::swap(m_mapped, o.m_mapped);
}

void Buffer::reset() {
    if (m_device && m_buffer) {
        vmaDestroyBuffer(m_device->allocator(), m_buffer, m_allocation);
    }
    m_device = nullptr;
    m_buffer = VK_NULL_HANDLE;
    m_allocation = VK_NULL_HANDLE;
    m_size = 0; m_address = 0; m_mapped = nullptr;
}

}
