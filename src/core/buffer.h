#pragma once
#include "vk_common.h"

namespace somegi {

class Device;

class Buffer {
public:
    Buffer() = default;
    Buffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage,
           VkMemoryPropertyFlags memProps);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept { swap(o); }
    Buffer& operator=(Buffer&& o) noexcept { if (this != &o) { reset(); swap(o); } return *this; }

    void swap(Buffer& o) noexcept;
    void reset();

    VkBuffer handle() const { return m_buffer; }
    VkDeviceMemory memory() const { return m_memory; }
    VkDeviceSize size() const { return m_size; }
    VkDeviceAddress deviceAddress() const { return m_address; }
    void* mapped() const { return m_mapped; }

private:
    Device* m_device = nullptr;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkDeviceSize m_size = 0;
    VkDeviceAddress m_address = 0;
    void* m_mapped = nullptr;
};

}
