#pragma once
#include "vk_common.h"

struct VmaAllocation_T;
typedef VmaAllocation_T* VmaAllocation;

namespace somegi {

class Device;

class Buffer {
public:
    Buffer() = default;
    Buffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage,
           VkMemoryPropertyFlags memProps,
           VkDeviceSize alignment = 0);  // 非 0 用于 AS scratch 等需大对齐的场景
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept { swap(o); }
    Buffer& operator=(Buffer&& o) noexcept { if (this != &o) { reset(); swap(o); } return *this; }

    void swap(Buffer& o) noexcept;
    void reset();

    VkBuffer handle() const { return m_buffer; }
    VkDeviceSize size() const { return m_size; }
    VkDeviceAddress deviceAddress() const { return m_address; }
    void* mapped() const { return m_mapped; }

private:
    Device* m_device = nullptr;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VkDeviceSize m_size = 0;
    VkDeviceAddress m_address = 0;
    void* m_mapped = nullptr;
    VkDeviceMemory m_manualMem = VK_NULL_HANDLE;  // 非 VMA 路径的手动分配 memory
};

}
