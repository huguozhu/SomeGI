#pragma once
#include "vk_common.h"

struct VmaAllocation_T;
typedef VmaAllocation_T* VmaAllocation;

struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;

namespace somegi {

class Device;

namespace rhi { class VkRHIDevice; }

class Buffer {
public:
    Buffer() = default;
    Buffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage,
           VkMemoryPropertyFlags memProps,
           VkDeviceSize alignment = 0);
    Buffer(rhi::VkRHIDevice& vkDev, VkDeviceSize size, VkBufferUsageFlags usage,
           VkMemoryPropertyFlags memProps, VkDeviceSize alignment = 0);
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
    VkDevice m_vkDev = VK_NULL_HANDLE;
    VmaAllocator m_vma = VK_NULL_HANDLE;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VkDeviceSize m_size = 0;
    VkDeviceAddress m_address = 0;
    void* m_mapped = nullptr;
    VkDeviceMemory m_manualMem = VK_NULL_HANDLE;
};

}
