#pragma once
#include "vk_common.h"
#include "rhi/base/device.h"
#include "rhi/base/buffer.h"
#include <memory>

namespace somegi {

namespace rhi { class VkRHIDevice; class RHIBuffer; }
class Device;

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

    VkBuffer handle() const;
    VkDeviceSize size() const;
    VkDeviceAddress deviceAddress() const;
    void* mapped() const;

private:
    void init(rhi::RHIDevice& device, VkDeviceSize size, VkBufferUsageFlags usage,
              VkMemoryPropertyFlags memProps, VkDeviceSize alignment);

    std::unique_ptr<rhi::RHIBuffer> m_rhiBuffer;  // RHI 拥有实际 VkBuffer
    VkDeviceSize m_size = 0;
};

}
