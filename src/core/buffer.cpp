#include "buffer.h"
#include "device.h"
#include "rhi/vulkan/vk_device.h"
#include <utility>

namespace somegi {

static rhi::BufferUsage toRhiUsage(VkBufferUsageFlags vk) {
    uint32_t bits = 0;
    if (vk & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)   bits |= (uint32_t)rhi::BufferUsage::Vertex;
    if (vk & VK_BUFFER_USAGE_INDEX_BUFFER_BIT)    bits |= (uint32_t)rhi::BufferUsage::Index;
    if (vk & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)  bits |= (uint32_t)rhi::BufferUsage::Uniform;
    if (vk & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)  bits |= (uint32_t)rhi::BufferUsage::Storage;
    if (vk & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) bits |= (uint32_t)rhi::BufferUsage::Indirect;
    if (vk & VK_BUFFER_USAGE_TRANSFER_SRC_BIT)    bits |= (uint32_t)rhi::BufferUsage::TransferSrc;
    if (vk & VK_BUFFER_USAGE_TRANSFER_DST_BIT)    bits |= (uint32_t)rhi::BufferUsage::TransferDst;
    if (vk & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) bits |= (uint32_t)rhi::BufferUsage::AccelStruct;
    if (vk & VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR) bits |= (uint32_t)rhi::BufferUsage::ShaderBindingTable;
    return (rhi::BufferUsage)bits;
}

static rhi::MemoryType toRhiMem(VkMemoryPropertyFlags vk) {
    if (vk & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        if (vk & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) return rhi::MemoryType::HostCached;
        return rhi::MemoryType::HostVisible;
    }
    return rhi::MemoryType::DeviceLocal;
}

void Buffer::init(rhi::RHIDevice& device, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags memProps, VkDeviceSize alignment) {
    rhi::BufferDesc desc;
    desc.size = size;
    desc.usage = toRhiUsage(usage);
    desc.memory = toRhiMem(memProps);
    desc.alignment = alignment;
    m_rhiBuffer = device.createBuffer(desc);
    m_size = size;
}

Buffer::Buffer(rhi::VkRHIDevice& vkDev, VkDeviceSize size, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags memProps, VkDeviceSize alignment) {
    init(vkDev, size, usage, memProps, alignment);
}

Buffer::Buffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags memProps, VkDeviceSize alignment) {
    init(d.rhiDev(), size, usage, memProps, alignment);
}

Buffer::~Buffer() = default;

void Buffer::swap(Buffer& o) noexcept {
    std::swap(m_rhiBuffer, o.m_rhiBuffer);
    std::swap(m_size, o.m_size);
}

void Buffer::reset() {
    m_rhiBuffer.reset();
    m_size = 0;
}

VkBuffer Buffer::handle() const {
    return m_rhiBuffer ? (VkBuffer)(uintptr_t)m_rhiBuffer->nativeHandle() : VK_NULL_HANDLE;
}

VkDeviceSize Buffer::size() const {
    return m_rhiBuffer ? m_rhiBuffer->size() : 0;
}

VkDeviceAddress Buffer::deviceAddress() const {
    return m_rhiBuffer ? m_rhiBuffer->deviceAddress() : 0;
}

void* Buffer::mapped() const {
    // RHIBuffer::map() 不是 const，但 mapped() 应该可以 const
    return m_rhiBuffer ? const_cast<rhi::RHIBuffer*>(m_rhiBuffer.get())->map() : nullptr;
}

}
