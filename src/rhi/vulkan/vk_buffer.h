// rhi/vulkan/vk_buffer.h
#pragma once
#include "../base/buffer.h"
#include "vk_device.h"

namespace somegi {
namespace rhi {

class VkRHIBuffer : public RHIBuffer {
public:
    VkRHIBuffer(VkRHIDevice& d) : m_device(d) {}
    static std::unique_ptr<RHIBuffer> create(VkRHIDevice& device, const BufferDesc& desc);
    // 非拥有型包装：析构时不销毁 VkBuffer（用于 FGResources 资源池包装）
    static std::unique_ptr<RHIBuffer> createNonOwning(VkRHIDevice& device, VkBuffer buffer, uint64_t size) {
        auto b = std::unique_ptr<VkRHIBuffer>(new VkRHIBuffer(device));
        b->m_ownsBuffer = false;
        b->m_buffer = buffer;
        b->m_size = size;
        return b;
    }
    ~VkRHIBuffer() override;
    void* map() override;
    void unmap() override;
    uint64_t size() const override { return m_size; }
    uint64_t deviceAddress() const override { return m_address; }
    void* nativeHandle() const override { return (void*)m_buffer; }
private:
    VkRHIDevice& m_device;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    uint64_t m_size = 0;
    uint64_t m_address = 0;
    void* m_mapped = nullptr;
    bool m_ownsBuffer = true;
};

} // namespace rhi
} // namespace somegi
