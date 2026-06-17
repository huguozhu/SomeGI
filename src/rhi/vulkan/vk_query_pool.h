// rhi/vulkan/vk_query_pool.h
#pragma once
#include "../command_buffer.h"  // RHIQueryPool
#include "vk_device.h"

namespace somegi {
namespace rhi {

class VkRHIQueryPool : public RHIQueryPool {
public:
    static std::unique_ptr<RHIQueryPool> create(VkRHIDevice& device, uint32_t count);
    ~VkRHIQueryPool() override;
    void getResults(uint32_t first, uint32_t count, uint64_t* data) override;
    void* nativeHandle() const override { return (void*)m_pool; }
private:
    VkRHIDevice& m_device;
    VkQueryPool m_pool = VK_NULL_HANDLE;
    uint32_t m_count = 0;
    VkRHIQueryPool(VkRHIDevice& d) : m_device(d) {}
};

} // namespace rhi
} // namespace somegi
