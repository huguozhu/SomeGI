// rhi/vulkan/vk_query_pool.cpp
#include "vk_query_pool.h"

namespace somegi {
namespace rhi {

std::unique_ptr<RHIQueryPool> VkRHIQueryPool::create(VkRHIDevice& device, uint32_t count) {
    auto p = std::unique_ptr<VkRHIQueryPool>(new VkRHIQueryPool(device));
    p->m_count = count;
    VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = count;
    vkCreateQueryPool(device.vkDevice(), &ci, nullptr, &p->m_pool);
    return p;
}
VkRHIQueryPool::~VkRHIQueryPool() { if (m_pool) vkDestroyQueryPool(m_device.vkDevice(), m_pool, nullptr); }
void VkRHIQueryPool::getResults(uint32_t first, uint32_t count, uint64_t* data) {
    vkGetQueryPoolResults(m_device.vkDevice(), m_pool, first, count,
        count * sizeof(uint64_t), data, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
}

} // namespace rhi
} // namespace somegi
