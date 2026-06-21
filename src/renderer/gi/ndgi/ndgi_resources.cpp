#include "renderer/gi/ndgi/ndgi_resources.h"
#include "core/device.h"
#include "rhi/base/device.h"
#include "rhi/vulkan/vk_buffer.h"
#include <cstring>

namespace somegi {

void NdgiResources::create(Device& d, rhi::RHIDevice& rhiD) {
    m_device = &d;
    auto sbuf = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    auto hmem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto dmem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // MLP weights — device local for GPU read/write during training
    m_w1 = Buffer(d, kW1Vec4 * sizeof(float) * 4, sbuf | VK_BUFFER_USAGE_TRANSFER_DST_BIT, dmem);
    m_b1 = Buffer(d, kB1Vec4 * sizeof(float) * 4, sbuf | VK_BUFFER_USAGE_TRANSFER_DST_BIT, dmem);
    m_w2 = Buffer(d, kW2Vec4 * sizeof(float) * 4, sbuf | VK_BUFFER_USAGE_TRANSFER_DST_BIT, dmem);
    m_b2 = Buffer(d, kB2Vec4 * sizeof(float) * 4, sbuf | VK_BUFFER_USAGE_TRANSFER_DST_BIT, dmem);
    m_w3 = Buffer(d, kW3Vec4 * sizeof(float) * 4, sbuf | VK_BUFFER_USAGE_TRANSFER_DST_BIT, dmem);
    m_b3 = Buffer(d, kB3Vec4 * sizeof(float) * 4, sbuf | VK_BUFFER_USAGE_TRANSFER_DST_BIT, dmem);

    // 样本 buffer: probe trace writes, training reads
    m_sampleBuf = Buffer(d, kSampleBufferFloats * sizeof(float), sbuf, dmem);

    // 样本计数: atomic counter for probe trace to write, training to read+reset
    m_sampleCount = Buffer(d, 4, sbuf | VK_BUFFER_USAGE_TRANSFER_DST_BIT, hmem);

    // RHI non-owning wrappers
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiD);
    m_rhiW1 = rhi::VkRHIBuffer::createNonOwning(vkDev, m_w1.handle(), m_w1.size());
    m_rhiB1 = rhi::VkRHIBuffer::createNonOwning(vkDev, m_b1.handle(), m_b1.size());
    m_rhiW2 = rhi::VkRHIBuffer::createNonOwning(vkDev, m_w2.handle(), m_w2.size());
    m_rhiB2 = rhi::VkRHIBuffer::createNonOwning(vkDev, m_b2.handle(), m_b2.size());
    m_rhiW3 = rhi::VkRHIBuffer::createNonOwning(vkDev, m_w3.handle(), m_w3.size());
    m_rhiB3 = rhi::VkRHIBuffer::createNonOwning(vkDev, m_b3.handle(), m_b3.size());
    m_rhiSampleBuf = rhi::VkRHIBuffer::createNonOwning(vkDev, m_sampleBuf.handle(), m_sampleBuf.size());
    m_rhiSampleCount = rhi::VkRHIBuffer::createNonOwning(vkDev, m_sampleCount.handle(), m_sampleCount.size());
}

void NdgiResources::destroy() {
    m_w1.reset(); m_b1.reset(); m_w2.reset(); m_b2.reset();
    m_w3.reset(); m_b3.reset();
    m_sampleBuf.reset();
    m_sampleCount.reset();
    m_rhiW1.reset(); m_rhiB1.reset(); m_rhiW2.reset(); m_rhiB2.reset();
    m_rhiW3.reset(); m_rhiB3.reset();
    m_rhiSampleBuf.reset();
    m_rhiSampleCount.reset();
    m_device = nullptr;
}

} // namespace somegi
