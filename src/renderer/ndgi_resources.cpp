#include "ndgi_resources.h"
#include "core/device.h"
#include <cstring>

namespace somegi {

void NdgiResources::create(Device& d) {
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
    m_sampleCount = Buffer(d, 4, sbuf, hmem);
}

void NdgiResources::destroy() {
    m_w1.reset(); m_b1.reset(); m_w2.reset(); m_b2.reset();
    m_w3.reset(); m_b3.reset();
    m_sampleBuf.reset();
    m_sampleCount.reset();
    m_device = nullptr;
}

} // namespace somegi
