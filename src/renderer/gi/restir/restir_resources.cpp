#include "renderer/gi/restir/restir_resources.h"
#include "core/device.h"
#include "rhi/base/device.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"
#include <cstring>

namespace somegi {

void RestirResources::create(Device& d, rhi::RHIDevice& rhiD, VkExtent2D screenExtent, uint32_t maxLights) {

    m_rhiDevice = &rhiD;
    m_maxLights = maxLights;
    m_screenExtent = screenExtent;
    createReservoirImages(d, screenExtent);

    // light SSBO：HOST_VISIBLE，CPU 每帧 memcpy 更新；shader 当 StructuredBuffer 读
    VkDeviceSize bytes = (VkDeviceSize)maxLights * sizeof(PointLightCpu);
    if (bytes < 32) bytes = 32;   // 至少够 1 个；空场景也别建 0 字节 buffer
    m_lightBuf = Buffer(d, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // RHI non-owning wrapper for lightBuf
    m_lightBufRhi = rhi::VkRHIBuffer::createNonOwning(
        static_cast<rhi::VkRHIDevice&>(rhiD), m_lightBuf.handle(), m_lightBuf.size());
}

void RestirResources::createReservoirImages(Device& d, VkExtent2D ext) {
    ImageDesc desc{};
    desc.format = VK_FORMAT_R32G32B32A32_UINT;
    desc.extent = {ext.width, ext.height, 1};
    // STORAGE：init/spatial 写；SAMPLED：spatial/shade 读（用 .Load()）
    desc.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    m_reservoirA = Image(d, desc);
    m_reservoirB = Image(d, desc);

    // RHI non-owning wrappers
    if (m_rhiDevice) {
        auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_rhiDevice);
        auto fmt = rhi::toRhiFormat(desc.format);
        m_reservoirATex = rhi::VkRHITexture::createNonOwning(vkDev, m_reservoirA.image(), fmt, ext.width, ext.height);
        m_reservoirAView = rhi::VkRHITextureView::createNonOwning(vkDev, m_reservoirA.view());
        m_reservoirBTex = rhi::VkRHITexture::createNonOwning(vkDev, m_reservoirB.image(), fmt, ext.width, ext.height);
        m_reservoirBView = rhi::VkRHITextureView::createNonOwning(vkDev, m_reservoirB.view());
    }
}

void RestirResources::resize(Device& d, VkExtent2D newExtent) {
    if (newExtent.width == m_screenExtent.width &&
        newExtent.height == m_screenExtent.height) return;
    m_reservoirA.reset();
    m_reservoirB.reset();
    m_reservoirATex.reset(); m_reservoirAView.reset();
    m_reservoirBTex.reset(); m_reservoirBView.reset();
    m_screenExtent = newExtent;
    createReservoirImages(d, newExtent);
}

void RestirResources::destroy() {
    if (!m_rhiDevice) return;
    m_reservoirA.reset();
    m_reservoirB.reset();
    m_lightBuf.reset();
    m_reservoirATex.reset(); m_reservoirAView.reset();
    m_reservoirBTex.reset(); m_reservoirBView.reset();
    m_lightBufRhi.reset();
    m_screenExtent = {};
    m_maxLights = 0;
    m_lightCount = 0;
    m_rhiDevice = nullptr;
    m_rhiDevice = nullptr;
}

void RestirResources::updateLights(const std::vector<PointLightCpu>& lights) {
    m_lightCount = (uint32_t)std::min<size_t>(lights.size(), m_maxLights);
    if (m_lightCount == 0) return;
    void* p = m_lightBuf.mapped();
    if (!p) return;
    std::memcpy(p, lights.data(), m_lightCount * sizeof(PointLightCpu));
}

}
