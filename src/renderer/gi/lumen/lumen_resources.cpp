#include "renderer/gi/lumen/lumen_resources.h"
#include "core/device.h"
#include "rhi/base/device.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_buffer.h"

namespace somegi {

void LumenResources::create(Device& d, rhi::RHIDevice& rhiD, VkExtent2D screenExt) {


    m_probeGridW = (screenExt.width  + kProbeTileSize - 1) / kProbeTileSize;
    m_probeGridH = (screenExt.height + kProbeTileSize - 1) / kProbeTileSize;

    ImageDesc desc{};
    desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    desc.extent = {atlasWidth(), atlasHeight(), 1};
    desc.type = VK_IMAGE_TYPE_2D;
    desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
               | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    m_probeAtlas    = Image(d, desc);
    m_filteredAtlas = Image(d, desc);
    m_prevAtlas     = Image(d, desc);

    size_t rb = (size_t)probeCount() * kRaysPerProbe * 8 * sizeof(float);
    m_rayBuffer = Buffer(d, rb,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // RHI non-owning wrappers
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiD);
    auto fmt = rhi::toRhiFormat(desc.format);
    m_probeAtlasTex = rhi::VkRHITexture::createNonOwning(vkDev, m_probeAtlas.image(), fmt, atlasWidth(), atlasHeight());
    m_probeAtlasView = rhi::VkRHITextureView::createNonOwning(vkDev, m_probeAtlas.view());
    m_filteredAtlasTex = rhi::VkRHITexture::createNonOwning(vkDev, m_filteredAtlas.image(), fmt, atlasWidth(), atlasHeight());
    m_filteredAtlasView = rhi::VkRHITextureView::createNonOwning(vkDev, m_filteredAtlas.view());
    m_prevAtlasTex = rhi::VkRHITexture::createNonOwning(vkDev, m_prevAtlas.image(), fmt, atlasWidth(), atlasHeight());
    m_prevAtlasView = rhi::VkRHITextureView::createNonOwning(vkDev, m_prevAtlas.view());
    m_rayBufferRhi = rhi::VkRHIBuffer::createNonOwning(vkDev, m_rayBuffer.handle(), m_rayBuffer.size());
}

void LumenResources::destroy() {
    m_probeAtlas.reset();
    m_filteredAtlas.reset();
    m_prevAtlas.reset();
    m_rayBuffer.reset();
    m_probeAtlasTex.reset(); m_probeAtlasView.reset();
    m_filteredAtlasTex.reset(); m_filteredAtlasView.reset();
    m_prevAtlasTex.reset(); m_prevAtlasView.reset();
    m_rayBufferRhi.reset();
    m_probeGridW = m_probeGridH = 0;
    m_rhiDevice = nullptr;
}

}
