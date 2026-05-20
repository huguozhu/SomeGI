#include "lumen_resources.h"
#include "core/device.h"

namespace somegi {

void LumenResources::create(Device& d, VkExtent2D screenExt) {
    m_device = &d;

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
}

void LumenResources::destroy() {
    m_probeAtlas.reset();
    m_filteredAtlas.reset();
    m_prevAtlas.reset();
    m_rayBuffer.reset();
    m_probeGridW = m_probeGridH = 0;
    m_device = nullptr;
}

}
