#include "prt_resources.h"
#include "core/device.h"

namespace somegi {

namespace {
Image makePrtSlice(Device& d, uint32_t resolution) {
    ImageDesc desc{};
    desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    desc.extent = {resolution, resolution, resolution};
    desc.type   = VK_IMAGE_TYPE_3D;
    desc.usage  = VK_IMAGE_USAGE_STORAGE_BIT
                | VK_IMAGE_USAGE_SAMPLED_BIT
                | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return Image(d, desc);
}
}

void PrtResources::create(Device& d, uint32_t resolution) {
    m_device = &d;
    m_resolution = resolution;
    m_image  = makePrtSlice(d, resolution);
    m_imageB = makePrtSlice(d, resolution);
    m_imageC = makePrtSlice(d, resolution);
    m_imageD = makePrtSlice(d, resolution);
    m_imageE = makePrtSlice(d, resolution);
}

void PrtResources::destroy() {
    m_image.reset();
    m_imageB.reset();
    m_imageC.reset();
    m_imageD.reset();
    m_imageE.reset();
    m_resolution = 0;
    m_device = nullptr;
}

}
