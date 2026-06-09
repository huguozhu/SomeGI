#include "renderer/gi/sdfgi/sdfgi_resources.h"
#include "core/device.h"

namespace somegi {

void SdfgiResources::create(Device& d, uint32_t resolution) {
    m_device = &d;
    m_resolution = resolution;

    // seedA / seedB：RGBA16F 3D，单 mip。STORAGE+SAMPLED；JFA 互写互读。
    ImageDesc s{};
    s.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    s.extent = {resolution, resolution, resolution};
    s.type = VK_IMAGE_TYPE_3D;
    s.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    m_seedA = Image(d, s);
    m_seedB = Image(d, s);

    // udf：R16F 3D，单 mip。STORAGE（finalize 写）+ SAMPLED（trace 读）。
    ImageDesc u{};
    u.format = VK_FORMAT_R16_SFLOAT;
    u.extent = {resolution, resolution, resolution};
    u.type = VK_IMAGE_TYPE_3D;
    u.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    m_udf = Image(d, u);
}

void SdfgiResources::destroy() {
    if (!m_device) return;
    m_seedA.reset();
    m_seedB.reset();
    m_udf.reset();
    m_resolution = 0;
    m_device = nullptr;
}

}
