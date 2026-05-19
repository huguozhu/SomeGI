#include "ddgi_resources.h"
#include "core/device.h"

namespace somegi {

void DdgiResources::create(Device& d) {
    m_device = &d;

    // irradiance atlas: probesX·octaIrr 宽 × probesY·probesZ·octaIrr 高，RGBA16F
    {
        ImageDesc desc{};
        desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        desc.extent = {irradianceAtlasW(), irradianceAtlasH(), 1};
        desc.type = VK_IMAGE_TYPE_2D;
        desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                   | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        m_irradiance = Image(d, desc);
    }
    // distance atlas: octaDist=16，RG16F 存 (mean, mean²) 给 Chebyshev VSM
    {
        ImageDesc desc{};
        desc.format = VK_FORMAT_R16G16_SFLOAT;
        desc.extent = {distanceAtlasW(), distanceAtlasH(), 1};
        desc.type = VK_IMAGE_TYPE_2D;
        desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                   | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        m_distance = Image(d, desc);
    }
    // ray buffer: 每 probe 64 rays，每 ray 8 floats (dir.xyz + pad + hitRgb + dist)
    {
        size_t bytes = (size_t)kProbeCount * kRaysPerProbe * 8 * sizeof(float);
        m_rayBuffer = Buffer(d, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }
    // B.5 probe states: kProbeCount uints；TRANSFER_DST 让 App 一次性
    // vkCmdFillBuffer 初始化为全 1（active），后续 classify pass 覆写。
    {
        size_t bytes = (size_t)kProbeCount * sizeof(uint32_t);
        m_probeStates = Buffer(d, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }
}

void DdgiResources::destroy() {
    m_irradiance.reset();
    m_distance.reset();
    m_rayBuffer.reset();
    m_probeStates.reset();
    m_device = nullptr;
}

}
