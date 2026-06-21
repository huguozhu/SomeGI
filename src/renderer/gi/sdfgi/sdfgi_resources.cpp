#include "renderer/gi/sdfgi/sdfgi_resources.h"
#include "core/device.h"
#include "rhi/base/device.h"
#include "rhi/vulkan/vk_texture.h"

namespace somegi {

void SdfgiResources::create(Device& d, rhi::RHIDevice& rhiD, uint32_t resolution) {
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

    // RHI non-owning wrappers
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiD);
    m_seedATex = rhi::VkRHITexture::createNonOwning(vkDev, m_seedA.image(), rhi::toRhiFormat(s.format), resolution, resolution);
    m_seedAView = rhi::VkRHITextureView::createNonOwning(vkDev, m_seedA.view());
    m_seedBTex = rhi::VkRHITexture::createNonOwning(vkDev, m_seedB.image(), rhi::toRhiFormat(s.format), resolution, resolution);
    m_seedBView = rhi::VkRHITextureView::createNonOwning(vkDev, m_seedB.view());
    m_udfTex = rhi::VkRHITexture::createNonOwning(vkDev, m_udf.image(), rhi::toRhiFormat(u.format), resolution, resolution);
    m_udfView = rhi::VkRHITextureView::createNonOwning(vkDev, m_udf.view());
}

void SdfgiResources::destroy() {
    if (!m_device) return;
    m_seedATex.reset(); m_seedAView.reset();
    m_seedBTex.reset(); m_seedBView.reset();
    m_udfTex.reset(); m_udfView.reset();
    m_seedA.reset();
    m_seedB.reset();
    m_udf.reset();
    m_resolution = 0;
    m_device = nullptr;
}

}
