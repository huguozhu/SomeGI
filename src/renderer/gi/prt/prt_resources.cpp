#include "renderer/gi/prt/prt_resources.h"
#include "core/device.h"
#include "rhi/base/device.h"
#include "rhi/vulkan/vk_texture.h"

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

void PrtResources::create(Device& d, rhi::RHIDevice& rhiD, uint32_t resolution) {
    m_device = &d;
    m_resolution = resolution;
    m_image  = makePrtSlice(d, resolution);
    m_imageB = makePrtSlice(d, resolution);
    m_imageC = makePrtSlice(d, resolution);
    m_imageD = makePrtSlice(d, resolution);
    m_imageE = makePrtSlice(d, resolution);

    // RHI non-owning wrappers
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiD);
    auto fmt = rhi::toRhiFormat(VK_FORMAT_R16G16B16A16_SFLOAT);
    m_tex = rhi::VkRHITexture::createNonOwning(vkDev, m_image.image(), fmt, resolution, resolution);
    m_view = rhi::VkRHITextureView::createNonOwning(vkDev, m_image.view());
    m_texB = rhi::VkRHITexture::createNonOwning(vkDev, m_imageB.image(), fmt, resolution, resolution);
    m_viewB = rhi::VkRHITextureView::createNonOwning(vkDev, m_imageB.view());
    m_texC = rhi::VkRHITexture::createNonOwning(vkDev, m_imageC.image(), fmt, resolution, resolution);
    m_viewC = rhi::VkRHITextureView::createNonOwning(vkDev, m_imageC.view());
    m_texD = rhi::VkRHITexture::createNonOwning(vkDev, m_imageD.image(), fmt, resolution, resolution);
    m_viewD = rhi::VkRHITextureView::createNonOwning(vkDev, m_imageD.view());
    m_texE = rhi::VkRHITexture::createNonOwning(vkDev, m_imageE.image(), fmt, resolution, resolution);
    m_viewE = rhi::VkRHITextureView::createNonOwning(vkDev, m_imageE.view());
}

void PrtResources::destroy() {
    m_image.reset();
    m_imageB.reset();
    m_imageC.reset();
    m_imageD.reset();
    m_imageE.reset();
    m_tex.reset(); m_view.reset();
    m_texB.reset(); m_viewB.reset();
    m_texC.reset(); m_viewC.reset();
    m_texD.reset(); m_viewD.reset();
    m_texE.reset(); m_viewE.reset();
    m_resolution = 0;
    m_device = nullptr;
}

}
