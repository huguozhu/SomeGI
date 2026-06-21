#include "renderer/gi/lpv/lpv_grid.h"
#include "core/device.h"
#include "rhi/base/device.h"
#include "rhi/vulkan/vk_texture.h"

namespace somegi {

namespace {
Image makeLpvImage(Device& d, uint32_t resolution) {
    ImageDesc desc{};
    desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;   // 4 SH 系数 / cell
    desc.extent = {resolution, resolution, resolution};
    desc.type   = VK_IMAGE_TYPE_3D;
    // STORAGE：inject / propagate 写；SAMPLED：lighting trilinear 读；
    // TRANSFER_DST：每帧 vkCmdClearColorImage 抹掉残留。
    desc.usage  = VK_IMAGE_USAGE_STORAGE_BIT
                | VK_IMAGE_USAGE_SAMPLED_BIT
                | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return Image(d, desc);
}
}

void LpvGrid::create(Device& d, rhi::RHIDevice& rhiD, uint32_t resolution) {
    lpvR = makeLpvImage(d, resolution);
    lpvG = makeLpvImage(d, resolution);
    lpvB = makeLpvImage(d, resolution);

    // RHI non-owning wrappers
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiD);
    auto fmt = rhi::toRhiFormat(VK_FORMAT_R16G16B16A16_SFLOAT);
    lpvRTex = rhi::VkRHITexture::createNonOwning(vkDev, lpvR.image(), fmt, resolution, resolution);
    lpvRView = rhi::VkRHITextureView::createNonOwning(vkDev, lpvR.view());
    lpvGTex = rhi::VkRHITexture::createNonOwning(vkDev, lpvG.image(), fmt, resolution, resolution);
    lpvGView = rhi::VkRHITextureView::createNonOwning(vkDev, lpvG.view());
    lpvBTex = rhi::VkRHITexture::createNonOwning(vkDev, lpvB.image(), fmt, resolution, resolution);
    lpvBView = rhi::VkRHITextureView::createNonOwning(vkDev, lpvB.view());
}

void LpvGrid::destroy() {
    lpvR.reset();
    lpvG.reset();
    lpvB.reset();
    lpvRTex.reset(); lpvRView.reset();
    lpvGTex.reset(); lpvGView.reset();
    lpvBTex.reset(); lpvBView.reset();
}

void LpvResources::create(Device& d, rhi::RHIDevice& rhiD, uint32_t resolution) {
    m_resolution = resolution;
    m_grids[0].create(d, rhiD, resolution);
    m_grids[1].create(d, rhiD, resolution);
    m_gv = makeLpvImage(d, resolution);   // 与 LPV 同形 RGBA16F 32³

    // RHI non-owning wrapper for gv
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiD);
    auto fmt = rhi::toRhiFormat(VK_FORMAT_R16G16B16A16_SFLOAT);
    m_gvTex = rhi::VkRHITexture::createNonOwning(vkDev, m_gv.image(), fmt, resolution, resolution);
    m_gvView = rhi::VkRHITextureView::createNonOwning(vkDev, m_gv.view());

    m_curIdx = 0;
}

void LpvResources::destroy() {
    m_grids[0].destroy();
    m_grids[1].destroy();
    m_gv.reset();
    m_gvTex.reset(); m_gvView.reset();
    m_resolution = 0;
    m_curIdx = 0;
}

}
