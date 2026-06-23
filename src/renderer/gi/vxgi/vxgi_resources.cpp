#include "renderer/gi/vxgi/vxgi_resources.h"
#include "core/device.h"
#include "rhi/base/device.h"
#include "rhi/vulkan/vk_texture.h"

namespace somegi {

namespace {
uint32_t mipCountForExtent(uint32_t e) {
    uint32_t m = 1;
    while ((e >> m) > 0) ++m;
    return m;   // e=128 → 8 levels (mip 0..7, 最小 1³)
}
}

void VxgiResources::create(Device& d, rhi::RHIDevice& rhiD, uint32_t resolution) {
    m_rhiDevice = &rhiD;
    m_resolution = resolution;
    uint32_t mips = mipCountForExtent(resolution);

    ImageDesc desc{};
    desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    desc.extent = {resolution, resolution, resolution};
    desc.type   = VK_IMAGE_TYPE_3D;
    desc.mipLevels = mips;
    // STORAGE: voxelize/inject/mipmap 都写；SAMPLED: lighting 采；
    // TRANSFER_DST: 每帧 vkCmdClearColorImage 抹 0 起步（避免上帧残留）。
    desc.usage = VK_IMAGE_USAGE_STORAGE_BIT
               | VK_IMAGE_USAGE_SAMPLED_BIT
               | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    m_image = Image(d, desc);

    // RHI non-owning texture + full view for the main voxel image
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiD);
    auto fmt = rhi::toRhiFormat(desc.format);
    m_imageTex = rhi::VkRHITexture::createNonOwning(vkDev, m_image.image(), fmt, resolution, resolution, mips);
    m_imageView = rhi::VkRHITextureView::createNonOwning(vkDev, m_image.view());

    // per-mip storage view —— 写 RWTexture3D 时必须指定单独 mip。
    m_mipViews.resize(mips);
    for (uint32_t i = 0; i < mips; ++i) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = m_image.image();
        vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vi.format = desc.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1};
        m_mipViews[i] = rhi::VkRHITextureView::createNonOwning(d.device(), vi);
    }

    // B.6 各向异性 alpha：与 voxelGrid 同分辨率 + mip 数。RGBA16F：
    // RGB = X/Y/Z 方向的 Beer-composite alpha，A = 备用（保持 isotropic）。
    ImageDesc adesc{};
    adesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    adesc.extent = {resolution, resolution, resolution};
    adesc.type = VK_IMAGE_TYPE_3D;
    adesc.mipLevels = mips;
    adesc.usage = VK_IMAGE_USAGE_STORAGE_BIT
                | VK_IMAGE_USAGE_SAMPLED_BIT
                | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    m_aniso = Image(d, adesc);

    // RHI non-owning wrappers for aniso
    m_anisoTex = rhi::VkRHITexture::createNonOwning(vkDev, m_aniso.image(), fmt, resolution, resolution, mips);
    m_anisoView = rhi::VkRHITextureView::createNonOwning(vkDev, m_aniso.view());

    m_anisoMipViews.resize(mips);
    for (uint32_t i = 0; i < mips; ++i) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = m_aniso.image();
        vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vi.format = adesc.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1};
        m_anisoMipViews[i] = rhi::VkRHITextureView::createNonOwning(d.device(), vi);
    }

    // C.2 relight scratch：单 mip RGBA16F 128³ ≈ 16 MB
    ImageDesc sdesc{};
    sdesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    sdesc.extent = {resolution, resolution, resolution};
    sdesc.type = VK_IMAGE_TYPE_3D;
    sdesc.usage = VK_IMAGE_USAGE_STORAGE_BIT
                | VK_IMAGE_USAGE_SAMPLED_BIT
                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    m_relightScratch = Image(d, sdesc);

    // RHI wrappers for relight scratch
    m_relightScratchTex = rhi::VkRHITexture::createNonOwning(vkDev, m_relightScratch.image(), fmt, resolution, resolution);
    m_relightScratchView = rhi::VkRHITextureView::createNonOwning(vkDev, m_relightScratch.view());

    // L.3a second scratch for multi-bounce ping-pong
    ImageDesc sdesc2{};
    sdesc2.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    sdesc2.extent = {resolution, resolution, resolution};
    sdesc2.type = VK_IMAGE_TYPE_3D;
    sdesc2.usage = VK_IMAGE_USAGE_STORAGE_BIT
                 | VK_IMAGE_USAGE_SAMPLED_BIT
                 | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    m_relightScratch2 = Image(d, sdesc2);

    // RHI wrappers for second scratch
    m_relightScratch2Tex = rhi::VkRHITexture::createNonOwning(vkDev, m_relightScratch2.image(), fmt, resolution, resolution);
    m_relightScratch2View = rhi::VkRHITextureView::createNonOwning(vkDev, m_relightScratch2.view());
}

void VxgiResources::createSixAxis(Device& d, rhi::RHIDevice& rhiD) {
    if (m_hasSixAxis) return;
    ImageDesc ax{};
    ax.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ax.extent = {m_resolution, m_resolution, m_resolution};
    ax.type = VK_IMAGE_TYPE_3D;
    ax.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    m_sixAxisX = Image(d, ax);
    m_sixAxisY = Image(d, ax);
    m_sixAxisZ = Image(d, ax);

    // RHI non-owning wrappers
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiD);
    auto fmt = rhi::toRhiFormat(ax.format);
    m_sixAxisXTex = rhi::VkRHITexture::createNonOwning(vkDev, m_sixAxisX.image(), fmt, m_resolution, m_resolution);
    m_sixAxisXView = rhi::VkRHITextureView::createNonOwning(vkDev, m_sixAxisX.view());
    m_sixAxisYTex = rhi::VkRHITexture::createNonOwning(vkDev, m_sixAxisY.image(), fmt, m_resolution, m_resolution);
    m_sixAxisYView = rhi::VkRHITextureView::createNonOwning(vkDev, m_sixAxisY.view());
    m_sixAxisZTex = rhi::VkRHITexture::createNonOwning(vkDev, m_sixAxisZ.image(), fmt, m_resolution, m_resolution);
    m_sixAxisZView = rhi::VkRHITextureView::createNonOwning(vkDev, m_sixAxisZ.view());
    m_hasSixAxis = true;
}

void VxgiResources::destroySixAxis() {
    m_sixAxisX.reset();
    m_sixAxisY.reset();
    m_sixAxisZ.reset();
    m_sixAxisXTex.reset(); m_sixAxisXView.reset();
    m_sixAxisYTex.reset(); m_sixAxisYView.reset();
    m_sixAxisZTex.reset(); m_sixAxisZView.reset();
    m_hasSixAxis = false;
}

void VxgiResources::destroy() {
    if (!m_rhiDevice) return;
    // unique_ptr reset 自动销毁 VkImageView（通过 RHI VkRHITextureView 析构）
    m_mipViews.clear();
    m_anisoMipViews.clear();
    m_image.reset();
    m_imageTex.reset(); m_imageView.reset();
    m_aniso.reset();
    m_anisoTex.reset(); m_anisoView.reset();
    m_relightScratch.reset();
    m_relightScratchTex.reset(); m_relightScratchView.reset();
    m_relightScratch2.reset();
    m_relightScratch2Tex.reset(); m_relightScratch2View.reset();
    m_resolution = 0;
    m_rhiDevice = nullptr;
}

}
