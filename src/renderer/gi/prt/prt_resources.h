#pragma once
#include "core/image.h"
#include "rhi/base/texture.h"
#include <memory>

// PrtResources —— PRT volume visibility transfer SH。
//
// 始终 bake order-3 SH16（16 系数）到 5 张 RGBA16F 32³ 3D image：
//   image  : coefs 0..3   (Y00, Y1-1, Y10, Y11)        SH4 段
//   imageB : coefs 4..7   (Y2-2, Y2-1, Y20, Y21)       SH9 段
//   imageC : coef  8      (Y22) + 3 unused             SH9 段
//   imageD : coefs 9..12  (Y3-3, Y3-2, Y3-1, Y30)      SH16 段
//   imageE : coefs 13..15 (Y31, Y32, Y33) + 1 unused   SH16 段
// 运行时 SH4/SH9/SH16 toggle 决定 lighting 读几张；bake 数据相同。

namespace somegi {
class Device;
namespace rhi { class RHIDevice; }

class PrtResources {
public:
    void create(Device& d, rhi::RHIDevice& rhiD, uint32_t resolution);
    void destroy();

    const Image& image()  const { return m_image; }
    const Image& imageB() const { return m_imageB; }
    const Image& imageC() const { return m_imageC; }
    const Image& imageD() const { return m_imageD; }
    const Image& imageE() const { return m_imageE; }
    VkImageView view()  const { return m_image.view(); }
    VkImageView viewB() const { return m_imageB.view(); }
    VkImageView viewC() const { return m_imageC.view(); }
    VkImageView viewD() const { return m_imageD.view(); }
    VkImageView viewE() const { return m_imageE.view(); }

    rhi::RHITexture* rhiTex()  const { return m_tex.get(); }
    rhi::RHITextureView* rhiView()  const { return m_view.get(); }
    rhi::RHITexture* rhiTexB() const { return m_texB.get(); }
    rhi::RHITextureView* rhiViewB() const { return m_viewB.get(); }
    rhi::RHITexture* rhiTexC() const { return m_texC.get(); }
    rhi::RHITextureView* rhiViewC() const { return m_viewC.get(); }
    rhi::RHITexture* rhiTexD() const { return m_texD.get(); }
    rhi::RHITextureView* rhiViewD() const { return m_viewD.get(); }
    rhi::RHITexture* rhiTexE() const { return m_texE.get(); }
    rhi::RHITextureView* rhiViewE() const { return m_viewE.get(); }

    uint32_t resolution() const { return m_resolution; }

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    Image m_image;
    Image m_imageB;
    Image m_imageC;
    Image m_imageD;
    Image m_imageE;
    std::unique_ptr<rhi::RHITexture> m_tex, m_texB, m_texC, m_texD, m_texE;
    std::unique_ptr<rhi::RHITextureView> m_view, m_viewB, m_viewC, m_viewD, m_viewE;
    uint32_t m_resolution = 0;
};

}
