#pragma once
#include "core/image.h"
#include "rhi/base/texture.h"
#include <vector>
#include <memory>

// VxgiResources —— M7：单 cascade voxel cone tracing 用的 3D 体素网格。
// 128³ RGBA16F + 完整 mipchain（log2(128)=7 → 共 8 级 mip）。
//
// 用法链：
//   voxelize → 写 mip 0 (RGB=albedo, A=opacity)
//   inject   → 改写 mip 0 RGB = 受光 radiance（保留 A=opacity）
//   mipmap   → mip0 → mip1 → ... 递推下采样（opacity-weighted）
//   cone trace 在 lighting.slang 里 SampleLevel 各级 mip
//
// 一致 image，多个 view：
//   - m_image.view() 是完整 mipchain（lighting 端用 sampler 自动选 mip）
//   - m_mipViews[i] 是 i 级的 storage view（voxelize / inject / mipmap
//     reduce 的 dst 端要"指定 mip"写入 RWTexture3D）

namespace somegi {
class Device;
namespace rhi { class RHIDevice; }

class VxgiResources {
public:
    // resolution = mip 0 的边长（必须是 2 的幂）。format 默认 RGBA16F。
    void create(Device& d, rhi::RHIDevice& rhiD, uint32_t resolution);
    void destroy();

    const Image& image() const { return m_image; }
    VkImageView fullView() const { return m_image.view(); }
    rhi::RHITexture* rhiTex() const { return m_imageTex.get(); }
    rhi::RHITextureView* rhiView() const { return m_imageView.get(); }
    rhi::RHITextureView* mipView(uint32_t level) const { return m_mipViews[level].get(); }
    uint32_t mipLevels() const { return (uint32_t)m_mipViews.size(); }
    uint32_t resolution() const { return m_resolution; }

    // B.6 各向异性 alpha mipchain：与 voxelGrid 共用 mip levels；rgb = 沿
    // X/Y/Z 方向的 Beer-composite 不透明度，alpha = 各向同性。
    const Image& aniso() const { return m_aniso; }
    VkImageView anisoFullView() const { return m_aniso.view(); }
    rhi::RHITexture* anisoRhiTex() const { return m_anisoTex.get(); }
    rhi::RHITextureView* anisoRhiView() const { return m_anisoView.get(); }
    rhi::RHITextureView* anisoMipView(uint32_t level) const { return m_anisoMipViews[level].get(); }

    // C.2 relight scratch：单 mip RGBA16F，作 relight pass 的 dst（避免
    // 同 image 边读边写）；relight 完 vkCmdCopyImage 回 voxelGrid mip 0。
    const Image& relightScratch() const { return m_relightScratch; }
    VkImageView relightScratchView() const { return m_relightScratch.view(); }
    rhi::RHITexture* relightScratchRhiTex() const { return m_relightScratchTex.get(); }
    rhi::RHITextureView* relightScratchRhiView() const { return m_relightScratchView.get(); }

    // L.3a multi-bounce ping-pong scratch
    const Image& relightScratch2() const { return m_relightScratch2; }
    VkImageView relightScratch2View() const { return m_relightScratch2.view(); }
    rhi::RHITexture* relightScratch2RhiTex() const { return m_relightScratch2Tex.get(); }
    rhi::RHITextureView* relightScratch2RhiView() const { return m_relightScratch2View.get(); }

    // L.3b 6-axis directional radiance: 3× RGBA16F 3D, +X/+Y/+Z 主轴 radiance
    const Image& sixAxisX() const { return m_sixAxisX; }
    const Image& sixAxisY() const { return m_sixAxisY; }
    const Image& sixAxisZ() const { return m_sixAxisZ; }
    rhi::RHITexture* sixAxisXRhiTex() const { return m_sixAxisXTex.get(); }
    rhi::RHITextureView* sixAxisXRhiView() const { return m_sixAxisXView.get(); }
    rhi::RHITexture* sixAxisYRhiTex() const { return m_sixAxisYTex.get(); }
    rhi::RHITextureView* sixAxisYRhiView() const { return m_sixAxisYView.get(); }
    rhi::RHITexture* sixAxisZRhiTex() const { return m_sixAxisZTex.get(); }
    rhi::RHITextureView* sixAxisZRhiView() const { return m_sixAxisZView.get(); }
    bool hasSixAxis() const { return m_hasSixAxis; }
    void createSixAxis(Device& d, rhi::RHIDevice& rhiD);
    void destroySixAxis();

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    Image m_image;
    std::unique_ptr<rhi::RHITexture> m_imageTex;
    std::unique_ptr<rhi::RHITextureView> m_imageView;
    std::vector<std::unique_ptr<rhi::RHITextureView>> m_mipViews;   // per-mip storage view (RHI-owning)
    Image m_aniso;
    std::unique_ptr<rhi::RHITexture> m_anisoTex;
    std::unique_ptr<rhi::RHITextureView> m_anisoView;
    std::vector<std::unique_ptr<rhi::RHITextureView>> m_anisoMipViews;
    Image m_relightScratch;
    std::unique_ptr<rhi::RHITexture> m_relightScratchTex;
    std::unique_ptr<rhi::RHITextureView> m_relightScratchView;
    Image m_relightScratch2;
    std::unique_ptr<rhi::RHITexture> m_relightScratch2Tex;
    std::unique_ptr<rhi::RHITextureView> m_relightScratch2View;
    Image m_sixAxisX, m_sixAxisY, m_sixAxisZ;
    std::unique_ptr<rhi::RHITexture> m_sixAxisXTex, m_sixAxisYTex, m_sixAxisZTex;
    std::unique_ptr<rhi::RHITextureView> m_sixAxisXView, m_sixAxisYView, m_sixAxisZView;
    bool  m_hasSixAxis = false;
    uint32_t m_resolution = 0;
};

}
