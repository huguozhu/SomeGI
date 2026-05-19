#pragma once
#include "core/image.h"
#include <vector>

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

class VxgiResources {
public:
    // resolution = mip 0 的边长（必须是 2 的幂）。format 默认 RGBA16F。
    void create(Device& d, uint32_t resolution);
    void destroy();

    const Image& image() const { return m_image; }
    VkImageView fullView() const { return m_image.view(); }
    VkImageView mipView(uint32_t level) const { return m_mipViews[level]; }
    uint32_t mipLevels() const { return (uint32_t)m_mipViews.size(); }
    uint32_t resolution() const { return m_resolution; }

    // B.6 各向异性 alpha mipchain：与 voxelGrid 共用 mip levels；rgb = 沿
    // X/Y/Z 方向的 Beer-composite 不透明度，alpha = 各向同性。
    const Image& aniso() const { return m_aniso; }
    VkImageView anisoFullView() const { return m_aniso.view(); }
    VkImageView anisoMipView(uint32_t level) const { return m_anisoMipViews[level]; }

    // C.2 relight scratch：单 mip RGBA16F，作 relight pass 的 dst（避免
    // 同 image 边读边写）；relight 完 vkCmdCopyImage 回 voxelGrid mip 0。
    const Image& relightScratch() const { return m_relightScratch; }
    VkImageView relightScratchView() const { return m_relightScratch.view(); }

private:
    Device* m_device = nullptr;
    Image m_image;
    std::vector<VkImageView> m_mipViews;   // per-mip storage view
    Image m_aniso;
    std::vector<VkImageView> m_anisoMipViews;
    Image m_relightScratch;
    uint32_t m_resolution = 0;
};

}
