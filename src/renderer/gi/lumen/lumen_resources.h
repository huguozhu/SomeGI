#pragma once
#include "core/image.h"
#include "core/buffer.h"
#include "rhi/base/texture.h"
#include "rhi/base/buffer.h"
#include <memory>

// LumenResources —— L 阶段 Lumen-lite 的屏幕 probe atlas + ray buffer。
//
// probe atlas：RGBA16F 2D，(probeGridW * 9) × probeGridH。
//   - 每 probe 占 9 列 texel（SH9 / 3 band），每列存一个 SH coeff 的 rgb
//   - probeIdx → atlasX = (probeIdx % probeGridW) * 9, atlasY = probeIdx / probeGridW
//   - coeff 0..8 依次排列在 atlasX + 0..8
//
// filteredAtlas / prevProbeAtlas：同尺寸，spatial/temporal filter 输出。
//
// ray buffer：SSBO，probeCount * raysPerProbe * 2 float4
//   - float4[0]: rayDir.xyz + pad
//   - float4[1]: hitRadiance.xyz + hitDistance

namespace somegi {
class Device;
namespace rhi { class RHIDevice; }

class LumenResources {
public:
    static constexpr uint32_t kProbeTileSize  = 16;
    static constexpr uint32_t kRaysPerProbe   = 48;
    static constexpr uint32_t kSHCoeffs       = 9;   // SH9 = 3 bands

    void create(Device& d, rhi::RHIDevice& rhiD, VkExtent2D screenExt);
    void destroy();

    const Image&  probeAtlas()    const { return m_probeAtlas; }
    const Image&  filteredAtlas() const { return m_filteredAtlas; }
    const Image&  prevAtlas()     const { return m_prevAtlas; }
    const Buffer& rayBuffer()     const { return m_rayBuffer; }

    rhi::RHITexture* probeAtlasRhiTex() const { return m_probeAtlasTex.get(); }
    rhi::RHITextureView* probeAtlasRhiView() const { return m_probeAtlasView.get(); }
    rhi::RHITexture* filteredAtlasRhiTex() const { return m_filteredAtlasTex.get(); }
    rhi::RHITextureView* filteredAtlasRhiView() const { return m_filteredAtlasView.get(); }
    rhi::RHITexture* prevAtlasRhiTex() const { return m_prevAtlasTex.get(); }
    rhi::RHITextureView* prevAtlasRhiView() const { return m_prevAtlasView.get(); }
    rhi::RHIBuffer* rayBufferRhi() const { return m_rayBufferRhi.get(); }

    uint32_t probeGridW()  const { return m_probeGridW; }
    uint32_t probeGridH()  const { return m_probeGridH; }
    uint32_t probeCount()  const { return m_probeGridW * m_probeGridH; }
    uint32_t atlasWidth()  const { return m_probeGridW * kSHCoeffs; }
    uint32_t atlasHeight() const { return m_probeGridH; }

    bool valid() const { return m_probeGridW != 0; }

private:
    Device* m_device = nullptr;

    Image  m_probeAtlas;
    Image  m_filteredAtlas;
    Image  m_prevAtlas;
    Buffer m_rayBuffer;

    std::unique_ptr<rhi::RHITexture> m_probeAtlasTex;
    std::unique_ptr<rhi::RHITextureView> m_probeAtlasView;
    std::unique_ptr<rhi::RHITexture> m_filteredAtlasTex;
    std::unique_ptr<rhi::RHITextureView> m_filteredAtlasView;
    std::unique_ptr<rhi::RHITexture> m_prevAtlasTex;
    std::unique_ptr<rhi::RHITextureView> m_prevAtlasView;
    std::unique_ptr<rhi::RHIBuffer> m_rayBufferRhi;

    uint32_t m_probeGridW = 0;
    uint32_t m_probeGridH = 0;
};

}
