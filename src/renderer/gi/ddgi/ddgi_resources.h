#pragma once
#include "core/buffer.h"
#include "core/image.h"
#include "rhi/base/texture.h"
#include "rhi/base/buffer.h"
#include <memory>

// DdgiResources —— Majercik 2019 DDGI（Dynamic Diffuse Global Illumination）
// 的 probe atlas + ray buffer。
//
// 256 probes (8×4×8) 按 scene AABB 等距摆。每 probe 存：
//   - irradiance octahedral 8×8（cosine-convolved radiance）→ atlas 大小
//     8*8 = 64 wide × 4*8 = 32 high probes laid out (probesX, probesY*probesZ)
//     像素 = (probesX·octaIrr, probesY·probesZ·octaIrr) RGBA16F
//   - distance octahedral 16×16（mean + mean² 给 Chebyshev VSM）→ atlas
//     RG16F，layout 同上但 octaDist=16
//
// 每帧 update：64 rays/probe Fibonacci 球面采样，march voxel grid 取
// 命中 radiance + distance；blend 阶段 per-texel 把 64 rays 按方向权重
// 累加 + 与上帧 atlas 做 0.05 alpha 时序 blend。

namespace somegi {
class Device;
namespace rhi { class RHIDevice; }

class DdgiResources {
public:
    static constexpr uint32_t kProbesX = 8;
    static constexpr uint32_t kProbesY = 4;
    static constexpr uint32_t kProbesZ = 8;
    static constexpr uint32_t kProbeCount = kProbesX * kProbesY * kProbesZ;
    static constexpr uint32_t kOctaIrr = 8;       // irradiance 边长
    static constexpr uint32_t kOctaDist = 16;     // distance/visibility 边长
    static constexpr uint32_t kRaysPerProbe = 64; // 每帧每 probe 投多少 ray

    void create(Device& d, rhi::RHIDevice& rhiD);
    void destroy();

    const Image& irradiance() const { return m_irradiance; }
    const Image& distance() const { return m_distance; }
    const Buffer& rayBuffer() const { return m_rayBuffer; }
    const Buffer& probeStates() const { return m_probeStates; }

    rhi::RHITexture* irradianceRhiTex() const { return m_irradianceTex.get(); }
    rhi::RHITextureView* irradianceRhiView() const { return m_irradianceView.get(); }
    rhi::RHITexture* distanceRhiTex() const { return m_distanceTex.get(); }
    rhi::RHITextureView* distanceRhiView() const { return m_distanceView.get(); }
    rhi::RHIBuffer* rayBufferRhi() const { return m_rayBufferRhi.get(); }
    rhi::RHIBuffer* probeStatesRhi() const { return m_probeStatesRhi.get(); }

    // atlas 尺寸（lighting 端 sampler UV 计算用）
    static uint32_t irradianceAtlasW() { return kProbesX * kOctaIrr; }
    static uint32_t irradianceAtlasH() { return kProbesY * kProbesZ * kOctaIrr; }
    static uint32_t distanceAtlasW()   { return kProbesX * kOctaDist; }
    static uint32_t distanceAtlasH()   { return kProbesY * kProbesZ * kOctaDist; }

private:
    Device* m_device = nullptr;
    Image m_irradiance;
    Image m_distance;
    Buffer m_rayBuffer;   // probe×ray 共享 SSBO，存 (rayDir+pad, hitRgb+hitDist)
    Buffer m_probeStates; // B.5 classification：每 probe 一个 uint，1=active 0=inactive

    std::unique_ptr<rhi::RHITexture> m_irradianceTex;
    std::unique_ptr<rhi::RHITextureView> m_irradianceView;
    std::unique_ptr<rhi::RHITexture> m_distanceTex;
    std::unique_ptr<rhi::RHITextureView> m_distanceView;
    std::unique_ptr<rhi::RHIBuffer> m_rayBufferRhi;
    std::unique_ptr<rhi::RHIBuffer> m_probeStatesRhi;
};

}
