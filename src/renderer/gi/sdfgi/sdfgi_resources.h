#pragma once
#include "core/image.h"
#include "rhi/base/texture.h"
#include <memory>

// SdfgiResources —— C.3：SDFGI-lite 用的 3D 资源。
//
// 三张 3D 图：
//   - seedA / seedB（RGBA16F 128³）：JFA ping-pong。rgb=最近 seed cell 坐标，
//     a=有效标志（1 表示已持有 seed，0 = sentinel）。
//   - udf（R16F 128³）：JFA 收敛后 finalize 出的 unsigned distance field，
//     单位 cell。trace pass 用其做 sphere-stepping。
//
// 大小估算（128³）：
//   seedA + seedB = 2 × 128³ × 8B  = 32 MB
//   udf           = 128³ × 2B      = 4 MB
//   合计           ~36 MB
//
// 仅在 SDFGI 模式下分配；其它 GI 模式不创建（节省显存）。

namespace somegi {
class Device;
namespace rhi { class RHIDevice; }

class SdfgiResources {
public:
    void create(Device& d, rhi::RHIDevice& rhiD, uint32_t resolution);
    void destroy();

    bool created() const { return m_resolution != 0; }
    uint32_t resolution() const { return m_resolution; }

    const Image& seedA() const { return m_seedA; }
    const Image& seedB() const { return m_seedB; }
    const Image& udf()   const { return m_udf;   }

    rhi::RHITexture* seedARhiTex() const { return m_seedATex.get(); }
    rhi::RHITextureView* seedARhiView() const { return m_seedAView.get(); }
    rhi::RHITexture* seedBRhiTex() const { return m_seedBTex.get(); }
    rhi::RHITextureView* seedBRhiView() const { return m_seedBView.get(); }
    rhi::RHITexture* udfRhiTex() const { return m_udfTex.get(); }
    rhi::RHITextureView* udfRhiView() const { return m_udfView.get(); }

private:
    Device* m_device = nullptr;
    Image m_seedA;
    Image m_seedB;
    Image m_udf;
    std::unique_ptr<rhi::RHITexture> m_seedATex;
    std::unique_ptr<rhi::RHITextureView> m_seedAView;
    std::unique_ptr<rhi::RHITexture> m_seedBTex;
    std::unique_ptr<rhi::RHITextureView> m_seedBView;
    std::unique_ptr<rhi::RHITexture> m_udfTex;
    std::unique_ptr<rhi::RHITextureView> m_udfView;
    uint32_t m_resolution = 0;
};

}
