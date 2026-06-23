#pragma once
#include "core/image.h"
#include "core/buffer.h"
#include "rhi/base/texture.h"
#include "rhi/base/buffer.h"
#include <glm/glm.hpp>
#include <vector>
#include <memory>

// RestirResources —— C.4 ReSTIR DI 资源。
//
// 拥有：
//   - 2 张 reservoir image（RGBA32_UINT，screen-res，ping-pong；init 写 A，
//     spatial 读 A 写 B，shade 读 B）
//   - light SSBO（PointLight 数组，HOST_VISIBLE，CPU 每帧更新）
//
// reservoir 像素打包：见 shaders/common/restir_common.slang
//   r=lightIdx, g=asuint(wSum), b=asuint(W), a=M

namespace somegi {
class Device;
namespace rhi { class RHIDevice; }

struct PointLightCpu {
    glm::vec3 pos;
    float     radius;
    glm::vec3 color;
    float     intensity;
};
static_assert(sizeof(PointLightCpu) == 32,
    "PointLightCpu 必须与 shader 端 PointLight 紧凑布局一致 (8 floats = 32 bytes)");

class RestirResources {
public:
    void create(Device& d, rhi::RHIDevice& rhiD, VkExtent2D screenExtent, uint32_t maxLights);
    void destroy();
    void resize(Device& d, VkExtent2D newExtent);   // 只重建 reservoir，light 不变

    // CPU 每帧调用 → memcpy 到 mapped pointer
    void updateLights(const std::vector<PointLightCpu>& lights);

    bool created() const { return m_screenExtent.width != 0; }
    VkExtent2D screenExtent() const { return m_screenExtent; }
    const Image& reservoirA() const { return m_reservoirA; }
    const Image& reservoirB() const { return m_reservoirB; }
    VkBuffer lightBuffer() const { return m_lightBuf.handle(); }
    uint32_t maxLights() const { return m_maxLights; }
    uint32_t currentLightCount() const { return m_lightCount; }

    rhi::RHITexture* reservoirARhiTex() const { return m_reservoirATex.get(); }
    rhi::RHITextureView* reservoirARhiView() const { return m_reservoirAView.get(); }
    rhi::RHITexture* reservoirBRhiTex() const { return m_reservoirBTex.get(); }
    rhi::RHITextureView* reservoirBRhiView() const { return m_reservoirBView.get(); }
    rhi::RHIBuffer* lightBufRhi() const { return m_lightBufRhi.get(); }

private:
    void createReservoirImages(Device& d, VkExtent2D ext);

    rhi::RHIDevice* m_rhiDevice = nullptr;
    Image m_reservoirA;
    Image m_reservoirB;
    Buffer m_lightBuf;
    std::unique_ptr<rhi::RHITexture> m_reservoirATex;
    std::unique_ptr<rhi::RHITextureView> m_reservoirAView;
    std::unique_ptr<rhi::RHITexture> m_reservoirBTex;
    std::unique_ptr<rhi::RHITextureView> m_reservoirBView;
    std::unique_ptr<rhi::RHIBuffer> m_lightBufRhi;
    VkExtent2D m_screenExtent{};
    uint32_t m_maxLights = 0;
    uint32_t m_lightCount = 0;
};

}
