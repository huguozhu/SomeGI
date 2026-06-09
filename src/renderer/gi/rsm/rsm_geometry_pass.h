#pragma once
#include "core/buffer.h"
#include "core/image.h"
#include "core/shader.h"
#include "scene/scene.h"
#include <glm/glm.hpp>

// RsmGeometryPass —— M5 RSM 的几何阶段。
//
// 从 sun 视角 ortho 投影渲场景，输出 4 张 RT：
//   RT0 rsmPosition  RGBA16F  xyz=worldPos
//   RT1 rsmNormal    RGBA16F  xyz=worldNormal（归一化）
//   RT2 rsmFlux      RGBA16F  rgb=flux=albedo·sunColor·sunIntensity·NdotL
//   depth            D32      给后续 indirect shadow 留位（M5.0 不消费）
//
// 自己持有 4 张 image（512² 固定大小，与 swapchain 解耦）+ RsmFrameUbo
// （sun viewProj + sunDir + sunColor·intensity）。归 RsmTechnique 使用。
//
// 调用顺序：init → bindScene → 每帧 updateLight + record。
// scene 切换（cube ↔ Sponza）时 AABB 变 → 调用方需 updateLight 重算 ortho。

namespace somegi {
class Device;

class RsmGeometryPass {
public:
    void init(Device& d, uint32_t maxTextures);
    void destroy();

    // scene 切换时重新写场景描述符。
    void bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount);

    // 计算 sun 的 ortho viewProj 让其 frustum 正好包住 scene AABB；
    // 写到 RsmFrameUbo（host-coherent，立即生效）。
    // sunDir：光传播方向（从 sun 指向 surface，与主 FrameUBO 同约定，
    //         调用方传 m_sunDir 即可，不需要先取反）。
    // sunColor：xyz=color；sunIntensity：标量。
    void updateLight(const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                     const glm::vec3& sunDir,
                     const glm::vec3& sunColor, float sunIntensity);

    // 录制 sun-view MRT 渲染。layout 转换由本方法内部完成：调用前后
    // 4 张 RT 都在 SHADER_READ_ONLY_OPTIMAL（方便后续 RsmSamplePass 直
    // 接读）。第一帧的 UNDEFINED → COLOR_ATTACHMENT 也正确处理。
    void record(VkCommandBuffer cmd, const SceneCpu& cpu, const SceneGpu& gpu);

    // 给 RsmSamplePass / Lighting 取这 4 张图与 UBO。
    const Image& position() const { return m_position; }
    const Image& normal()   const { return m_normal;   }
    const Image& flux()     const { return m_flux;     }
    const Image& depth()    const { return m_depth;    }
    VkBuffer frameUboHandle() const { return m_rsmFrameUbo.handle(); }

    static constexpr uint32_t kRsmSize = 512;

private:
    void buildPipeline();
    void destroyPipeline();

    Device* m_device = nullptr;

    // 4 张 RT —— 固定 512²，由本类持有。Image 的 RAII 析构会回收 GPU 资源。
    Image m_position;
    Image m_normal;
    Image m_flux;
    Image m_depth;

    // RsmFrameUniforms：sun 视角 viewProj + sun 光参数。host-coherent 映射，
    // updateLight memcpy 即时可见。
    Buffer m_rsmFrameUbo;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    uint32_t m_maxTextures = 0;
};

}
