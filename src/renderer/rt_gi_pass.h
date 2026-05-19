#pragma once
#include "render_targets.h"
#include "scene_rt_as.h"

namespace somegi {

class Device;
struct SceneGpu;

// RtGiPass — M9 硬件光线追踪 GI（Ray Query 风格）。
//
// 使用 KHR_ray_query 在 compute shader 内发 ray，无需 RT pipeline。
// 每像素发 1 条 cosine-hemisphere ray，命中后计算该点的太阳直接光
// （含 shadow ray），输出到 rtGI（RGBA16F）。
//
// 依赖：TLAS（SceneRtAS）+ 场景 SSBO（顶点/索引/材质）。
//
// 仅当 Device::features().accelStruct && .rayQuery 为 true 时初始化。
class RtGiPass {
public:
    void init(Device& d);
    void destroy();
    void bindFrame(Device& d, const RenderTargets& rt, VkBuffer frameUbo,
                   const SceneRtAS& rtAS, const SceneGpu& sceneGpu);
    void record(VkCommandBuffer cmd, const RenderTargets& rt);

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkSampler m_linearClamp = VK_NULL_HANDLE;
};

}
