#pragma once
#include "core/vk_common.h"
#include "scene/scene.h"
#include "vxgi_resources.h"
#include <glm/glm.hpp>

// VxgiVoxelizePass —— M7.0：每帧把场景三角形 scatter 到 128³ voxel mip 0。
// compute scatter（不是 GS 保守光栅），每 thread 一个三角形。算法见
// shaders/gi/vxgi/vxgi_voxelize.slang。
//
// 调用方约定：record 之前 voxelGrid mip 0 已 transition 到 GENERAL（写）；
// record 内部按 primitive 多次 dispatch（每次 push 不同的 model + material）；
// record 结束后 voxelGrid mip 0 仍在 GENERAL，调用方再做后续 inject /
// mipmap / barrier。

namespace somegi {
class Device;

class VxgiVoxelizePass {
public:
    void init(Device& d, uint32_t maxTextures);
    void destroy();

    // 写场景级 binding：vertex / index / material / texture array。voxelGrid
    // 的 mip 0 view 从 VxgiResources 取，scene 切换 / VXGI 资源重建时调。
    void bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount,
                   const VxgiResources& vxgi);

    // 录制：遍历 cpu.nodes / mesh.primitives，逐 primitive 推 push constants
    // + dispatch (triCount + 63)/64。grid 几何参数（gridMin, cellSize,
    // resolution）从 App 当前帧值传入。
    void record(VkCommandBuffer cmd, const SceneCpu& cpu, const SceneGpu& gpu,
                const glm::vec3& gridMin, float cellSize, uint32_t gridResolution);

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    uint32_t m_maxTextures = 0;
};

}
