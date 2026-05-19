#pragma once
#include "core/vk_common.h"
#include "vxgi_resources.h"
#include "ddgi_resources.h"
#include <glm/glm.hpp>

// DdgiPass —— DDGI 每帧 pipeline：
//   1) update：每 probe 投 N rays，march voxel grid 取命中数据，写到 ray buffer
//   2) blend irradiance：per atlas texel 把 ray buffer 按方向 cosine 累加
//   3) blend distance：同上但 sharper kernel + 存 mean / mean²
// 三个 dispatch 串成一帧。

namespace somegi {
class Device;

class DdgiPass {
public:
    void init(Device& d);
    void destroy();

    // ddgi resources + voxel grid (作为 ray source)。voxel grid 每帧由 App
    // 重新 voxelize+mipmap 提供（同 PRT bake 那条流程，但每帧重做让 dynamic
    // sun / 物体动起来 GI 跟得上）。
    void bindResources(Device& d, const DdgiResources& ddgi, const VxgiResources& vxgi);

    // 预条件：voxel grid all mips SHADER_READ_ONLY；ddgi ray buffer + atlases
    // 在 GENERAL（首次调用前 App 一次性 transition）。
    void record(VkCommandBuffer cmd, const DdgiResources& ddgi,
                const glm::vec3& ddgiOrigin, const glm::vec3& ddgiSpacing,
                const glm::vec3& vxgiGridMin, float vxgiCellSize, uint32_t vxgiResolution,
                float randomRotation, uint32_t frameIndex);

private:
    Device* m_device = nullptr;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipelineUpdate = VK_NULL_HANDLE;
    VkPipeline m_pipelineClassify = VK_NULL_HANDLE;   // B.5
    VkPipeline m_pipelineBlendIrr = VK_NULL_HANDLE;
    VkPipeline m_pipelineBlendDist = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_setUpdate = VK_NULL_HANDLE;   // voxelGrid + sampler + rayBuf
    VkDescriptorSet m_setBlend = VK_NULL_HANDLE;    // rayBuf + irrAtlas + distAtlas
    VkDescriptorSet m_setClassify = VK_NULL_HANDLE; // B.5: rayBuf + probeStates

    VkDescriptorSetLayout m_setLayoutBlend = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayoutBlend = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayoutClassify = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayoutClassify = VK_NULL_HANDLE;

    VkSampler m_linearClamp = VK_NULL_HANDLE;
};

}
