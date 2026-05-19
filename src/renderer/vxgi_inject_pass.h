#pragma once
#include "core/vk_common.h"
#include "vxgi_resources.h"
#include <glm/glm.hpp>

// VxgiInjectPass —— M7.1：把 RSM (pos, flux) 写到 voxelGrid mip 0 的
// 对应 cell 的 RGB（保留 voxelize 阶段写的 alpha=1）。
//
// 与 LpvInjectPass 同思路但没有 SH 投影 —— voxel 的 RGB 直接当 outgoing
// radiance 用。

namespace somegi {
class Device;
class Image;

class VxgiInjectPass {
public:
    void init(Device& d, uint32_t rsmSize);
    void destroy();

    // RSM 2 张图（pos, flux）+ voxelGrid mip 0 view 写到 set=0。
    void bindResources(Device& d,
                       const Image& rsmPos, const Image& rsmFlux,
                       const VxgiResources& vxgi);

    // 预条件：voxelGrid mip 0 在 GENERAL；RSM pos/flux 在 SHADER_READ_ONLY。
    void record(VkCommandBuffer cmd, uint32_t gridResolution,
                const glm::vec3& gridMin, float cellSize);

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    uint32_t m_rsmSize = 0;
};

}
