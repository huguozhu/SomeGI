#pragma once
#include "core/vk_common.h"
#include <glm/glm.hpp>

// VxgiResolve6AxisPass —— L.3b：isotropic voxelGrid → 3-axis radiance。
//
// 一个 compute dispatch：每 voxel 6 方向 cone-trace → 写 3 张 axis image。

namespace somegi {
class Device;
class VxgiResources;

class VxgiResolve6AxisPass {
public:
    void init(Device& d);
    void destroy();

    void bindResources(Device& d, const VxgiResources& vxgi);
    void record(VkCommandBuffer cmd, uint32_t gridResolution, uint32_t mipLevels,
                float cellSize, const glm::vec3& gridMin, float strength);

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
