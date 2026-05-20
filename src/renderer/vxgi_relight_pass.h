#pragma once
#include "core/vk_common.h"
#include "vxgi_resources.h"
#include <glm/glm.hpp>

// VxgiRelightPass —— C.2 + L.3a：multi-bounce voxel relight。
//
// 每 voxel 朝 6 主轴 cone-trace 源 voxel grid，间接光加回当前 voxel。
// 单次 record = 1 bounce；multi-bounce 需 ping-pong 两张 scratch image。
//
// set "srcVoxel" = read voxelGrid + aniso → write scratch
// set "pingPong" = read scratch → write scratch2（交替 src/dst）
//
// L.3a 升级：bindResourcesPingPong + 第二个 scratch。

namespace somegi {
class Device;

class VxgiRelightPass {
public:
    void init(Device& d);
    void destroy();

    // Bounce 1: read voxelGrid → write dst (scratch)
    void bindResources(Device& d, const VxgiResources& vxgi, VkImageView dstMip0View);

    // Bounce 2+: read scratch → write scratch2 (ping-pong)
    // swap=true: read scratch2, write scratch; swap=false: read scratch, write scratch2
    void bindResourcesPingPong(Device& d, const VxgiResources& vxgi, bool swap);

    void record(VkCommandBuffer cmd, VkDescriptorSet set,
                uint32_t gridResolution, uint32_t mipLevels,
                float cellSize, const glm::vec3& gridMin, float bounceStrength);

    VkDescriptorSet voxelSet()  const { return m_set; }
    VkDescriptorSet pingSet0()  const { return m_setPP0; }
    VkDescriptorSet pingSet1()  const { return m_setPP1; }

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;        // voxelGrid→scratch
    VkDescriptorSet m_setPP0 = VK_NULL_HANDLE;     // scratch→scratch2
    VkDescriptorSet m_setPP1 = VK_NULL_HANDLE;     // scratch2→scratch
    VkSampler m_linearClamp = VK_NULL_HANDLE;
};

}
