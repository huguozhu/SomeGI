#pragma once
#include "core/vk_common.h"
#include "vxgi_resources.h"
#include <glm/glm.hpp>

// VxgiRelightPass —— C.2：multi-bounce voxel relight（Lumen-lite）。
// 每 voxel 朝 6 主轴 cone-trace voxelGrid 自身，把 indirect radiance
// 加回当前 voxel RGB。每次 record 是 1 bounce；调用方 N 次迭代得 N+1
// bounces 视觉。
//
// 需要 ping-pong：读 src voxel 写 dst voxel（不能同 image 边读边写）。
// 调用方在两次迭代之间需 src ↔ dst 的 layout / mip 链管理（复杂）；
// 为简化，本里程碑只支持 **1 次** bounce relight，dst 写完后 mipmap
// 重新生成给 lighting cone trace 用（不再 ping-pong 多轮）。

namespace somegi {
class Device;

class VxgiRelightPass {
public:
    void init(Device& d);
    void destroy();
    // src = 当前帧 voxel grid（mip 0..N，已 inject + mipmap，SHADER_READ_ONLY）；
    // dst = relight 输出（mip 0 storage write，GENERAL）。
    // anisoSrc = 当前帧 vxgiAniso（SHADER_READ_ONLY）。
    void bindResources(Device& d, const VxgiResources& vxgi, VkImageView dstMip0View);
    void record(VkCommandBuffer cmd, uint32_t gridResolution, uint32_t mipLevels,
                float cellSize, const glm::vec3& gridMin, float bounceStrength);

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
