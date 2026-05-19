#pragma once
#include "core/vk_common.h"
#include "vxgi_resources.h"
#include "prt_resources.h"
#include <glm/glm.hpp>

// PrtBakePass —— M8.1：一次性烘焙 PRT 体素 visibility transfer SH。
// 算法见 shaders/gi/prt/prt_bake.slang。

namespace somegi {
class Device;

class PrtBakePass {
public:
    void init(Device& d);
    void destroy();
    void bindResources(Device& d, const VxgiResources& vxgi, const PrtResources& prt);

    // 预条件：voxelGrid 已 voxelize 且转 SHADER_READ_ONLY；prtTransfer 在 GENERAL。
    void record(VkCommandBuffer cmd,
                const glm::vec3& prtGridMin, float prtCellSize, uint32_t prtResolution,
                const glm::vec3& vxgiGridMin, float vxgiCellSize, uint32_t vxgiResolution,
                uint32_t numSamples = 64);

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
