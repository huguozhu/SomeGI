#pragma once
#include "core/vk_common.h"
#include "vxgi_resources.h"
#include <vector>

// VxgiMipmapPass —— M7.2：iterate 从 mip 1 到 mipMax，每级 dispatch 一次
// reduction。算法见 shaders/gi/vxgi/vxgi_mipmap.slang。
//
// 每级需要不同的 (src, dst) 描述符，所以本类持有 mipMax-1 组 descriptor
// set，bindResources 时一次性填好。

namespace somegi {
class Device;

class VxgiMipmapPass {
public:
    void init(Device& d, uint32_t mipLevels);
    void destroy();

    // 把 vxgi 的每级 mip view 填到对应的 descriptor set。M7 单 cascade 这
    // 些资源不变，scene 切换不需要 rebind。
    void bindResources(Device& d, const VxgiResources& vxgi);

    // 录制：从 mip 1 到 mipLevels-1 逐级 dispatch。每级前后做 barrier
    // （上一级 GENERAL→SHADER_READ_ONLY，本级 GENERAL）。
    void record(VkCommandBuffer cmd, const VxgiResources& vxgi);

private:
    Device* m_device = nullptr;
    uint32_t m_mipLevels = 0;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_sets;   // 一组对应一级 mip（dst level）
};

}
