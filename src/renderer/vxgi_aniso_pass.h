#pragma once
#include "core/vk_common.h"
#include "vxgi_resources.h"

// VxgiAnisoPass —— B.6：构建 vxgiAniso 各级 mip。
// 每帧 voxelize+inject+(isotropic) mipmap 跑完后调一次。
// 内部级链：mip 1 从 voxelGrid mip 0 build；mip 2+ 从 aniso mip i-1 build。

namespace somegi {
class Device;

class VxgiAnisoPass {
public:
    void init(Device& d, uint32_t mipLevels);
    void destroy();
    void bindResources(Device& d, const VxgiResources& vxgi);

    // 预条件：voxelGrid 全 mip SHADER_READ_ONLY；aniso 全 mip GENERAL（写）。
    // 结尾：aniso 全 mip GENERAL（write 完）；调用方再转 SHADER_READ_ONLY。
    void record(VkCommandBuffer cmd, const VxgiResources& vxgi);

private:
    Device* m_device = nullptr;
    uint32_t m_mipLevels = 0;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_sets;   // 一组对应一个 dst level
};

}
