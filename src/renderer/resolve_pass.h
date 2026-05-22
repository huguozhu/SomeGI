#pragma once
#include "core/vk_common.h"
#include <glm/glm.hpp>

namespace somegi {
class Device;
struct SceneGpu;
struct RenderTargets;

class ResolvePass {
public:
    // maxTextures must match the count used when uploading the scene
    void init(Device& d, uint32_t maxTextures);
    void destroy();
    void bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount,
                   VkBuffer frameUbo);
    void bindTargets(Device& d, const RenderTargets& rt, uint32_t frameIdx);
    void record(VkCommandBuffer cmd, const RenderTargets& rt, uint32_t frameIdx);

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_sets[kFramesInFlight]{};
};

}
