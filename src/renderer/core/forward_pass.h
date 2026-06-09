#pragma once
#include "core/buffer.h"
#include "core/shader.h"
#include "scene/scene.h"
#include "renderer/core/render_targets.h"
#include "renderer/core/frame_ubo.h"
#include "gi/ibl_baker.h"                 // IblResources
#include <glm/glm.hpp>

namespace somegi {
class Device;

class ForwardPass {
public:
    void init(Device& d, VkFormat colorFmt, VkFormat depthFmt, uint32_t maxTextures);
    void destroy();

    // 绑定 IBL 预烘焙资源到 set=1（init 之后、首帧之前调用一次）
    void bindIblResources(Device& d, const IblResources& ibl);

    void bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount);
    void bindDrawData(Device& d, VkBuffer drawDataBuf);
    void updateFrame(const FrameUBO& ubo);

    void record(VkCommandBuffer cmd, const RenderTargets& rt,
                VkBuffer indirectBuf, uint32_t drawCount, const SceneGpu& gpu);

    // 更新 set=0 中的 NDGI MLP 权重 binding
    void setNdgiWeights(Device& d,
        VkBuffer w1, VkBuffer b1, VkBuffer w2, VkBuffer b2,
        VkBuffer w3, VkBuffer b3);

private:
    void buildPipeline();
    void destroyPipeline();

    Device* m_device = nullptr;
    VkFormat m_colorFmt = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFmt = VK_FORMAT_UNDEFINED;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;  // Set=0
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;       // Set=0

    // IBL set=1（bindIblResources 时分配）
    VkDescriptorSetLayout m_iblDsl = VK_NULL_HANDLE;
    VkDescriptorPool m_iblPool = VK_NULL_HANDLE;
    VkDescriptorSet m_iblSet = VK_NULL_HANDLE;

    Buffer m_frameUbo;
    uint32_t m_maxTextures = 0;
};

}
