#pragma once
#include "core/buffer.h"
#include "core/shader.h"
#include "scene/scene.h"
#include "renderer/core/render_targets.h"
#include "renderer/core/frame_ubo.h"
#include "gi/ibl_baker.h"                 // IblResources
#include "scene/draw_list.h"             // DrawEntry
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

    // Mesh Shader 双模式
    void setMeshShaderEnabled(bool v) { m_useMeshShader = v; }
    bool meshShaderEnabled() const { return m_useMeshShader; }
    // 绑定 Hi-Z mip views 到 mesh descriptor set（每帧 Task Shader 前调用）
    void bindHiZViews(VkImageView mip1, VkImageView mip2, VkImageView mip3, VkImageView mip4);
    // 更新 Task Shader 的 CullUbo
    void buildMeshGroups(const std::vector<DrawEntry>& entries);
    uint32_t meshGroupCount() const { return m_meshGroupCount; }

    void updateCullUbo(const glm::mat4& viewProj, const glm::vec4 frustum[6],
                       uint32_t drawCount, uint32_t hizMaxMip,
                       uint32_t screenW, uint32_t screenH);

private:
    void buildPipeline();
    void buildMeshPipeline();
    void destroyPipeline();

    Device* m_device = nullptr;
    VkFormat m_colorFmt = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFmt = VK_FORMAT_UNDEFINED;

    // VS 路径 set=0
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    // IBL set=1（bindIblResources 时分配，VS+MS 共用）
    VkDescriptorSetLayout m_iblDsl = VK_NULL_HANDLE;
    VkDescriptorPool m_iblPool = VK_NULL_HANDLE;
    VkDescriptorSet m_iblSet = VK_NULL_HANDLE;

    // Mesh Shader 路径
    bool m_useMeshShader = false;
    VkPipelineLayout m_meshPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_meshPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_meshSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_meshPool = VK_NULL_HANDLE;
    VkDescriptorSet m_meshSet = VK_NULL_HANDLE;
    Buffer m_cullUbo;
    Buffer m_meshGroupBuf;
    uint32_t m_meshGroupCount = 0;

    Buffer m_frameUbo;
    uint32_t m_maxTextures = 0;
};

}
