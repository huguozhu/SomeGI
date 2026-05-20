#pragma once
#include "core/buffer.h"
#include "core/shader.h"
#include "scene/scene.h"
#include "render_targets.h"
#include <glm/glm.hpp>

namespace somegi {
class Device;
class IGITechnique;

struct FrameUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
    glm::mat4 invViewProj;      // M4 deferred world-pos reconstruction
    glm::mat4 prevViewProj;     // B.4 SSGI 时序累积：上一帧 viewProj 用于 reproject
    glm::vec4 cameraPos;
    glm::vec4 sunDir;
    glm::vec4 sunColor_intensity;
    glm::vec4 ambient;
    glm::ivec4 counts;          // x=materialCount, y=iblSpecularMips,
                                // z=iblEnabled (0/1), w=rsmEnabled (0/1)
    glm::ivec4 lpvCounts;       // x=gridResolution, y=lpvEnabled (0/1)
    glm::vec4 lpvGridMinCell;   // xyz=gridMin, w=cellSize
    glm::ivec4 vxgiCounts;      // M7: x=gridResolution, y=vxgiEnabled, z=mipLevels
    glm::vec4 vxgiGridMinCell;  // xyz=gridMin, w=cellSize
    glm::ivec4 prtCounts;       // M8: x=gridResolution, y=prtEnabled
    glm::vec4 prtGridMinCell;
    glm::vec4 prtLightSH_R;     // SH4 系数（R 通道），CPU 每帧投影 sun
    glm::vec4 prtLightSH_G;
    glm::vec4 prtLightSH_B;
    glm::vec4 prtLightSH9_R0;   // B.9 SH9 扩展：l=2 共 5 系数 / 通道
    glm::vec4 prtLightSH9_R1;
    glm::vec4 prtLightSH9_G0;
    glm::vec4 prtLightSH9_G1;
    glm::vec4 prtLightSH9_B0;
    glm::vec4 prtLightSH9_B1;
    glm::vec4 prtLightSH16_R0;  // B.10 SH16 扩展：l=3 共 7 系数 / 通道
    glm::vec4 prtLightSH16_R1;
    glm::vec4 prtLightSH16_G0;
    glm::vec4 prtLightSH16_G1;
    glm::vec4 prtLightSH16_B0;
    glm::vec4 prtLightSH16_B1;
    glm::ivec4 ddgiCounts;      // M11: probesX, probesY, probesZ, enabled
    glm::vec4 ddgiOrigin;
    glm::vec4 ddgiSpacing;
    glm::ivec4 ddgiOctaSizes;   // octaIrr, octaDist
    glm::ivec4 lumenCounts;      // Lumen-lite: x=lumenEnabled (0/1)
};

class ForwardPass {
public:
    void init(Device& d, VkFormat colorFmt, VkFormat depthFmt, uint32_t maxTextures);
    void destroy();

    void bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount);
    void updateFrame(const FrameUBO& ubo);

    // Switch to a GI technique (or nullptr for the default no-IBL variant).
    // Rebuilds pipeline + pipelineLayout. Caller must vkDeviceWaitIdle if
    // any prior frames may still be in flight.
    void setTechnique(IGITechnique* tech);

    void record(VkCommandBuffer cmd, const RenderTargets& rt,
                const SceneCpu& cpu, const SceneGpu& gpu);

private:
    void buildPipeline(const char* variant, VkDescriptorSetLayout giDsl);
    void destroyPipeline();

    Device* m_device = nullptr;
    VkFormat m_colorFmt = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFmt = VK_FORMAT_UNDEFINED;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;  // Set=0
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;       // Set=0

    Buffer m_frameUbo;
    uint32_t m_maxTextures = 0;

    IGITechnique* m_tech = nullptr;
};

}
