#pragma once
#include "core/image.h"
#include "core/buffer.h"
#include <glm/glm.hpp>
#include <vector>

namespace somegi {

class Device;
struct RenderTargets;
struct SceneGpu;
struct DrawEntry;

enum class ShadowMethod : int {
    None = 0,
    HardShadowMap = 1,
    PCF = 2,
    PCSS = 3,
    VSM = 4,
    RTHard = 5,    // Phase 2
    RTSoft = 6,    // Phase 2
    Count
};

struct ShadowEntry {
    const char* name;
    bool implemented;
    bool requiresRt;
};

// Phase 1 available algorithms (RT deferred to Phase 2)
constexpr ShadowEntry kShadows[] = {
    {"None",               true,  false},
    {"Hard Shadow Map",    true,  false},
    {"PCF Soft Shadow",    true,  false},
    {"PCSS Soft Shadow",   true,  false},
    {"VSM Soft Shadow",    true,  false},
    {"RT Hard Shadow",     true,  true},
    {"RT Soft Shadow",     true,  true},
};
constexpr int kShadowCount = (int)(sizeof(kShadows) / sizeof(kShadows[0]));

class ShadowPass {
public:
    void init(Device& d, VkExtent2D shadowMapSize, VkExtent2D outputSize);
    void destroy();

    // Per-frame record: dispatches based on m_method
    void record(VkCommandBuffer cmd, const RenderTargets& rt,
                VkBuffer frameUbo, const SceneGpu& sceneGpu,
                VkBuffer indirectBuf, uint32_t drawCount,
                uint32_t frameIndex = 0);

    // Algorithm selection
    ShadowMethod method() const { return m_method; }
    void setMethod(ShadowMethod m) { m_method = m; }

    // Scene AABB for sun-view projection computation
    void setSceneAabb(const glm::vec3& mn, const glm::vec3& mx) {
        m_sceneAabbMin = mn; m_sceneAabbMax = mx;
    }

    // Sun direction（lightDir = from sun to surface，与 FrameUBO::sunDir 一致）
    void setSunDir(const glm::vec3& dir) { m_sunDir = dir; }

    // 绑定场景 GPU 资源到 SM render descriptor set（场景加载后、record 前调用）
    void bindScene(Device& d, const SceneGpu& gpu);

    // 绑定每帧资源（FrameUniforms + GBuffer depth + normal）到 resolve set
    void bindFrameResources(Device& d, VkBuffer frameUbo, VkImageView depthView, VkImageView normalView);

    // 绑定 TLAS（RT shadow 用，场景加载后调用）
    void bindTLAS(Device& d, VkAccelerationStructureKHR tlas);

    // Shadow output (R8_UNORM) — LightingPass reads from here
    const Image& shadowMask() const { return m_shadowMask; }

    // Sampler used for shadow map sampling (exposed for LightingPass)
    VkSampler shadowSampler() const { return m_shadowSampler; }

private:
    void recordNone(VkCommandBuffer cmd);
    void recordHardSM(VkCommandBuffer cmd, const RenderTargets& rt,
                      VkBuffer frameUbo, const SceneGpu& sceneGpu,
                      VkBuffer indirectBuf, uint32_t drawCount);
    void recordPCF(VkCommandBuffer cmd, const RenderTargets& rt,
                   VkBuffer frameUbo, const SceneGpu& sceneGpu,
                   VkBuffer indirectBuf, uint32_t drawCount);
    void recordPCSS(VkCommandBuffer cmd, const RenderTargets& rt,
                    VkBuffer frameUbo, const SceneGpu& sceneGpu,
                    VkBuffer indirectBuf, uint32_t drawCount);
    void recordVSM(VkCommandBuffer cmd, const RenderTargets& rt,
                   VkBuffer frameUbo, const SceneGpu& sceneGpu,
                   VkBuffer indirectBuf, uint32_t drawCount);
    void recordRTHard(VkCommandBuffer cmd);
    void recordRTSoft(VkCommandBuffer cmd);

    // Shared shadow map render (sun-view depth-only)
    void renderShadowMap(VkCommandBuffer cmd, VkBuffer frameUbo,
                         const SceneGpu& sceneGpu,
                         VkBuffer indirectBuf, uint32_t drawCount);

    void buildPipeline_HardSM();
    void buildPipeline_VSMGen();
    void buildPipeline_VSMBlur();
    void buildPipeline_RTHard();
    void buildResolvePipeline();
    void destroyPipelines();

    Device* m_device = nullptr;
    ShadowMethod m_method = ShadowMethod::HardShadowMap;
    uint32_t m_currentFrameIndex = 0;  // 用于 RT soft shadow 随机种子

    // Shadow map target (D32_SFLOAT, 2048x2048)
    Image m_shadowMap;
    VkExtent2D m_shadowMapSize{2048, 2048};

    // Output shadowMask (R8_UNORM, full resolution)
    Image m_shadowMask;
    VkExtent2D m_outputSize{};

    // VSM-specific: depth + depth^2 (R32G32_SFLOAT)
    Image m_vsmMap;
    Image m_vsmBlur;     // 2×2 box blur 中间结果

    // Shadow map render pipelines
    VkPipelineLayout m_smPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_smPipeline = VK_NULL_HANDLE;          // Hard SM: depth-only
    VkPipeline m_vsmGenPipeline = VK_NULL_HANDLE;      // VSM: depth+depth^2

    // shadowMask resolve pipelines
    VkPipelineLayout m_resolveLayout = VK_NULL_HANDLE;
    VkPipeline m_resolveHard = VK_NULL_HANDLE;
    VkPipeline m_resolvePCF = VK_NULL_HANDLE;
    VkPipeline m_resolveVSM = VK_NULL_HANDLE;

    // PCSS resolve（独立 pipeline layout，push constant 含 lightSize）
    VkPipelineLayout m_pcssResolveLayout = VK_NULL_HANDLE;
    VkPipeline m_resolvePCSS = VK_NULL_HANDLE;

    // Descriptors（resolve compute：UBO + COMBINED_IMAGE_SAMPLER + STORAGE_IMAGE）
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    // 每帧资源 descriptor set（set=1：FrameUniforms UBO + GBuffer depth）
    VkDescriptorSetLayout m_frameSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_framePool = VK_NULL_HANDLE;
    VkDescriptorSet m_frameSet = VK_NULL_HANDLE;

    // SM graphics pipeline descriptors（UBO + SSBO vertices/indices/drawData）
    VkDescriptorSetLayout m_smSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_smPool = VK_NULL_HANDLE;
    VkDescriptorSet m_smSet = VK_NULL_HANDLE;

    // VSM blur pipeline（compute：vsmMap → vsmBlur）
    VkPipelineLayout m_vsmBlurLayout = VK_NULL_HANDLE;
    VkPipeline m_vsmBlurPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_vsmBlurSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_vsmBlurPool = VK_NULL_HANDLE;
    VkDescriptorSet m_vsmBlurSet = VK_NULL_HANDLE;

    // RT shadow（仅 HW 支持时创建）
    VkPipelineLayout m_rtHardLayout = VK_NULL_HANDLE;
    VkPipeline m_rtHardPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_rtSoftLayout = VK_NULL_HANDLE;
    VkPipeline m_rtSoftPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_rtSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_rtPool = VK_NULL_HANDLE;
    VkDescriptorSet m_rtSet = VK_NULL_HANDLE;
    VkAccelerationStructureKHR m_tlas = VK_NULL_HANDLE;

    VkSampler m_shadowSampler = VK_NULL_HANDLE;   // depth compare (PCF 用)
    VkSampler m_vsmSampler    = VK_NULL_HANDLE;   // 线性、无 compare（VSM 用）

    // Shadow view/proj UBO (sun-space)
    Buffer m_shadowUbo;

    // Index buffer saved from bindScene for vkCmdBindIndexBuffer
    VkBuffer m_indexBuffer = VK_NULL_HANDLE;

    // Sun direction（lightDir，setSunDir 或 record 时从调用方更新）
    glm::vec3 m_sunDir{0.0f, -1.0f, 0.0f};

    // Scene bounds for sun-view projection
    glm::vec3 m_sceneAabbMin{0};
    glm::vec3 m_sceneAabbMax{0};
};

} // namespace somegi
