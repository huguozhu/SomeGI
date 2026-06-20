// ShadowPass — 多种阴影算法的统一录制入口。已迁移到 RHI（resources），record 保留 VkCompat。
#pragma once
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/sampler.h"
#include "core/image.h"
#include "core/buffer.h"
#include <glm/glm.hpp>
#include <vector>

namespace somegi {

class Device;
struct RenderTargets;
struct SceneGpu;
struct DrawEntry;
namespace rhi { class RHICommandBuffer; class RHIBuffer; }

enum class ShadowMethod : int {
    None = 0,
    HardShadowMap = 1,
    PCF = 2,
    PCSS = 3,
    VSM = 4,
    RTHard = 5,
    RTSoft = 6,
    Count
};

struct ShadowEntry {
    const char* name;
    bool implemented;
    bool requiresRt;
};

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
    void init(Device& d, rhi::RHIDevice& rhiDevice, VkExtent2D shadowMapSize, VkExtent2D outputSize);
    void destroy();

    void record(rhi::RHICommandBuffer& cmd, const RenderTargets& rt,
                const rhi::RHIBuffer& frameUbo, const SceneGpu& sceneGpu,
                const rhi::RHIBuffer& indirectBuf, uint32_t drawCount,
                uint32_t frameIndex = 0);
    void record(VkCommandBuffer cmd, const RenderTargets& rt,
                VkBuffer frameUbo, const SceneGpu& sceneGpu,
                VkBuffer indirectBuf, uint32_t drawCount,
                uint32_t frameIndex = 0);

    ShadowMethod method() const { return m_method; }
    void setMethod(ShadowMethod m) { m_method = m; }

    void setSceneAabb(const glm::vec3& mn, const glm::vec3& mx) {
        m_sceneAabbMin = mn; m_sceneAabbMax = mx;
    }
    void setSunDir(const glm::vec3& dir) { m_sunDir = dir; }
    void bindScene(Device& d, const SceneGpu& gpu);
    void bindFrameResources(Device& d, VkBuffer frameUbo, VkImageView depthView, VkImageView normalView);
    void bindTLAS(Device& d, VkAccelerationStructureKHR tlas);

    const Image& shadowMask() const { return m_shadowMask; }
    VkSampler shadowSampler() const { return (VkSampler)(uintptr_t)m_shadowSampler->nativeHandle(); }

    void setFgAutoBarrier(bool v) { m_fgAutoBarrier = v; }
    bool fgAutoBarrier() const { return m_fgAutoBarrier; }

    float& rtSunRadius()  { return m_rtSunRadius; }
    int&   rtRayCount()   { return m_rtRayCount; }

private:
    void recordNone(rhi::RHICommandBuffer& cmd);
    void recordNone(VkCommandBuffer cmd);
    void recordHardSM(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& indirectBuf, uint32_t drawCount);
    void recordHardSM(VkCommandBuffer cmd, const RenderTargets& rt,
                      VkBuffer frameUbo, const SceneGpu& sceneGpu,
                      VkBuffer indirectBuf, uint32_t drawCount);
    void recordPCF(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& indirectBuf, uint32_t drawCount);
    void recordPCF(VkCommandBuffer cmd, const RenderTargets& rt,
                   VkBuffer frameUbo, const SceneGpu& sceneGpu,
                   VkBuffer indirectBuf, uint32_t drawCount);
    void recordPCSS(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& indirectBuf, uint32_t drawCount);
    void recordPCSS(VkCommandBuffer cmd, const RenderTargets& rt,
                    VkBuffer frameUbo, const SceneGpu& sceneGpu,
                    VkBuffer indirectBuf, uint32_t drawCount);
    void recordVSM(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& indirectBuf, uint32_t drawCount);
    void recordVSM(VkCommandBuffer cmd, const RenderTargets& rt,
                   VkBuffer frameUbo, const SceneGpu& sceneGpu,
                   VkBuffer indirectBuf, uint32_t drawCount);
    void recordRTHard(rhi::RHICommandBuffer& cmd);
    void recordRTHard(VkCommandBuffer cmd);
    void recordRTSoft(rhi::RHICommandBuffer& cmd);
    void recordRTSoft(VkCommandBuffer cmd);
    void renderShadowMap(rhi::RHICommandBuffer& cmd, const rhi::RHIBuffer& indirectBuf, uint32_t drawCount);
    void renderShadowMap(VkCommandBuffer cmd, VkBuffer frameUbo,
                         const SceneGpu& sceneGpu,
                         VkBuffer indirectBuf, uint32_t drawCount);

    void buildPipeline_HardSM();
    void buildPipeline_VSMGen();
    void buildPipeline_VSMBlur();
    void buildResolvePipeline();
    void buildPipeline_RTHard();
    void destroyPipelines();

    Device* m_device = nullptr;
    rhi::RHIDevice* m_rhiDevice = nullptr;
    ShadowMethod m_method = ShadowMethod::HardShadowMap;
    uint32_t m_currentFrameIndex = 0;

    // ── Images（保留 VK，未迁移到 RHITexture）──
    Image m_shadowMap;
    VkExtent2D m_shadowMapSize{2048, 2048};
    Image m_shadowMask;
    VkExtent2D m_outputSize{};
    Image m_vsmMap;
    Image m_vsmBlur;

    // ── Samplers（RHI）──
    std::unique_ptr<rhi::RHISampler> m_shadowSampler;  // depth compare
    std::unique_ptr<rhi::RHISampler> m_vsmSampler;     // 线性

    // ── Descriptor Set Layouts（RHI）──
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;        // Resolve compute
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_smSetLayout;      // SM graphics
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_frameSetLayout;   // Frame UBO+depth
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_vsmBlurSetLayout; // VSM blur
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_rtSetLayout;      // RT shadow

    // ── Descriptor Sets（RHI）──
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;        // Resolve
    std::unique_ptr<rhi::RHIDescriptorSet> m_smSet;      // SM
    std::unique_ptr<rhi::RHIDescriptorSet> m_frameSet;   // Frame
    std::unique_ptr<rhi::RHIDescriptorSet> m_vsmBlurSet; // VSM blur
    std::unique_ptr<rhi::RHIDescriptorSet> m_rtSet;      // RT

    // ── Pipelines（RHI）──
    std::unique_ptr<rhi::RHIPipelineState> m_smPipeline;      // Hard SM (graphics)
    std::unique_ptr<rhi::RHIPipelineState> m_vsmGenPipeline;  // VSM gen (graphics)
    std::unique_ptr<rhi::RHIPipelineState> m_vsmBlurPipeline; // VSM blur (compute)
    std::unique_ptr<rhi::RHIPipelineState> m_resolveHard;     // Hard resolve
    std::unique_ptr<rhi::RHIPipelineState> m_resolvePCF;      // PCF resolve
    std::unique_ptr<rhi::RHIPipelineState> m_resolveVSM;      // VSM resolve
    std::unique_ptr<rhi::RHIPipelineState> m_resolvePCSS;     // PCSS resolve
    std::unique_ptr<rhi::RHIPipelineState> m_rtHardPipeline;  // RT hard
    std::unique_ptr<rhi::RHIPipelineState> m_rtSoftPipeline;  // RT soft

    // ── RT（保留 VK，未迁移到 RHIAccelerationStructure）──
    VkAccelerationStructureKHR m_tlas = VK_NULL_HANDLE;

    // ── RT 可调参数
    float m_rtSunRadius = 0.03f;
    int   m_rtRayCount  = 8;

    // ── UBO + index buffer ──
    Buffer m_shadowUbo;
    VkBuffer m_indexBuffer = VK_NULL_HANDLE;
    glm::vec3 m_sunDir{0.0f, -1.0f, 0.0f};
    glm::vec3 m_sceneAabbMin{0};
    glm::vec3 m_sceneAabbMax{0};
    bool m_fgAutoBarrier = false;
};

} // namespace somegi
