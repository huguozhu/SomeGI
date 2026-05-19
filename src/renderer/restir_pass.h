#pragma once
#include "core/vk_common.h"
#include "render_targets.h"
#include "vxgi_resources.h"
#include "restir_resources.h"

// RestirPass —— C.4 ReSTIR DI 软件版编排。
//
// 三个子 pipeline：
//   1. init     —— per-pixel M-候选 RIS → reservoir A
//   2. spatial  —— K 邻居 reservoir 合并 → reservoir B
//   3. shade    —— 读 B 的 picked light + 1 次 voxel 可视性测试 → rt.restir
//
// 共享 set / pool；shade 阶段的 voxel 可视性用 VxgiResources.fullView()。
//
// layout 约定：
//   - 进入：reservoirA / reservoirB 在 GENERAL（首次 init bootstrap），
//           voxelGrid 在 SHADER_READ_ONLY，rt.restir 在 GENERAL
//   - 离开：rt.restir 在 GENERAL（外部转 SHADER_READ_ONLY 给 lighting）

namespace somegi {
class Device;

class RestirPass {
public:
    void init(Device& d, bool hwRtAvailable = false);
    void destroy();

    // 资源 binding：在 init 后调用一次；resize / scene 切换时重调。
    void bindResources(Device& d,
                       const RestirResources& res,
                       const VxgiResources& vxgi,
                       const RenderTargets& rt,
                       VkBuffer frameUbo);

    // M10 RT shade 资源 binding：绑定 TLAS 替代 voxel grid。
    void bindResourcesRt(Device& d,
                         const RestirResources& res,
                         const RenderTargets& rt,
                         VkBuffer frameUbo,
                         VkAccelerationStructureKHR tlas);

    // 一次跑完 init → spatial → shade。调用方负责把 reservoirA/B 在首帧
    // transition 到 GENERAL，rt.restir 转 GENERAL（可写）。
    // useRtVisibility=true 使用 M10 RT shade pipeline（需先调用 bindResourcesRt）。
    void record(VkCommandBuffer cmd,
                const RestirResources& res,
                const RenderTargets& rt,
                uint32_t numLights,
                uint32_t numCandidates,
                uint32_t numNeighbors,
                float    spatialRadiusPx,
                uint32_t shadowSteps,
                float    intensityScale,
                uint32_t frameIndex,
                bool     useRtVisibility = false);

    bool enabled = false;
    int  numCandidates  = 8;
    int  numNeighbors   = 4;
    float spatialRadius = 24.0f;
    int   shadowSteps   = 6;
    float intensityScale = 1.0f;

private:
    void initInitPipeline();
    void initSpatialPipeline();
    void initShadePipeline();
    void initShadeRtPipeline();

    Device* m_device = nullptr;

    VkSampler m_linearClamp = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_initSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      m_initPlLayout  = VK_NULL_HANDLE;
    VkPipeline            m_initPipeline  = VK_NULL_HANDLE;
    VkDescriptorSet       m_initSet       = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_spatialSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      m_spatialPlLayout  = VK_NULL_HANDLE;
    VkPipeline            m_spatialPipeline  = VK_NULL_HANDLE;
    VkDescriptorSet       m_spatialSet       = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_shadeSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      m_shadePlLayout  = VK_NULL_HANDLE;
    VkPipeline            m_shadePipeline  = VK_NULL_HANDLE;
    VkDescriptorSet       m_shadeSet       = VK_NULL_HANDLE;

    // M10 RT shade pipeline（硬件 ray query 可见性）
    VkDescriptorSetLayout m_shadeRtSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      m_shadeRtPlLayout  = VK_NULL_HANDLE;
    VkPipeline            m_shadeRtPipeline  = VK_NULL_HANDLE;
    VkDescriptorSet       m_shadeRtSet       = VK_NULL_HANDLE;
    bool                  m_rtShadeReady     = false;
};

}
