#pragma once
#include "core/vk_common.h"
#include "render_targets.h"
#include "vxgi_resources.h"
#include "sdfgi_resources.h"
#include <glm/glm.hpp>
#include <vector>

// SdfgiPass —— C.3：SDFGI-lite。
//
// 四个 sub-pass：
//   1. seed     —— voxelGrid.alpha → seedA（cellPos+isSeed flag）
//   2. jfa      —— k=64,32,...,1 共 7 步 jump-flooding；ping-pong seedA/seedB
//   3. finalize —— 收敛后的 seed → udf（R16F UDF in cell units）
//   4. trace    —— GBuffer normal/depth → N rays sphere-step → sample
//                  voxelGrid → 写 rt.ssgi（与 SSGI/GTGI 共享）
//
// 每帧 record() 一次性跑完 4 步。SDF 重建很慢（7 × 128³ JFA）但只在 SDFGI
// 模式开启时跑；用户可后续加 "Re-bake SDF" 控制按帧/按需。
//
// 共享：seedA/seedB/udf 由 SdfgiResources 提供；voxelGrid + 各向异性 alpha
// 由 VxgiResources 提供（要求 voxelize+inject+mipmap+aniso 链已经跑完，
// voxelGrid 全 mip SHADER_READ_ONLY）。
//
// 输出：rt.ssgi（RGBA16F，layout GENERAL 写）。
//
// 注意 layout 约定：
//   - seedA / seedB / udf：进入 record 时由调用方保证 GENERAL（首次 init
//     用一次性 transition；之后 pass 自己维护 GENERAL/SR_O 间切换）。
//   - rt.ssgi：进入时 GENERAL（外部 SSGI 链路约定）。
//   - voxelGrid 全 mip + aniso 全 mip：进入时 SHADER_READ_ONLY。

namespace somegi {
class Device;

class SdfgiPass {
public:
    void init(Device& d);
    void destroy();

    // 资源 binding 在 SDFGI 启用 / resize / scene 切换时调用。
    void bindResources(Device& d,
                       const SdfgiResources& sdfgi,
                       const VxgiResources& vxgi,
                       const RenderTargets& rt,
                       VkBuffer frameUbo);

    // 每帧 record：跑完 seed → JFA loop → finalize → trace 全链。
    // 调用方负责把 seedA/seedB/udf 在首次使用前 transition 到 GENERAL；
    // pass 内部走 GENERAL ↔ SHADER_READ_ONLY 转换。
    void record(VkCommandBuffer cmd,
                const SdfgiResources& sdfgi,
                const RenderTargets& rt,
                uint32_t frameIndex,
                float    seedThreshold,
                float    maxDistCells,
                uint32_t numRays,
                uint32_t maxSteps,
                float    rayMaxCells,
                float    hitEpsCells);

    bool enabled = false;
    // ImGui 可调参数（默认值）
    float seedThreshold  = 0.05f;
    float maxDistCells   = 240.0f;   // 写 udf 的上限（128³ 对角线 ~221）
    int   numRays        = 4;
    int   maxSteps       = 48;
    float rayMaxCells    = 96.0f;
    float hitEpsCells    = 0.6f;

private:
    void initSeedPipeline();
    void initJfaPipeline();
    void initFinalizePipeline();
    void initTracePipeline();

    Device* m_device = nullptr;

    // 四个 set layout / pipeline / pool。set 各自独立 alloc。
    VkSampler m_linearClamp = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;

    // seed
    VkDescriptorSetLayout m_seedSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      m_seedPlLayout  = VK_NULL_HANDLE;
    VkPipeline            m_seedPipeline  = VK_NULL_HANDLE;
    VkDescriptorSet       m_seedSet       = VK_NULL_HANDLE;

    // jfa（两组 set：A→B 和 B→A）
    VkDescriptorSetLayout m_jfaSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      m_jfaPlLayout  = VK_NULL_HANDLE;
    VkPipeline            m_jfaPipeline  = VK_NULL_HANDLE;
    VkDescriptorSet       m_jfaSetAB     = VK_NULL_HANDLE;   // src=A, dst=B
    VkDescriptorSet       m_jfaSetBA     = VK_NULL_HANDLE;   // src=B, dst=A

    // finalize（src 由 JFA 落点决定 —— 偶数步后 src=B；动态选 set）
    VkDescriptorSetLayout m_finSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      m_finPlLayout  = VK_NULL_HANDLE;
    VkPipeline            m_finPipeline  = VK_NULL_HANDLE;
    VkDescriptorSet       m_finSetA      = VK_NULL_HANDLE;
    VkDescriptorSet       m_finSetB      = VK_NULL_HANDLE;

    // trace
    VkDescriptorSetLayout m_traceSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      m_tracePlLayout  = VK_NULL_HANDLE;
    VkPipeline            m_tracePipeline  = VK_NULL_HANDLE;
    VkDescriptorSet       m_traceSet       = VK_NULL_HANDLE;
};

}
