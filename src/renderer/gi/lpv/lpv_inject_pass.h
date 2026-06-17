// LpvInjectPass — RSM VPL 注入 LPV (Compute)，已迁移到 RHI。
#pragma once
#include "core/image.h"
#include "renderer/gi/lpv/lpv_grid.h"
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {
class Device;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }

class LpvInjectPass {
public:
    ~LpvInjectPass();
    void init(rhi::RHIDevice& d, uint32_t rsmSize);
    void destroy();
    void bindResources(const Image& rsmPos, const Image& rsmN, const Image& rsmFlux,
                       const LpvGrid& grid, const Image& gv);
    void record(rhi::RHICommandBuffer& cmd, uint32_t gridRes,
                const glm::vec3& gridMin, float cellSize);
    void record(VkCommandBuffer cmd, uint32_t gridRes,
                const glm::vec3& gridMin, float cellSize);
    uint32_t rsmSize() const { return m_rsmSize; }

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_setLayout;
    std::unique_ptr<rhi::RHIPipelineState> m_pipeline;
    std::unique_ptr<rhi::RHIDescriptorSet> m_set;
    uint32_t m_rsmSize=0;
    // 缓存 grid image 用于 record 内的 clearColor
    const Image* m_lpvR=nullptr, *m_lpvG=nullptr, *m_lpvB=nullptr, *m_gv=nullptr;
};

} // namespace somegi
