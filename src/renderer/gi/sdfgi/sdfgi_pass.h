// SdfgiPass — SDFGI-lite (4管线)，RHI 管理资源，record VkCompat。
#pragma once
#include "rhi/base/sampler.h"
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>
namespace somegi { class SdfgiResources; class VxgiResources; struct RenderTargets;
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; }
class SdfgiPass {
public:
    ~SdfgiPass(); void init(rhi::RHIDevice& d); void destroy();
    void bindResources(const SdfgiResources& sf,const VxgiResources& vx,const RenderTargets& rt,VkBuffer fb);
    void record(VkCommandBuffer cmd,const SdfgiResources& sf,const RenderTargets& rt,uint32_t fi,float st,float md,uint32_t nr,uint32_t ms,float rm,float he);
    bool enabled=false; float seedThreshold=0.05f,maxDistCells=240.f; int numRays=4,maxSteps=48; float rayMaxCells=96.f,hitEpsCells=0.6f;
private:
    rhi::RHIDevice* m_rhiDevice=nullptr;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_seedDsl,m_jfaDsl,m_finDsl,m_traceDsl;
    std::unique_ptr<rhi::RHIPipelineState> m_seedPipe,m_jfaPipe,m_finPipe,m_tracePipe;
    std::unique_ptr<rhi::RHIDescriptorSet> m_seedSet,m_jfaAB,m_jfaBA,m_finA,m_finB,m_traceSet;
    std::unique_ptr<rhi::RHISampler> m_linearClamp;
}; } // namespace somegi
