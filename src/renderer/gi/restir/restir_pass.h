// RestirPass — ReSTIR DI (3管线+可选RT)，RHI管理资源，VkCompat bind/record。
#pragma once
#include "renderer/core/render_targets.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include "renderer/gi/restir/restir_resources.h"
#include "rhi/base/sampler.h"
#include "rhi/base/acceleration_structure.h"
#include <memory>
#include <vulkan/vulkan.h>
namespace somegi {
namespace rhi { class RHIDevice; class RHIDescriptorSetLayout; class RHIPipelineState; class RHIDescriptorSet; class RHICommandBuffer; }
class RestirPass {
public:
    ~RestirPass(); void init(rhi::RHIDevice& d,bool hwRt=false); void destroy();
    void bindResources(const RestirResources& res,const VxgiResources& vxgi,const RenderTargets& rt,VkBuffer frameUbo);
    void bindResourcesRt(const RestirResources& res,const RenderTargets& rt,VkBuffer frameUbo,const rhi::RHIAccelerationStructure& tlas);
    void record(rhi::RHICommandBuffer& cmd,const RestirResources& res,const RenderTargets& rt,uint32_t nl,uint32_t nc,uint32_t nn,float sr,uint32_t ss,float is,uint32_t fi,bool useRt=false);
    bool enabled=false; int numCandidates=8,numNeighbors=4,shadowSteps=6; float spatialRadius=24.f,intensityScale=1.f;
private:
    rhi::RHIDevice* m_rhiDevice=nullptr; std::unique_ptr<rhi::RHISampler> m_linearClamp; bool m_rtShadeReady=false;
    std::unique_ptr<rhi::RHIDescriptorSetLayout> m_initDsl,m_spatialDsl,m_shadeDsl,m_shadeRtDsl;
    std::unique_ptr<rhi::RHIPipelineState> m_initPipe,m_spatialPipe,m_shadePipe,m_shadeRtPipe;
    std::unique_ptr<rhi::RHIDescriptorSet> m_initSet,m_spatialSet,m_shadeSet,m_shadeRtSet;
}; } // namespace somegi
