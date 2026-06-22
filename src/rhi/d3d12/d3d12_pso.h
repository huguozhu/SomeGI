// rhi/d3d12/d3d12_pso.h — D3D12 管线状态对象和描述符集
#pragma once
#include "../base/pipeline_state.h"
#include "d3d12_descriptor.h"
#include "d3d12_shader.h"
#include <d3d12.h>
#include <vector>
#include <memory>
#include <unordered_map>

namespace somegi {
namespace rhi {

class D3D12RHIDevice;

// ════════════════════════════════════════════════════════════════
// D3D12RHIPipelineState — ID3D12PipelineState + ID3D12RootSignature
// ════════════════════════════════════════════════════════════════
class D3D12RHIPipelineState : public RHIPipelineState {
public:
    // Graphics PSO
    D3D12RHIPipelineState(D3D12RHIDevice& device, const GraphicsPSODesc& desc);
    // Compute PSO
    D3D12RHIPipelineState(D3D12RHIDevice& device, const ComputePSODesc& desc);
    ~D3D12RHIPipelineState() override;

    void* nativeHandle() const override { return (void*)m_pipeline; }

    ID3D12PipelineState* pipeline() const { return m_pipeline; }
    ID3D12RootSignature* rootSignature() const { return m_rootSig; }
    bool isCompute() const { return m_isCompute; }
    D3D12_PRIMITIVE_TOPOLOGY topology() const { return m_topology; }

    // 获取 Vulkan 描述符集 → D3D12 根参数索引映射
    uint32_t getResourceParamForSet(uint32_t setIdx) const {
        return setIdx < m_setParamMap.size() ? m_setParamMap[setIdx].first : ~0u;
    }
    uint32_t getSamplerParamForSet(uint32_t setIdx) const {
        return setIdx < m_setParamMap.size() ? m_setParamMap[setIdx].second : ~0u;
    }
    void addSetParamMapping(uint32_t resIdx, uint32_t smpIdx) {
        m_setParamMap.push_back({resIdx, smpIdx});
    }

private:
    void createRootSignature(const std::vector<RHIDescriptorSetLayout*>& setLayouts,
                             const std::vector<PushConstantRange>& pushConstants);
    void createGraphicsPSO(const GraphicsPSODesc& desc);
    void createComputePSO(const ComputePSODesc& desc);

    D3D12RHIDevice& m_device;
    ID3D12PipelineState* m_pipeline = nullptr;
    ID3D12RootSignature* m_rootSig = nullptr;
    bool m_isCompute = false;
    D3D12_PRIMITIVE_TOPOLOGY m_topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    // {resourceParamIdx, samplerParamIdx} per descriptor set
    std::vector<std::pair<uint32_t, uint32_t>> m_setParamMap;
};

} // namespace rhi
} // namespace somegi
