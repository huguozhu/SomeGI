// rhi/d3d12/d3d12_pso.h — D3D12 管线状态对象和描述符集
#pragma once
#include "../base/pipeline_state.h"
#include "../base/descriptor.h"
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

    ID3D12PipelineState* pipeline() { return m_pipeline; }
    ID3D12RootSignature* rootSignature() { return m_rootSig; }
    bool isCompute() const { return m_isCompute; }

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
};

// ════════════════════════════════════════════════════════════════
// D3D12RHIDescriptorSetLayout — 稀疏：仅记录绑定信息
// ════════════════════════════════════════════════════════════════
class D3D12RHIDescriptorSetLayout : public RHIDescriptorSetLayout {
public:
    D3D12RHIDescriptorSetLayout(const DescSetLayoutDesc& desc);
    void* nativeHandle() const override { return nullptr; }
    const std::vector<DescriptorBinding>& bindings() const { return m_bindings; }
private:
    std::vector<DescriptorBinding> m_bindings;
};

// ════════════════════════════════════════════════════════════════
// D3D12RHIDescriptorSet — GPU 可见描述符堆 ring-buffer 分配
// ════════════════════════════════════════════════════════════════
class D3D12RHIDescriptorSet : public RHIDescriptorSet {
public:
    D3D12RHIDescriptorSet(D3D12RHIDevice& device, D3D12RHIDescriptorSetLayout& layout);
    ~D3D12RHIDescriptorSet() override;
    void write(const std::vector<DescriptorWrite>& writes) override;
    void* nativeHandle() const override { return (void*)(uintptr_t)m_gpuStart.ptr; }

    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle() const { return m_gpuStart; }
    uint32_t descriptorCount() const { return m_count; }

private:
    D3D12RHIDevice& m_device;
    D3D12_GPU_DESCRIPTOR_HANDLE m_gpuStart{};
    uint32_t m_count = 0;
    bool m_dirty = true;

    // 缓存的写入（在 bind 前批量提交到 GPU 可见堆）
    struct CachedWrite {
        uint32_t binding;
        DescriptorType type;
        D3D12_CPU_DESCRIPTOR_HANDLE srcHandle{};
    };
    std::vector<CachedWrite> m_pendingWrites;
};

} // namespace rhi
} // namespace somegi
