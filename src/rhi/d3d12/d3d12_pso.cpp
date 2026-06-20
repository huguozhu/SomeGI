// rhi/d3d12/d3d12_pso.cpp — D3D12 PSO + DescriptorSet 实现
#include "d3d12_pso.h"
#include "d3d12_device.h"
#include "d3d12_texture.h"
#include <stdexcept>
#include <cstdio>

namespace somegi {
namespace rhi {

// ════════════════════════════════════════════════════════════════
// 辅助映射
// ════════════════════════════════════════════════════════════════

static D3D12_PRIMITIVE_TOPOLOGY_TYPE toD3D12Topology(PrimitiveTopology t) {
    switch (t) {
        case PrimitiveTopology::TriangleList:
        case PrimitiveTopology::TriangleStrip: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        case PrimitiveTopology::LineList:      return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        case PrimitiveTopology::PointList:     return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
        default: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    }
}

static D3D12_FILL_MODE toD3D12Fill(FillMode f) {
    return (f == FillMode::Wireframe) ? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
}

static D3D12_CULL_MODE toD3D12Cull(CullMode c) {
    switch (c) {
        case CullMode::None:  return D3D12_CULL_MODE_NONE;
        case CullMode::Front: return D3D12_CULL_MODE_FRONT;
        default:              return D3D12_CULL_MODE_BACK;
    }
}

static D3D12_COMPARISON_FUNC toD3D12Cmp(CompareFunc f) {
    switch (f) {
        case CompareFunc::Never:        return D3D12_COMPARISON_FUNC_NEVER;
        case CompareFunc::Less:         return D3D12_COMPARISON_FUNC_LESS;
        case CompareFunc::Equal:        return D3D12_COMPARISON_FUNC_EQUAL;
        case CompareFunc::LessEqual:    return D3D12_COMPARISON_FUNC_LESS_EQUAL;
        case CompareFunc::Greater:      return D3D12_COMPARISON_FUNC_GREATER;
        case CompareFunc::NotEqual:     return D3D12_COMPARISON_FUNC_NOT_EQUAL;
        case CompareFunc::GreaterEqual: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
        case CompareFunc::Always:       return D3D12_COMPARISON_FUNC_ALWAYS;
        default: return D3D12_COMPARISON_FUNC_LESS;
    }
}

static DXGI_FORMAT toD3D12VertexFormat(VertexFormat f) {
    switch (f) {
        case VertexFormat::Float:  return DXGI_FORMAT_R32_FLOAT;
        case VertexFormat::Float2: return DXGI_FORMAT_R32G32_FLOAT;
        case VertexFormat::Float3: return DXGI_FORMAT_R32G32B32_FLOAT;
        case VertexFormat::Float4: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case VertexFormat::Uint:   return DXGI_FORMAT_R32_UINT;
        case VertexFormat::Uint2:  return DXGI_FORMAT_R32G32_UINT;
        case VertexFormat::Uint3:  return DXGI_FORMAT_R32G32B32_UINT;
        case VertexFormat::Uint4:  return DXGI_FORMAT_R32G32B32A32_UINT;
        default: return DXGI_FORMAT_R32G32B32_FLOAT;
    }
}

static D3D12_DESCRIPTOR_RANGE_TYPE toD3D12RangeType(DescriptorType t) {
    switch (t) {
        case DescriptorType::SampledImage:          return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        case DescriptorType::StorageImage:          return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        case DescriptorType::UniformBuffer:         return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        case DescriptorType::StorageBuffer:         return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        case DescriptorType::Sampler:               return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        case DescriptorType::AccelerationStructure: return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        default: return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    }
}

// ════════════════════════════════════════════════════════════════
// D3D12RHIPipelineState
// ════════════════════════════════════════════════════════════════

D3D12RHIPipelineState::D3D12RHIPipelineState(D3D12RHIDevice& device,
                                               const GraphicsPSODesc& desc)
    : m_device(device), m_isCompute(false) {
    // 收集 descriptor set layouts
    std::vector<RHIDescriptorSetLayout*> layouts;
    for (auto* l : desc.descriptorSetLayouts) {
        if (l) layouts.push_back(const_cast<RHIDescriptorSetLayout*>(l));
    }
    createRootSignature(layouts, desc.pushConstants);
    createGraphicsPSO(desc);
}

D3D12RHIPipelineState::D3D12RHIPipelineState(D3D12RHIDevice& device,
                                               const ComputePSODesc& desc)
    : m_device(device), m_isCompute(true) {
    std::vector<RHIDescriptorSetLayout*> layouts;
    for (auto* l : desc.descriptorSetLayouts) {
        if (l) layouts.push_back(const_cast<RHIDescriptorSetLayout*>(l));
    }
    createRootSignature(layouts, desc.pushConstants);
    createComputePSO(desc);
}

D3D12RHIPipelineState::~D3D12RHIPipelineState() {
    if (m_pipeline) m_pipeline->Release();
    if (m_rootSig)  m_rootSig->Release();
}

void D3D12RHIPipelineState::createRootSignature(
    const std::vector<RHIDescriptorSetLayout*>& setLayouts,
    const std::vector<PushConstantRange>& pushConstants) {

    std::vector<D3D12_ROOT_PARAMETER1> params;
    std::vector<std::vector<D3D12_DESCRIPTOR_RANGE1>> rangesPerSet(setLayouts.size());

    // 每个 descriptor set → 一个 descriptor table
    for (size_t s = 0; s < setLayouts.size(); ++s) {
        auto* d3dLayout = static_cast<D3D12RHIDescriptorSetLayout*>(setLayouts[s]);
        auto& ranges = rangesPerSet[s];

        for (auto& b : d3dLayout->bindings()) {
            D3D12_DESCRIPTOR_RANGE1 r{};
            r.RangeType = toD3D12RangeType(b.type);
            r.NumDescriptors = b.count;
            r.BaseShaderRegister = b.binding;
            r.RegisterSpace = (UINT)s; // descriptor set index → register space
            r.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
            r.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            ranges.push_back(r);
        }

        if (!ranges.empty()) {
            D3D12_ROOT_PARAMETER1 p{};
            p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            p.DescriptorTable.NumDescriptorRanges = (UINT)ranges.size();
            p.DescriptorTable.pDescriptorRanges = ranges.data();
            p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(p);
        }
    }

    // push constants → root constants
    for (auto& pc : pushConstants) {
        D3D12_ROOT_PARAMETER1 p{};
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        p.Constants.ShaderRegister = 0;
        p.Constants.RegisterSpace = 1000; // 独立 register space
        p.Constants.Num32BitValues = pc.size / 4;
        p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rsd{};
    rsd.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rsd.Desc_1_1.NumParameters = (UINT)params.size();
    rsd.Desc_1_1.pParameters = params.empty() ? nullptr : params.data();
    rsd.Desc_1_1.NumStaticSamplers = 0;
    rsd.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    ID3DBlob* signature = nullptr;
    ID3DBlob* error = nullptr;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&rsd, &signature, &error);
    if (FAILED(hr)) {
        std::string errMsg = error ? (const char*)error->GetBufferPointer() : "unknown";
        if (error) error->Release();
        throw std::runtime_error("[d3d12] SerializeRootSignature failed: " + errMsg);
    }
    hr = m_device.device()->CreateRootSignature(0,
        signature->GetBufferPointer(), signature->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSig));
    signature->Release();
    if (FAILED(hr))
        throw std::runtime_error("[d3d12] CreateRootSignature failed");
}

void D3D12RHIPipelineState::createGraphicsPSO(const GraphicsPSODesc& desc) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psd{};
    psd.pRootSignature = m_rootSig;

    // Shaders
    if (desc.vertexShader) {
        auto* vShader = static_cast<const D3D12RHIShader*>(desc.vertexShader);
        psd.VS = { vShader->bytecodeData(), vShader->bytecodeSize() };
    }
    if (desc.fragmentShader) {
        auto* pShader = static_cast<const D3D12RHIShader*>(desc.fragmentShader);
        psd.PS = { pShader->bytecodeData(), pShader->bytecodeSize() };
    }

    // Input layout
    std::vector<D3D12_INPUT_ELEMENT_DESC> inputElements;
    for (auto& attr : desc.vertexInput.attributes) {
        D3D12_INPUT_ELEMENT_DESC ie{};
        ie.SemanticName = "TEXCOORD";
        ie.SemanticIndex = attr.location;
        ie.Format = toD3D12VertexFormat(attr.format);
        ie.InputSlot = attr.binding;
        ie.AlignedByteOffset = attr.offset;
        if (attr.binding < desc.vertexInput.bindings.size()) {
            ie.InputSlotClass = desc.vertexInput.bindings[attr.binding].perInstance
                ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
        }
        inputElements.push_back(ie);
    }
    psd.InputLayout = { inputElements.data(), (UINT)inputElements.size() };

    // 其他状态
    psd.PrimitiveTopologyType = toD3D12Topology(desc.topology);
    psd.RasterizerState.FillMode = toD3D12Fill(desc.rasterization.fill);
    psd.RasterizerState.CullMode = toD3D12Cull(desc.rasterization.cull);
    psd.RasterizerState.FrontCounterClockwise = desc.rasterization.frontCCW ? TRUE : FALSE;
    psd.RasterizerState.DepthBias = desc.rasterization.depthBiasEnable
        ? (INT)desc.rasterization.depthBiasConstantFactor : 0;
    psd.RasterizerState.DepthBiasClamp = desc.rasterization.depthBiasClamp;
    psd.RasterizerState.SlopeScaledDepthBias = desc.rasterization.depthBiasSlopeFactor;

    psd.DepthStencilState.DepthEnable = desc.depthStencil.depthTest ? TRUE : FALSE;
    psd.DepthStencilState.DepthWriteMask = desc.depthStencil.depthWrite
        ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    psd.DepthStencilState.DepthFunc = toD3D12Cmp(desc.depthStencil.depthCompare);
    psd.DSVFormat = DXGI_FORMAT_D32_FLOAT; // 暂时硬编码

    psd.BlendState.AlphaToCoverageEnable = FALSE;
    psd.BlendState.IndependentBlendEnable = FALSE;
    for (size_t i = 0; i < desc.blend.attachments.size() && i < 8; ++i) {
        auto& ba = desc.blend.attachments[i];
        psd.BlendState.RenderTarget[i].BlendEnable = ba.blendEnable ? TRUE : FALSE;
        psd.BlendState.RenderTarget[i].SrcBlend = D3D12_BLEND_ONE;
        psd.BlendState.RenderTarget[i].DestBlend = D3D12_BLEND_ZERO;
        psd.BlendState.RenderTarget[i].BlendOp = D3D12_BLEND_OP_ADD;
        psd.BlendState.RenderTarget[i].SrcBlendAlpha = D3D12_BLEND_ONE;
        psd.BlendState.RenderTarget[i].DestBlendAlpha = D3D12_BLEND_ZERO;
        psd.BlendState.RenderTarget[i].BlendOpAlpha = D3D12_BLEND_OP_ADD;
        psd.BlendState.RenderTarget[i].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    }

    psd.NumRenderTargets = (UINT)desc.renderTargets.colorFormats.size();
    for (size_t i = 0; i < desc.renderTargets.colorFormats.size() && i < 8; ++i) {
        // 忽略深度格式
    }
    psd.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT; // 暂时硬编码
    if (psd.NumRenderTargets > 1) psd.RTVFormats[1] = DXGI_FORMAT_R8G8B8A8_UNORM;

    psd.SampleDesc = { desc.renderTargets.sampleCount, 0 };
    psd.SampleMask = UINT_MAX;

    m_topology = (desc.topology == PrimitiveTopology::TriangleStrip)
        ? D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP
        : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

    if (FAILED(m_device.device()->CreateGraphicsPipelineState(&psd,
            IID_PPV_ARGS(&m_pipeline))))
        throw std::runtime_error("[d3d12] CreateGraphicsPipelineState failed");
}

void D3D12RHIPipelineState::createComputePSO(const ComputePSODesc& desc) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd{};
    psd.pRootSignature = m_rootSig;
    if (desc.computeShader) {
        auto* d3dShader = static_cast<const D3D12RHIShader*>(desc.computeShader);
        psd.CS = { d3dShader->bytecodeData(), d3dShader->bytecodeSize() };
    }
    if (FAILED(m_device.device()->CreateComputePipelineState(&psd,
            IID_PPV_ARGS(&m_pipeline))))
        throw std::runtime_error("[d3d12] CreateComputePipelineState failed");
}

// ════════════════════════════════════════════════════════════════
// D3D12RHIDescriptorSetLayout
// ════════════════════════════════════════════════════════════════

D3D12RHIDescriptorSetLayout::D3D12RHIDescriptorSetLayout(const DescSetLayoutDesc& desc)
    : m_bindings(desc.bindings) {}

// ════════════════════════════════════════════════════════════════
// D3D12RHIDescriptorSet
// ════════════════════════════════════════════════════════════════

D3D12RHIDescriptorSet::D3D12RHIDescriptorSet(D3D12RHIDevice& device,
                                               D3D12RHIDescriptorSetLayout& layout)
    : m_device(device) {
    for (auto& b : layout.bindings()) {
        m_count += b.count;
    }
    // 从 GPU 可见堆分配空间
    if (m_count > 0) {
        auto alloc = device.allocDescriptors(m_count);
        m_gpuStart = alloc.gpu;
    }
}

D3D12RHIDescriptorSet::~D3D12RHIDescriptorSet() = default;

void D3D12RHIDescriptorSet::write(const std::vector<DescriptorWrite>& writes) {
    // 向 GPU 可见堆拷贝描述符
    // 简化：仅处理 textureView 类型（SRV）
    for (auto& w : writes) {
        if (w.textureView) {
            auto* view = static_cast<const D3D12RHITextureView*>(w.textureView);
            D3D12_CPU_DESCRIPTOR_HANDLE src = view->srvCpuHandle();
            // 计算目标位置
            D3D12_CPU_DESCRIPTOR_HANDLE dst = m_device.gpuDescriptorHeap()->GetCPUDescriptorHandleForHeapStart();
            dst.ptr += static_cast<SIZE_T>(m_gpuStart.ptr -
                m_device.gpuDescriptorHeap()->GetGPUDescriptorHandleForHeapStart().ptr);
            // 偏移到 binding 对应的槽位
            // Phase 5 完整实现 binding→槽位映射
            m_device.device()->CopyDescriptorsSimple(1, dst, src,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
    }
}

} // namespace rhi
} // namespace somegi
