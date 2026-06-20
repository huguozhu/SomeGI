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

    // 每个 descriptor set → 两个 descriptor table（资源 + 采样器分离）
    for (size_t s = 0; s < setLayouts.size(); ++s) {
        auto* d3dLayout = static_cast<D3D12RHIDescriptorSetLayout*>(setLayouts[s]);
        std::vector<D3D12_DESCRIPTOR_RANGE1> resRanges, smpRanges;

        for (auto& b : d3dLayout->bindings()) {
            D3D12_DESCRIPTOR_RANGE1 r{};
            r.RangeType = toD3D12RangeType(b.type);
            r.NumDescriptors = b.count;
            r.BaseShaderRegister = (b.hlslRegister != ~0u ? b.hlslRegister : b.binding);
            r.RegisterSpace = (UINT)s;
            r.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
            r.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

            // D3D12 要求采样器必须在独立的 descriptor table 中
            if (r.RangeType == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER)
                smpRanges.push_back(r);
            else
                resRanges.push_back(r);
        }

        uint32_t resIdx = ~0u, smpIdx = ~0u;
        // 资源 table (SRV/CBV/UAV)
        if (!resRanges.empty()) {
            rangesPerSet.push_back(resRanges);
            D3D12_ROOT_PARAMETER1 p{};
            p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            p.DescriptorTable.NumDescriptorRanges = (UINT)resRanges.size();
            p.DescriptorTable.pDescriptorRanges = rangesPerSet.back().data();
            p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            resIdx = (uint32_t)params.size();
            d3dLayout->setResourceParam(resIdx);
            params.push_back(p);
        }
        // 采样器 table
        if (!smpRanges.empty()) {
            rangesPerSet.push_back(smpRanges);
            D3D12_ROOT_PARAMETER1 p{};
            p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            p.DescriptorTable.NumDescriptorRanges = (UINT)smpRanges.size();
            p.DescriptorTable.pDescriptorRanges = rangesPerSet.back().data();
            p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            smpIdx = (uint32_t)params.size();
            d3dLayout->setSamplerParam(smpIdx);
            params.push_back(p);
        }
        m_setParamMap.push_back({resIdx, smpIdx});
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

    HRESULT hr = m_device.device()->CreateGraphicsPipelineState(&psd,
        IID_PPV_ARGS(&m_pipeline));
    if (FAILED(hr)) {
        ID3D12InfoQueue* infoQueue = nullptr;
        if (SUCCEEDED(m_device.device()->QueryInterface(IID_PPV_ARGS(&infoQueue)))) {
            UINT64 msgCount = infoQueue->GetNumStoredMessages();
            for (UINT64 i = 0; i < msgCount && i < 3; ++i) {
                SIZE_T msgLen = 0;
                infoQueue->GetMessage(i, nullptr, &msgLen);
                if (msgLen > 0) {
                    auto* msg = (D3D12_MESSAGE*)alloca(msgLen);
                    infoQueue->GetMessage(i, msg, &msgLen);
                    std::fprintf(stderr, "[d3d12]   %s\n", msg->pDescription);
                }
            }
            infoQueue->ClearStoredMessages();
            infoQueue->Release();
        }
        throw std::runtime_error("[d3d12] CreateGraphicsPipelineState failed");
    }
}

void D3D12RHIPipelineState::createComputePSO(const ComputePSODesc& desc) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd{};
    psd.pRootSignature = m_rootSig;
    if (desc.computeShader) {
        auto* d3dShader = static_cast<const D3D12RHIShader*>(desc.computeShader);
        psd.CS = { d3dShader->bytecodeData(), d3dShader->bytecodeSize() };
    }
    HRESULT hr = m_device.device()->CreateComputePipelineState(&psd,
        IID_PPV_ARGS(&m_pipeline));
    if (FAILED(hr)) {
        // 尝试从 info queue 获取详细错误
        ID3D12InfoQueue* infoQueue = nullptr;
        if (SUCCEEDED(m_device.device()->QueryInterface(IID_PPV_ARGS(&infoQueue)))) {
            UINT64 msgCount = infoQueue->GetNumStoredMessages();
            for (UINT64 i = 0; i < msgCount && i < 5; ++i) {
                SIZE_T msgLen = 0;
                infoQueue->GetMessage(i, nullptr, &msgLen);
                if (msgLen > 0) {
                    auto* msg = (D3D12_MESSAGE*)alloca(msgLen);
                    infoQueue->GetMessage(i, msg, &msgLen);
                    std::fprintf(stderr, "[d3d12]   %s\n", msg->pDescription);
                }
            }
            infoQueue->ClearStoredMessages();
            infoQueue->Release();
        }
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "[d3d12] CreateComputePipelineState failed: HRESULT=0x%08X", (unsigned)hr);
        throw std::runtime_error(buf);
    }
}

} // namespace rhi
} // namespace somegi
