#pragma once
#include "core/buffer.h"
#include "rhi/base/buffer.h"
#include <memory>

// NDGI (Neural Dynamic Global Illumination) 资源
//
// 微型 MLP（2 隐藏层 × 16）替代 DDGI 的 SH probe atlas。
// 权重以 float4 打包存储，兼容通用 compute shader 无 Tensor Core 推理。
//
// 架构: 6(输入) → 16 → 16 → 3(输出)
//   输入: worldPos(3) + normal(3)
//   输出: RGB irradiance
//   激活: leaky ReLU (α=0.01)
//   总参数: 6×16 + 16 + 16×16 + 16 + 16×3 + 3 = 451 floats ≈ 1.8KB
//
// 训练: per-frame online SGD，从 probe ray trace 收集样本

namespace somegi {
class Device;
namespace rhi { class RHIDevice; }

class NdgiResources {
public:
    // 探针网格参数（复用 DDGI 布局）
    static constexpr uint32_t kProbesX = 8;
    static constexpr uint32_t kProbesY = 4;
    static constexpr uint32_t kProbesZ = 8;
    static constexpr uint32_t kProbeCount = kProbesX * kProbesY * kProbesZ;
    static constexpr uint32_t kRaysPerProbe = 32;  // 每帧每探针光线数

    // MLP 架构
    static constexpr uint32_t kInputDim = 6;   // pos(3) + normal(3)
    static constexpr uint32_t kHiddenDim = 16;  // 隐藏层神经元
    static constexpr uint32_t kOutputDim = 3;   // RGB irradiance

    // 权重打包: 每 4 个 float 打包为 1 个 float4
    // Layer 1: 6×16 = 96 weights → 24 float4; bias: 16 → 4 float4
    // Layer 2: 16×16 = 256 weights → 64 float4; bias: 16 → 4 float4
    // Layer 3: 16×3 = 48 weights → 12 float4; bias: 3 → 1 float4
    static constexpr uint32_t kW1Floats  = kHiddenDim * kInputDim;    // 96
    static constexpr uint32_t kB1Floats  = kHiddenDim;                // 16
    static constexpr uint32_t kW2Floats  = kHiddenDim * kHiddenDim;   // 256
    static constexpr uint32_t kB2Floats  = kHiddenDim;                // 16
    static constexpr uint32_t kW3Floats  = kOutputDim * kHiddenDim;   // 48
    static constexpr uint32_t kB3Floats  = kOutputDim;                // 3

    // float4-packed sizes
    static constexpr uint32_t kW1Vec4 = kW1Floats / 4;   // 24
    static constexpr uint32_t kB1Vec4 = kB1Floats / 4;   // 4
    static constexpr uint32_t kW2Vec4 = kW2Floats / 4;   // 64
    static constexpr uint32_t kB2Vec4 = kB2Floats / 4;   // 4
    static constexpr uint32_t kW3Vec4 = kW3Floats / 4;   // 12
    static constexpr uint32_t kB3Vec4 = 1;               // 1 (3 floats padded to float4)

    // 样本 buffer: 每个样本 (pos.xyz, normal.xyz, radiance.rgb) = 9 floats
    static constexpr uint32_t kSampleFloats = kInputDim + kOutputDim;    // 9
    static constexpr uint32_t kMaxSamples = kProbeCount * kRaysPerProbe; // 8192
    static constexpr uint32_t kSampleBufferFloats = kMaxSamples * kSampleFloats; // 73728

    void create(Device& d, rhi::RHIDevice& rhiD);
    void destroy();

    // 权重 buffer（训练更新、推理读取）
    const Buffer& weights1() const { return m_w1; }
    const Buffer& bias1()    const { return m_b1; }
    const Buffer& weights2() const { return m_w2; }
    const Buffer& bias2()    const { return m_b2; }
    const Buffer& weights3() const { return m_w3; }
    const Buffer& bias3()    const { return m_b3; }

    // 样本 buffer（probe trace 写入、训练读取）
    const Buffer& sampleBuf()    const { return m_sampleBuf; }
    const Buffer& sampleCount()  const { return m_sampleCount; }

    rhi::RHIBuffer* rhiWeights1() const { return m_rhiW1.get(); }
    rhi::RHIBuffer* rhiBias1()    const { return m_rhiB1.get(); }
    rhi::RHIBuffer* rhiWeights2() const { return m_rhiW2.get(); }
    rhi::RHIBuffer* rhiBias2()    const { return m_rhiB2.get(); }
    rhi::RHIBuffer* rhiWeights3() const { return m_rhiW3.get(); }
    rhi::RHIBuffer* rhiBias3()    const { return m_rhiB3.get(); }
    rhi::RHIBuffer* rhiSampleBuf()   const { return m_rhiSampleBuf.get(); }
    rhi::RHIBuffer* rhiSampleCount() const { return m_rhiSampleCount.get(); }

private:
    rhi::RHIDevice* m_rhiDevice = nullptr;
    Buffer m_w1, m_b1, m_w2, m_b2, m_w3, m_b3;
    Buffer m_sampleBuf;
    Buffer m_sampleCount;
    std::unique_ptr<rhi::RHIBuffer> m_rhiW1, m_rhiB1, m_rhiW2, m_rhiB2, m_rhiW3, m_rhiB3;
    std::unique_ptr<rhi::RHIBuffer> m_rhiSampleBuf;
    std::unique_ptr<rhi::RHIBuffer> m_rhiSampleCount;
};

} // namespace somegi
