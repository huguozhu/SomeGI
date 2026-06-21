// rhi/d3d12/d3d12_acceleration_structure.h — D3D12 加速结构（光线追踪）
#pragma once
#include "../base/acceleration_structure.h"
#include <d3d12.h>
#include <memory>

namespace somegi {
namespace rhi {

// D3D12 加速度结构包装器。
// D3D12 中 TLAS/BLAS 构建结果存储在 ID3D12Resource 中。
// 支持 owning（管理 resource 生命周期）和 non-owning（包装外部创建的 AS）两种模式。
class D3D12RHIAccelerationStructure : public RHIAccelerationStructure {
public:
    // 非拥有型工厂：包装外部创建的 TLAS/BLAS buffer
    static std::unique_ptr<RHIAccelerationStructure> createNonOwning(ID3D12Resource* as);

    ~D3D12RHIAccelerationStructure() override;
    void* nativeHandle() const override { return m_resource; }

private:
    // 拥有型构造（当前仅 createNonOwning 使用 owns=false）
    D3D12RHIAccelerationStructure(ID3D12Resource* resource, bool owns);

    ID3D12Resource* m_resource = nullptr;
    bool m_owns = false;
};

} // namespace rhi
} // namespace somegi
