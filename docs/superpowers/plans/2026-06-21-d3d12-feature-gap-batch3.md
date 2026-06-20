# D3D12 功能差距补齐 Batch 3 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐 D3D12 后端的 3 个功能差距：AccelerationStructure 包装、Buffer Barrier 阶段翻译、Sampler 描述符表绑定。

**Architecture:** 三个独立模块，均在 `src/rhi/d3d12/` 目录内修改。按复杂度递增顺序实施：AccelerationStructure（新增 1 个 cpp）→ Buffer Barrier（改 1 个文件）→ Sampler 绑定（改 5 个文件）。

**Tech Stack:** D3D12, C++17

---

### Task 1: AccelerationStructure 包装 — 头文件更新

**Files:**
- Modify: `src/rhi/d3d12/d3d12_acceleration_structure.h`

- [ ] **Step 1: 更新类声明**

将当前的空壳类替换为完整的类声明：

```cpp
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
```

- [ ] **Step 2: 验证编译通过**

```powershell
cmake --build build --target somegi_rhi
```

这个阶段会因缺少 `d3d12_acceleration_structure.cpp` 中的实现而链接失败，这正是下一步要创建的。

---

### Task 2: AccelerationStructure 包装 — 实现文件

**Files:**
- Create: `src/rhi/d3d12/d3d12_acceleration_structure.cpp`
- Modify: `src/rhi/CMakeLists.txt:23-34`

- [ ] **Step 1: 创建实现文件**

```cpp
// rhi/d3d12/d3d12_acceleration_structure.cpp — D3D12 加速结构实现
#include "d3d12_acceleration_structure.h"

namespace somegi {
namespace rhi {

D3D12RHIAccelerationStructure::D3D12RHIAccelerationStructure(ID3D12Resource* resource, bool owns)
    : m_resource(resource), m_owns(owns) {}

D3D12RHIAccelerationStructure::~D3D12RHIAccelerationStructure() {
    if (m_owns && m_resource) {
        m_resource->Release();
    }
}

std::unique_ptr<RHIAccelerationStructure> D3D12RHIAccelerationStructure::createNonOwning(ID3D12Resource* as) {
    return std::unique_ptr<RHIAccelerationStructure>(new D3D12RHIAccelerationStructure(as, false));
}

} // namespace rhi
} // namespace somegi
```

- [ ] **Step 2: 注册到 CMakeLists.txt**

在 `src/rhi/CMakeLists.txt` 的 D3D12 源文件列表末尾添加：

```
        d3d12/d3d12_acceleration_structure.cpp
```

修改位置：`d3d12/d3d12_query_pool.cpp` 之后，`)` 之前。

- [ ] **Step 3: 验证编译通过**

```powershell
cmake --build build --target somegi_rhi
```

Expected: 编译链接成功，无错误。

- [ ] **Step 4: 提交**

```bash
git add src/rhi/d3d12/d3d12_acceleration_structure.h \
        src/rhi/d3d12/d3d12_acceleration_structure.cpp \
        src/rhi/CMakeLists.txt
git commit -m "实现 D3D12 AccelerationStructure 包装

新增 createNonOwning 工厂，与 Vulkan 端接口对齐。
支持 owning/non-owning 两种生命周期模式。

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Buffer Barrier — 实现阶段/访问翻译

**Files:**
- Modify: `src/rhi/d3d12/d3d12_command.cpp:392-400`

- [ ] **Step 1: 添加 BufferAccess → D3D12_RESOURCE_STATES 映射函数**

在 `d3d12_command.cpp` 中，`bufferBarrier` 方法之前添加静态辅助函数：

```cpp
// ════════════════════════════════════════════════════════════════
// BufferAccess → D3D12_RESOURCE_STATES 映射（用于 bufferBarrier）
// ════════════════════════════════════════════════════════════════
static D3D12_RESOURCE_STATES toD3D12BufferState(BufferAccess access) {
    if (access == BufferAccess::None) return D3D12_RESOURCE_STATE_COMMON;

    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;

    if ((uint32_t)access & (uint32_t)BufferAccess::UniformRead)
        state |= D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
               | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if ((uint32_t)access & (uint32_t)BufferAccess::StorageRead)
        state |= D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if ((uint32_t)access & (uint32_t)BufferAccess::StorageWrite)
        state |= D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if ((uint32_t)access & (uint32_t)BufferAccess::IndexRead)
        state |= D3D12_RESOURCE_STATE_INDEX_BUFFER;
    if ((uint32_t)access & (uint32_t)BufferAccess::VertexRead)
        state |= D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    if ((uint32_t)access & (uint32_t)BufferAccess::IndirectRead)
        state |= D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    if ((uint32_t)access & (uint32_t)BufferAccess::TransferRead)
        state |= D3D12_RESOURCE_STATE_COPY_SOURCE;
    if ((uint32_t)access & (uint32_t)BufferAccess::TransferWrite)
        state |= D3D12_RESOURCE_STATE_COPY_DEST;

    return state;
}
```

- [ ] **Step 2: 重写 bufferBarrier() 方法**

替换当前 `d3d12_command.cpp:392-400` 的方法体：

```cpp
void D3D12RHICommandBuffer::bufferBarrier(const RHIBuffer& buf,
                                           PipelineStage /*srcStage*/,
                                           PipelineStage /*dstStage*/,
                                           BufferAccess srcAccess,
                                           BufferAccess dstAccess) {
    auto& d3dBuf = static_cast<const D3D12RHIBuffer&>(buf);

    D3D12_RESOURCE_STATES before = m_device.getResourceState(d3dBuf.resource());
    D3D12_RESOURCE_STATES after  = toD3D12BufferState(dstAccess);

    if (before == after) {
        // UAV → UAV 特殊处理：TRANSITION barrier 不允许 before == after，
        // fallback 为 UAV barrier 确保 UAV 写入对其他 pass 可见
        if (before == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12_RESOURCE_BARRIER rb{};
            rb.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            rb.UAV.pResource = d3dBuf.resource();
            m_cmdList->ResourceBarrier(1, &rb);
        }
        // 其他相同状态：跳过（无需 barrier）
        return;
    }

    m_device.trackResourceState(d3dBuf.resource(), after);

    D3D12_RESOURCE_BARRIER rb{};
    rb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    rb.Transition.pResource   = d3dBuf.resource();
    rb.Transition.StateBefore = before;
    rb.Transition.StateAfter  = after;
    rb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_cmdList->ResourceBarrier(1, &rb);
}
```

- [ ] **Step 3: 验证编译通过**

```powershell
cmake --build build --target somegi_rhi
```

Expected: 编译链接成功，无错误。

- [ ] **Step 4: 提交**

```bash
git add src/rhi/d3d12/d3d12_command.cpp
git commit -m "实现 D3D12 Buffer Barrier 阶段/访问翻译

将 bufferBarrier 从统一 UAV barrier 改为基于状态追踪器的 TRANSITION barrier。
BufferAccess → D3D12_RESOURCE_STATES 映射表支持所有 9 种访问类型。
UAV→UAV 场景 fallback 到 UAV barrier（TRANSITION 不允许 before==after）。

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Sampler 绑定 — Device 端 GPU Sampler 堆

**Files:**
- Modify: `src/rhi/d3d12/d3d12_device.h:86-115`
- Modify: `src/rhi/d3d12/d3d12_device.cpp:148-178`

- [ ] **Step 1: 更新 d3d12_device.h — 添加采样器堆成员和方法**

在 `d3d12_device.h` 中：

**(a) 在 `cpuSamplerHeap()` 之后、`cpuRtvIncrement()` 之前，添加公开方法：**

```cpp
    // GPU 可见采样器描述符堆（与 CBV_SRV_UAV 堆配对绑定）
    ID3D12DescriptorHeap* gpuSamplerHeap()  { return m_gpuSamplerHeap; }
    uint32_t gpuSamplerIncrement() const { return m_gpuSamplerIncrement; }
    DescAlloc allocSamplerDescriptors(uint32_t count);
    void resetSamplerHeap();
```

**(b) 在 `m_cpuSamplerInc` 之后、`m_resourceStates` 之前，添加私有成员：**

```cpp
    // GPU 可见采样器描述符堆
    ID3D12DescriptorHeap* m_gpuSamplerHeap = nullptr;
    uint32_t m_gpuSamplerIncrement = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE m_gpuSamplerStartCPU{};
    D3D12_GPU_DESCRIPTOR_HANDLE m_gpuSamplerStartGPU{};
    uint32_t m_gpuSamplerOffset = 0;
    static constexpr uint32_t kGpuSamplerHeapSize = 2048;
```

- [ ] **Step 2: 更新 d3d12_device.cpp — 创建 GPU Sampler 堆**

在 `createDescriptorHeap()` 末尾（`std::printf` 之后）添加 sampler 堆创建：

```cpp
    // ── 创建 GPU 可见采样器描述符堆 ──
    {
        D3D12_DESCRIPTOR_HEAP_DESC hdSmp{};
        hdSmp.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        hdSmp.NumDescriptors = kGpuSamplerHeapSize;
        hdSmp.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(m_device->CreateDescriptorHeap(&hdSmp, IID_PPV_ARGS(&m_gpuSamplerHeap)))) {
            throw std::runtime_error("[d3d12] CreateDescriptorHeap(SAMPLER, SHADER_VISIBLE) failed");
        }
        m_gpuSamplerIncrement = m_device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        m_gpuSamplerStartCPU = m_gpuSamplerHeap->GetCPUDescriptorHandleForHeapStart();
        m_gpuSamplerStartGPU = m_gpuSamplerHeap->GetGPUDescriptorHandleForHeapStart();
        std::printf("[d3d12] GPU sampler heap: %u slots\n", kGpuSamplerHeapSize);
    }
```

- [ ] **Step 3: 添加 allocSamplerDescriptors / resetSamplerHeap 实现**

在 `resetDescriptorHeap()` 之后添加：

```cpp
D3D12RHIDevice::DescAlloc D3D12RHIDevice::allocSamplerDescriptors(uint32_t count) {
    uint32_t offset = m_gpuSamplerOffset;
    m_gpuSamplerOffset += count;

    DescAlloc a;
    a.offset = offset;
    a.cpu = m_gpuSamplerStartCPU;
    a.cpu.ptr += static_cast<SIZE_T>(offset) * m_gpuSamplerIncrement;
    a.gpu = m_gpuSamplerStartGPU;
    a.gpu.ptr += static_cast<SIZE_T>(offset) * m_gpuSamplerIncrement;
    return a;
}

void D3D12RHIDevice::resetSamplerHeap() {
    m_gpuSamplerOffset = 0;
}
```

- [ ] **Step 4: 在 present() 中调用 resetSamplerHeap()**

修改 `present()` 中 `resetDescriptorHeap()` 调用的位置，在它之后添加：

```cpp
    resetDescriptorHeap(); // 每帧重置描述符堆
    resetSamplerHeap();    // 每帧重置采样器堆
```

- [ ] **Step 5: 在析构函数中释放 GPU sampler 堆**

在 `~D3D12RHIDevice()` 的清理代码中，`if (m_cpuSamplerHeap)` 之后添加：

```cpp
    if (m_gpuSamplerHeap) { m_gpuSamplerHeap->Release(); }
```

- [ ] **Step 6: 在 begin() 中设置双描述符堆**

修改 `d3d12_command.cpp` 的 `begin()` 方法，在 `m_cmdList->Reset()` 之后、`m_recording = true` 之前添加：

```cpp
    // 绑定 GPU 可见描述符堆（CBV_SRV_UAV + Sampler）
    {
        ID3D12DescriptorHeap* heaps[] = {
            m_device.gpuDescriptorHeap(),
            m_device.gpuSamplerHeap()
        };
        m_cmdList->SetDescriptorHeaps(2, heaps);
    }
```

- [ ] **Step 7: 验证编译通过**

```powershell
cmake --build build --target somegi_rhi
```

Expected: 编译链接成功，无错误。

---

### Task 5: Sampler 绑定 — DescriptorSet 端支持

**Files:**
- Modify: `src/rhi/d3d12/d3d12_descriptor.h:29-42`
- Modify: `src/rhi/d3d12/d3d12_descriptor.cpp:10-38`

- [ ] **Step 1: 更新 d3d12_descriptor.h — D3D12RHIDescriptorSetLayout 添加 samplerCount()**

在 `D3D12RHIDescriptorSetLayout` 类的 public 区域添加：

```cpp
    uint32_t samplerCount() const { return m_samplerCount; }
```

在 private 区域 `m_samplerParamIdx` 之后添加：

```cpp
    uint32_t m_samplerCount = 0;
```

- [ ] **Step 2: 更新 d3d12_descriptor.h — D3D12RHIDescriptorSet 添加 sampler GPU 句柄**

在 `D3D12RHIDescriptorSet` 类的 public 区域 `gpuHandle()` 之后添加：

```cpp
    D3D12_GPU_DESCRIPTOR_HANDLE samplerGpuHandle() const { return m_samplerGpuStart; }
```

在 private 区域 `m_count` 之后添加：

```cpp
    D3D12_GPU_DESCRIPTOR_HANDLE m_samplerGpuStart{};
    uint32_t m_samplerCount = 0;
```

- [ ] **Step 3: 更新 d3d12_descriptor.cpp — Layout 构造中统计 sampler 数量**

替换 `D3D12RHIDescriptorSetLayout` 构造函数：

```cpp
D3D12RHIDescriptorSetLayout::D3D12RHIDescriptorSetLayout(const DescSetLayoutDesc& desc)
    : m_bindings(desc.bindings) {
    for (auto& b : desc.bindings) {
        if (b.type == DescriptorType::Sampler)
            m_samplerCount += b.count;
    }
}
```

- [ ] **Step 4: 更新 d3d12_descriptor.cpp — DescriptorSet 构造中分配 sampler 堆空间**

替换 `D3D12RHIDescriptorSet` 构造函数，在已有 resource 分配逻辑后追加 sampler 分配：

```cpp
D3D12RHIDescriptorSet::D3D12RHIDescriptorSet(D3D12RHIDevice& device,
                                               D3D12RHIDescriptorSetLayout& layout)
    : m_device(device) {
    // 统计资源描述符（CBV/SRV/UAV）数量
    uint32_t resCount = 0;
    for (auto& b : layout.bindings()) {
        if (b.type != DescriptorType::Sampler)
            resCount += b.count;
    }
    m_count = resCount;
    if (m_count > 0) {
        auto alloc = device.allocDescriptors(m_count);
        m_gpuStart = alloc.gpu;
    }

    // 分配采样器描述符堆空间
    uint32_t smpCount = layout.samplerCount();
    if (smpCount > 0) {
        auto alloc = device.allocSamplerDescriptors(smpCount);
        m_samplerGpuStart = alloc.gpu;
        m_samplerCount = smpCount;
    }
}
```

- [ ] **Step 5: 添加 sampler 头文件依赖**

在 `d3d12_descriptor.cpp` 的现有 `#include` 块末尾添加：

```cpp
#include "d3d12_sampler.h"
```

- [ ] **Step 6: 更新 d3d12_descriptor.cpp — write() 中处理 sampler 写入**

替换 `D3D12RHIDescriptorSet::write()` 方法，在已有的纹理 view 分支中增加 sampler 分支：

```cpp
void D3D12RHIDescriptorSet::write(const std::vector<DescriptorWrite>& writes) {
    uint32_t smpIndex = 0;
    for (auto& w : writes) {
        if (w.sampler) {
            // 采样器写入：从 CPU sampler 堆拷贝到 GPU 可见 sampler 堆
            auto* smp = static_cast<const D3D12RHISampler*>(w.sampler);
            D3D12_CPU_DESCRIPTOR_HANDLE src = smp->cpuHandle();

            ID3D12DescriptorHeap* gpuSmpHeap = m_device.gpuSamplerHeap();
            D3D12_CPU_DESCRIPTOR_HANDLE dst = gpuSmpHeap->GetCPUDescriptorHandleForHeapStart();
            SIZE_T baseOffset = m_samplerGpuStart.ptr
                - gpuSmpHeap->GetGPUDescriptorHandleForHeapStart().ptr;
            dst.ptr += baseOffset
                       + static_cast<SIZE_T>(smpIndex) * m_device.gpuSamplerIncrement();

            m_device.device()->CopyDescriptorsSimple(1, dst, src,
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
            smpIndex++;
        } else if (w.textureView) {
            auto* view = static_cast<const D3D12RHITextureView*>(w.textureView);
            D3D12_CPU_DESCRIPTOR_HANDLE src = view->srvCpuHandle();
            D3D12_CPU_DESCRIPTOR_HANDLE dst = m_device.gpuDescriptorHeap()->GetCPUDescriptorHandleForHeapStart();
            dst.ptr += m_gpuStart.ptr -
                m_device.gpuDescriptorHeap()->GetGPUDescriptorHandleForHeapStart().ptr;
            m_device.device()->CopyDescriptorsSimple(1, dst, src,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
    }
}
```

- [ ] **Step 7: 验证编译通过**

```powershell
cmake --build build --target somegi_rhi
```

Expected: 编译链接成功，无错误。

---

### Task 6: Sampler 绑定 — Command Buffer 端绑定采样器表

**Files:**
- Modify: `src/rhi/d3d12/d3d12_command.cpp:105-119`

- [ ] **Step 1: 替换 bindDescriptorSet() 中的 (void)smpParam 为实际绑定**

将 `bindDescriptorSet()` 中第 114 行的 `(void)smpParam;` 替换为：

```cpp
        if (smpParam != ~0u && d3dSet.samplerGpuHandle().ptr != 0) {
            if (m_boundPSO->isCompute())
                m_cmdList->SetComputeRootDescriptorTable(smpParam,
                    d3dSet.samplerGpuHandle());
            else
                m_cmdList->SetGraphicsRootDescriptorTable(smpParam,
                    d3dSet.samplerGpuHandle());
        }
```

完整的 `bindDescriptorSet()` 方法变为：

```cpp
void D3D12RHICommandBuffer::bindDescriptorSet(uint32_t slot,
                                               const RHIDescriptorSet& set) {
    auto& d3dSet = static_cast<const D3D12RHIDescriptorSet&>(set);
    if (m_boundPSO) {
        uint32_t resParam = m_boundPSO->getResourceParamForSet(slot);
        uint32_t smpParam = m_boundPSO->getSamplerParamForSet(slot);
        if (resParam != ~0u) {
            if (m_boundPSO->isCompute())
                m_cmdList->SetComputeRootDescriptorTable(resParam, d3dSet.gpuHandle());
            else
                m_cmdList->SetGraphicsRootDescriptorTable(resParam, d3dSet.gpuHandle());
        }
        if (smpParam != ~0u && d3dSet.samplerGpuHandle().ptr != 0) {
            if (m_boundPSO->isCompute())
                m_cmdList->SetComputeRootDescriptorTable(smpParam,
                    d3dSet.samplerGpuHandle());
            else
                m_cmdList->SetGraphicsRootDescriptorTable(smpParam,
                    d3dSet.samplerGpuHandle());
        }
    } else {
        m_cmdList->SetComputeRootDescriptorTable(slot, d3dSet.gpuHandle());
    }
}
```

- [ ] **Step 2: 验证编译通过**

```powershell
cmake --build build --target somegi_rhi
```

Expected: 编译链接成功，无错误。

- [ ] **Step 3: 提交**

```bash
git add src/rhi/d3d12/d3d12_device.h \
        src/rhi/d3d12/d3d12_device.cpp \
        src/rhi/d3d12/d3d12_descriptor.h \
        src/rhi/d3d12/d3d12_descriptor.cpp \
        src/rhi/d3d12/d3d12_command.cpp
git commit -m "实现 D3D12 Sampler 描述符表绑定

新增 GPU 可见采样器描述符堆及分配/重置管理。
DescriptorSet 构造时分配采样器空间，write() 中拷贝采样器到 GPU 堆。
bindDescriptorSet() 绑定采样器根参数表。
begin() 中调用 SetDescriptorHeaps 绑定双堆（CBV_SRV_UAV + Sampler）。

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: 全量编译验证

- [ ] **Step 1: 完整编译项目**

```powershell
cmake --build build --config Debug 2>&1 | Select-Object -Last 30
```

Expected: 编译零错误零警告。

- [ ] **Step 2: 检查最终文件状态**

```bash
git status
git diff --stat
```

Expected: 所有改动限于 `src/rhi/d3d12/` 和 `src/rhi/CMakeLists.txt`。
