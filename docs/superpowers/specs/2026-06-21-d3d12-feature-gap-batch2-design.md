# D3D12 功能差距补齐 Batch 2 设计文档

日期：2026-06-21
分支：d3d12
状态：设计阶段
父文档：[[rhi-design]]

---

## 1. 背景

Batch 1（commit `84feb0a`）补齐了以下 D3D12 缺失功能：
`bufferBarrier`（UAV barrier）、`fillBuffer`、`pushConstants`（双路径）、
`writeTimestamp`、`resetQueryPool`、`createQueryPool`。

Batch 2（commit `6a8b695`）补了 `clearColor` + `clearDepth`。

本轮 Batch 3 继续补齐 3 个剩余差距：

| # | 差距 | 当前状态 | 类型 |
|---|------|----------|------|
| 1 | Buffer Barrier | 无视 `PipelineStage`/`BufferAccess` 参数，统一 UAV barrier | 逻辑修正 |
| 2 | Sampler 描述符表绑定 | `bindDescriptorSet` 跳过 sampler 参数 | 功能补全 |
| 3 | AccelerationStructure | Header-only 空壳，`nativeHandle()` 返回 `nullptr` | 功能补全 |

---

## 2. 模块 1：Buffer Barrier 阶段/访问翻译

### 2.1 问题

`d3d12_command.cpp:392-400` 当前实现：

```cpp
void D3D12RHICommandBuffer::bufferBarrier(const RHIBuffer& buf,
                                           PipelineStage, PipelineStage,
                                           BufferAccess, BufferAccess) {
    // 四个参数全部未使用
    D3D12_RESOURCE_BARRIER rb{};
    rb.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    rb.UAV.pResource = d3dBuf.resource();
    m_cmdList->ResourceBarrier(1, &rb);
}
```

无法区分为 Vertex Buffer → Compute Shader 等转换，所有 buffer barrier 等同 UAV barrier。

### 2.2 设计

改为 `D3D12_RESOURCE_BARRIER_TYPE_TRANSITION`，复用已有的 `m_resourceStates` 状态追踪器。

#### 映射表：BufferAccess → D3D12_RESOURCE_STATES

| BufferAccess | D3D12_RESOURCE_STATES |
|---|---|
| `UniformRead` | `VERTEX_AND_CONSTANT_BUFFER \| NON_PIXEL_SHADER_RESOURCE` |
| `StorageRead` | `UNORDERED_ACCESS` |
| `StorageWrite` | `UNORDERED_ACCESS` |
| `IndexRead` | `INDEX_BUFFER` |
| `VertexRead` | `VERTEX_AND_CONSTANT_BUFFER` |
| `IndirectRead` | `INDIRECT_ARGUMENT` |
| `TransferRead` | `COPY_SOURCE` |
| `TransferWrite` | `COPY_DEST` |
| `None` | 保持当前状态（仅同步，不转换） |
| 组合位标志 | 状态 OR |

> `PipelineStage` 参数记录但不直接映射到 D3D12_RESOURCE_STATES，
> 因为 D3D12 buffer 状态主要由访问类型决定，而非管线阶段。
> 保留参数用于未来可能的验证/调试断言。

#### 实现逻辑

```
srcState = toD3D12ResourceStates(srcAccess)  // 映射表
dstState = toD3D12ResourceStates(dstAccess)

before = device.getResourceState(buf)  // 状态追踪器当前值
after  = dstState

if (before == after) return  // 无需 barrier

device.trackResourceState(buf, after)  // 更新追踪器

// 发出 TRANSITION barrier
D3D12_RESOURCE_BARRIER rb{};
rb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
rb.Transition.pResource = buf.resource();
rb.Transition.StateBefore = before;
rb.Transition.StateAfter = after;
rb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
```

#### UAV → UAV 特殊情况

当 `before == after == UNORDERED_ACCESS` 时（如 Compute → Compute），
TRANSITION barrier 不允许 StateBefore == StateAfter。
此时 fallback 为 UAV barrier（类型不变但确保 UAV 写入完成）。

#### 改动范围

- **修改**：`src/rhi/d3d12/d3d12_command.cpp`
  - 新增 `toD3D12ResourceStates(BufferAccess)` 静态辅助函数
  - 重写 `bufferBarrier()` 方法体

---

## 3. 模块 2：Sampler 描述符表绑定

### 3.1 问题

`DescriptorWrite` 接口已有 `sampler` 字段（`base/descriptor.h:62`），
`D3D12RHIDescriptorSetLayout` 已有 `samplerParamIdx` 跟踪，
但 `D3D12RHIDescriptorSet::write()` 未处理采样器写入，
`bindDescriptorSet()` 第 114 行明确标注"采样器需要单独的表（暂时不做）"。

结果：独立采样器无法通过描述符表绑定到着色器。

### 3.2 设计

#### 3.2.1 Device：GPU 可见 Sampler 描述符堆

在 `D3D12RHIDevice` 中新增：

```cpp
// 新增成员
ID3D12DescriptorHeap* m_gpuSamplerHeap = nullptr;
uint32_t m_gpuSamplerIncrement = 0;
D3D12_CPU_DESCRIPTOR_HANDLE m_gpuSamplerStartCPU{};
D3D12_GPU_DESCRIPTOR_HANDLE m_gpuSamplerStartGPU{};
uint32_t m_gpuSamplerOffset = 0;
static constexpr uint32_t kGpuSamplerHeapSize = 2048;  // sampler 数量远少于资源

// 新增方法（模式与 allocDescriptors 一致）
DescAlloc allocSamplerDescriptors(uint32_t count);
void resetSamplerHeap();  // 每帧与 resetDescriptorHeap() 一起调用
```

#### 3.2.2 D3D12RHIDescriptorSetLayout：分离资源/采样器计数

在构造函数中统计 sampler 绑定数量（已有 `m_samplerParamIdx`），
为 descriptor set 的 sampler 堆分配提供计数信息：

```cpp
// 新增方法
uint32_t samplerCount() const { return m_samplerCount; }
// 新增成员
uint32_t m_samplerCount = 0;
```

#### 3.2.3 D3D12RHIDescriptorSet：存储 Sampler GPU 句柄

```cpp
// 新增成员
D3D12_GPU_DESCRIPTOR_HANDLE m_samplerGpuStart{};
uint32_t m_samplerCount = 0;

// 新增公开方法
D3D12_GPU_DESCRIPTOR_HANDLE samplerGpuHandle() const { return m_samplerGpuStart; }
```

构造函数中，若 layout 的 `samplerCount() > 0`，调用 `allocSamplerDescriptors()` 分配空间。

#### 3.2.4 D3D12RHIDescriptorSet::write()：处理采样器写入

在现有纹理写入循环中增加分支：

```cpp
if (w.sampler) {
    auto* smp = static_cast<const D3D12RHISampler*>(w.sampler);
    D3D12_CPU_DESCRIPTOR_HANDLE src = smp->cpuHandle();
    D3D12_CPU_DESCRIPTOR_HANDLE dst = m_device.gpuSamplerHeap()->GetCPUDescriptorHandleForHeapStart();
    dst.ptr += (m_samplerGpuStart.ptr - m_device.gpuSamplerHeap()->GetGPUDescriptorHandleForHeapStart().ptr)
               + samplerIndex * m_device.gpuSamplerIncrement();
    m_device.device()->CopyDescriptorsSimple(1, dst, src, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    samplerIndex++;
}
```

#### 3.2.5 bindDescriptorSet()：绑定采样器表

```cpp
// 替换当前的 (void)smpParam
if (smpParam != ~0u && d3dSet.samplerGpuHandle().ptr != 0) {
    if (m_boundPSO->isCompute())
        m_cmdList->SetComputeRootDescriptorTable(smpParam, d3dSet.samplerGpuHandle());
    else
        m_cmdList->SetGraphicsRootDescriptorTable(smpParam, d3dSet.samplerGpuHandle());
}
```

#### 3.2.6 堆绑定

D3D12 要求同时设置 shader-visible 的 CBV/SRV/UAV 堆和 Sampler 堆。
在需要 sampler 的命令列表操作前，调用：

```cpp
ID3D12DescriptorHeap* heaps[] = { m_device.gpuDescriptorHeap(), m_device.gpuSamplerHeap() };
m_cmdList->SetDescriptorHeaps(2, heaps);
```

由于当前代码在 `begin()` 时已设置单个 descriptor heap，需扩展为同时设置两个堆。

#### 改动范围

| 文件 | 改动 |
|------|------|
| `d3d12_device.h` | 新增 `m_gpuSamplerHeap` 成员、`allocSamplerDescriptors()`、`resetSamplerHeap()` 声明 |
| `d3d12_device.cpp` | 创建堆、实现分配/重置方法、`begin()` 中设置双堆 |
| `d3d12_descriptor.h` | `D3D12RHIDescriptorSet` 新增 `m_samplerGpuStart`、`samplerGpuHandle()`；`D3D12RHIDescriptorSetLayout` 新增 `samplerCount()` |
| `d3d12_descriptor.cpp` | 构造时分配 sampler 堆、`write()` 中处理采样器写入 |
| `d3d12_command.cpp` | `bindDescriptorSet()` 中绑定采样器表 |

---

## 4. 模块 3：AccelerationStructure 包装

### 4.1 问题

`d3d12_acceleration_structure.h` 是 header-only 空壳：

```cpp
class D3D12RHIAccelerationStructure : public RHIAccelerationStructure {
public:
    void* nativeHandle() const override { return nullptr; }
};
```

Vulkan 对应实现 `VkRHIAccelerationStructure` 支持 owning/non-owning 两种模式，
non-owning 用于包装外部已创建的 TLAS/BLAS。

### 4.2 设计

新增 `d3d12_acceleration_structure.cpp`，效仿 Vulkan 的包装模式。

#### 类结构

```cpp
class D3D12RHIAccelerationStructure : public RHIAccelerationStructure {
public:
    // 非拥有型工厂：包装外部创建的 TLAS/BLAS buffer
    static std::unique_ptr<RHIAccelerationStructure> createNonOwning(ID3D12Resource* as);

    ~D3D12RHIAccelerationStructure() override;
    void* nativeHandle() const override { return m_resource; }

private:
    // 拥有型构造（仅 createNonOwning 内部使用，后续 Phase 扩展为 owning）
    D3D12RHIAccelerationStructure(ID3D12Resource* resource, bool owns);
    ID3D12Resource* m_resource = nullptr;
    bool m_owns = false;
};
```

#### 关键细节

- D3D12 中 TLAS/BLAS 构建结果存储在 `ID3D12Resource` 中，包装此 resource
- 非拥有模式：`createNonOwning()` 创建不管理生命周期的包装器
- 拥有模式：预留 `owns = true` 路径，析构时 `m_resource->Release()`
- 接口与 Vulkan `VkRHIAccelerationStructure::createNonOwning(device, vkAs)` 对齐

#### 改动范围

| 文件 | 改动 |
|------|------|
| `d3d12_acceleration_structure.h` | 更新类声明，添加方法 |
| `d3d12_acceleration_structure.cpp` | **新增**，实现 constructor/destructor/createNonOwning |

---

## 5. 实施顺序

三个模块相互独立，建议按复杂度递增顺序实施：

1. **模块 3**（AccelerationStructure）— 最简单，新增 1 个 cpp，约 30 行
2. **模块 1**（Buffer Barrier）— 修改 1 个文件，约 40 行
3. **模块 2**（Sampler 绑定）— 修改 5 个文件，约 80 行

### 验证方式

- **模块 1**：编译通过 + 运行时 buffer barrier 不再产生 D3D12 debug layer 警告
- **模块 2**：使用独立采样器的材质渲染正确（可通过 PIX 验证描述符表绑定）
- **模块 3**：编译通过 + 与 Vulkan 的非拥有型接口一致

---

## 6. 不纳入本轮的范围

以下功能明确延后到后续 Batch：

- `drawMeshTasks` / `drawMeshTasksIndirect`（依赖 `ID3D12GraphicsCommandList6::DispatchMesh` + 硬件 Mesh Shader 支持）
- `createRayTracingPSO`（依赖 `ID3D12Device5::CreateStateObject` + DXR 支持）
- `m_limits.meshShaderSupported` / `m_limits.rayTracingSupported` 运行时检测（与上述功能配套）
- 静态采样器替代方案（当前设计使用动态描述符表，后续可评估静态采样器优化）
