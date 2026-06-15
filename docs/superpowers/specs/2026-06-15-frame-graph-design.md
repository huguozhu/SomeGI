# Frame Graph 渲染帧图系统 — 设计文档

日期： 2026-06-15
状态： 设计中
分支： frame_graph

---

## 1. 问题陈述

当前 SomeGI 渲染管线使用 `RenderPipeline`（线性执行表）管理 Pass 调度。存在以下痛点：

### 1.1 手动 Barrier 管理

每个渲染步骤的 `record` lambda 内嵌显式 `transitionImage()` 调用，开发者必须记住每个资源的当前 layout。布局依赖隐式的执行顺序 — 一旦调整步骤顺序，barrier 极易出错。当前 `registerPipelineSteps()` 中有超过 60 处手动 barrier，调试成本高。

### 1.2 线性执行模型

`RenderPipeline` 是扁平列表（`std::vector<RenderStep*>`），无法表达 Pass 间的并行性。例如 Shadow 和 RSM-Geometry 两个 Prepasse 互相独立，但无法自动并行执行。

### 1.3 无资源生命周期管理

临时资源（中间结果纹理、Scratch Buffer）由用户手动管理，做不到内存别名复用。每种 GI 技术的中间结果独占物理内存，即使它们的生命周期完全不重叠。

### 1.4 Pass 配置分散

Pass 注册在 `App::registerPipelineSteps()`（录制 lambda + 手工 barrier），开关控制在 `App::buildPipelineTable()`（`setEnabled`）。两者分离，新增 Pass 时容易遗漏。

---

## 2. 设计目标

| 目标 | 衡量标准 |
|---|---|
| 自动 Barrier 推导 | 用户只声明 read/write，graph 自动插入 barrier；Synchronization Validation Layer 零警告 |
| DAG 依赖图 | Pass 按拓扑序执行；无依赖的 Pass 可识别为可并行 |
| 资源别名复用 | 临时资源根据寿命分析合并内存分配，预期节省 30-50% 显存 |
| 死 Pass 剔除 | 输出不被消费的 Pass 自动跳过（cull propagation） |
| 双轨兼容 | 新旧系统通过 `m_useFrameGraph` 开关共存，逐个 Phase 迁移验证 |
| 可视化调试 | ImGui 面板展示 DAG 结构、资源寿命图、别名分组、Barrier 日志 |

---

## 3. 架构概览

### 3.1 文件布局

```
src/renderer/fg/
├── fg_common.h           — FGHandle, FGResourceDesc, FGTextureDesc, 通用类型
├── fg_resource_node.h    — FGResourceNode: 内部资源节点（寿命/别名/barrier 状态）
├── fg_pass_node.h        — FGPassNode: 内部 Pass 节点（读写边/执行回调）
├── fg_builder.h          — FGBuilder: Pass 声明期 API（createTexture/read/write）
├── fg_graph.h            — FrameGraph: 顶层入口（addPass/compile/execute）
├── fg_compiler.h         — FGCompiler: DAG 构建/剔除/拓扑排序/别名分析
├── fg_executor.h         — FGExecutor: 物理资源分配/barrier插入/执行
├── fg_resources.h        — FGResources: execute 期提供给 pass 的资源视图查询
└── fg_debug.h            — FGDebug: ImGui 可视化数据结构
```

### 3.2 三层架构

```
用户 API (FrameGraph / FGBuilder)
         │
    ┌────▼────┐
    │ Compile │  FGCompiler: Cull → Edges → Lifetimes → Alias → Sort
    └────┬────┘
         │
    ┌────▼────┐
    │ Execute │  FGExecutor: Alloc → Barriers → Dispatch → Update state
    └─────────┘
```

### 3.3 使用示例

```cpp
// === 初始化：导入持久资源 ===
auto gbDepth  = fg.importTexture("GBufDepth",  depthImage,  depthDesc,  layout);
auto gbNormal = fg.importTexture("GBufNormal", normalImage, normalDesc, layout);

// === 每帧 ===
fg.reset();

// 声明临时资源（帧内生命周期，可 aliasing）
auto ssao = fg.createTexture("SSAO", {extent, R8_UNORM, ...});

// 声明 Pass：只声明依赖，不写 barrier
fg.addPass("SSAO", [&](FGBuilder& b) {
    b.read(gbDepth);
    b.read(gbNormal);
    b.write(ssao);

    b.setExecute([=](VkCommandBuffer cmd, const FGResources& r) {
        auto depthView  = r.getTextureView(gbDepth);
        auto normalView = r.getTextureView(gbNormal);
        auto ssaoView   = r.getTextureView(ssao);
        // 直接 dispatch，无需手动 transitionImage！
        ssaoPass.dispatch(cmd, depthView, normalView, ssaoView);
    });
});

// 编译 + 执行
fg.compile();
fg.execute(cmd);
```

---

## 4. 资源系统

### 4.1 FGHandle

轻量级不透明句柄，32-bit index + 32-bit generation（用于复用检测和 debug）。

```cpp
struct FGHandle {
    uint32_t index = UINT32_MAX;
    uint32_t generation = 0;
    bool valid() const { return index != UINT32_MAX; }
};
```

### 4.2 资源描述符

```cpp
struct FGTextureDesc {
    VkExtent3D extent{1,1,1};
    VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t   mipLevels = 1;
    uint32_t   arrayLayers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageUsageFlags usage = 0;
    bool isCubemap = false;
};

struct FGBufferDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
};

struct FGResourceDesc {
    FGResourceType type;   // Texture 或 Buffer
    union {
        FGTextureDesc texture;
        FGBufferDesc  buffer;
    };
    const char* debugName = nullptr;
};
```

### 4.3 资源分类

| 类型 | 所有者 | 生命周期 | 创建方式 | 可 Aliasing |
|---|---|---|---|---|
| 导入 (Imported) | 外部 (RenderTargets) | 跨帧持久 | `fg.importTexture(name, vkImage, desc, layout)` | 否 |
| 托管 (Managed) | FrameGraph | 单帧 | `fg.createTexture(name, desc)` | 是 |

---

## 5. Pass 声明 API (FGBuilder)

```cpp
class FGBuilder {
public:
    // 显式指定 Pass 类型（覆盖自动推导）
    void setPassType(FGPassType type);

    // 声明读：push-constant / descriptor 只读
    FGHandle read(FGHandle handle);

    // 声明写：render target / storage image / storage buffer
    FGHandle write(FGHandle handle);

    // 声明读-修改-写（in-place update）
    FGHandle readWrite(FGHandle handle);

    // 创建托管资源
    FGHandle createTexture(const char* name, const FGTextureDesc& desc);
    FGHandle createBuffer(const char* name, const FGBufferDesc& desc);

    // 设置执行回调（compile 之后、execute 时调用）
    template<typename F>
    void setExecute(F&& fn);
    // fn: void(VkCommandBuffer cmd, const FGResources& resources)
};
```

**使用示例 — Ray Tracing Pass：**

```cpp
fg.addPass("RT-Shadow", [&](FGBuilder& b) {
    b.setPassType(FGPassType::RayTracing);   // 使用 RT pipeline stage

    b.read(m_tlas);              // TLAS → RAY_TRACING_SHADER_BIT
    b.write(m_shadowMask);       // storage → RAY_TRACING_SHADER_BIT

    b.setExecute([=](VkCommandBuffer cmd, const FGResources& r) {
        auto tlas = r.getBuffer(m_tlas);
        auto mask = r.getTextureView(m_shadowMask);
        // vkCmdTraceRaysKHR(...)
    });
});

// 使用示例 — Mesh Shader Pass：
fg.addPass("GBuffer-Mesh", [&](FGBuilder& b) {
    b.setPassType(FGPassType::MeshShading);  // 使用 MESH/TASK stage

    b.read(m_sceneData);         // buffer → MESH_SHADER_BIT
    b.write(m_gbufferA);         // color attachment
    b.write(m_depth);            // depth attachment

    b.setExecute([=](VkCommandBuffer cmd, const FGResources& r) {
        auto scene = r.getBuffer(m_sceneData);
        // vkCmdDrawMeshTasksEXT(...)
    });
});

### 5.1 Pass 类型枚举

Pass 类型通过 `FGBuilder::setPassType()` 显式指定，决定 read/write 推导的 pipeline stage：

```cpp
enum class FGPassType {
    Compute,         // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
    Graphics,        // VERTEX + FRAGMENT (传统光栅化)
    MeshShading,     // TASK + MESH + FRAGMENT (mesh shader 管线)
    RayTracing,      // VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
};

// 默认推导规则：FGBuilder 根据 write 资源 usage 自动设置
//   - write() 含 COLOR_ATTACHMENT/DEPTH_STENCIL → Graphics
//   - write() 仅含 STORAGE → Compute
//   - 显式调用 setPassType() 覆盖自动推导
```

### 5.2 Layout 自动推导规则

**读取操作：**

| 声明 | Pass 类型 | 推导 Layout | 推导 Access | 推导 Pipeline Stage |
|---|---|---|---|---|
| `read()` (texture) | Compute | `SHADER_READ_ONLY_OPTIMAL` | `SHADER_SAMPLED_READ` | `COMPUTE_SHADER` |
| `read()` (texture) | Graphics | `SHADER_READ_ONLY_OPTIMAL` | `SHADER_SAMPLED_READ` | `FRAGMENT_SHADER` |
| `read()` (texture) | MeshShading | `SHADER_READ_ONLY_OPTIMAL` | `SHADER_SAMPLED_READ` | `MESH_SHADER_BIT_EXT \| FRAGMENT_SHADER` |
| `read()` (texture) | RayTracing | `SHADER_READ_ONLY_OPTIMAL` | `SHADER_SAMPLED_READ` | `RAY_TRACING_SHADER_BIT_KHR` |
| `read()` (buffer) | Compute | — | `SHADER_STORAGE_READ` | `COMPUTE_SHADER` |
| `read()` (buffer) | MeshShading | — | `SHADER_STORAGE_READ` | `MESH_SHADER_BIT_EXT` |
| `read()` (buffer) | RayTracing | — | `SHADER_STORAGE_READ` | `RAY_TRACING_SHADER_BIT_KHR` |

**写入操作：**

| 声明 | Resource Usage | 推导 Layout | 推导 Access | 推导 Pipeline Stage |
|---|---|---|---|---|
| `write()` (texture) | `COLOR_ATTACHMENT` | `COLOR_ATTACHMENT_OPTIMAL` | `COLOR_ATTACHMENT_WRITE` | `COLOR_ATTACHMENT_OUTPUT` |
| `write()` (texture) | `STORAGE` | `GENERAL` | `SHADER_STORAGE_WRITE` | `COMPUTE_SHADER` |
| `write()` (texture) | `STORAGE` (RayTracing) | `GENERAL` | `SHADER_STORAGE_WRITE` | `RAY_TRACING_SHADER_BIT_KHR` |
| `write()` (texture) | `DEPTH_STENCIL_ATTACHMENT` | `DEPTH_ATTACHMENT_OPTIMAL` | `DEPTH_STENCIL_ATTACHMENT_WRITE` | `EARLY_FRAGMENT_TESTS` |
| `write()` (buffer) | `STORAGE` | — | `SHADER_STORAGE_WRITE` | `COMPUTE_SHADER` |
| `write()` (buffer) | `STORAGE` (RayTracing) | — | `SHADER_STORAGE_WRITE` | `RAY_TRACING_SHADER_BIT_KHR` |
| `readWrite()` | `STORAGE` | `GENERAL` | `SHADER_STORAGE_READ \| WRITE` | 跟随 Pass 类型 |

**加速结构（Acceleration Structure）：**

| 声明 | 推导 Access | 推导 Pipeline Stage |
|---|---|---|
| `read()` (TLAS) | `SHADER_READ` | `RAY_TRACING_SHADER_BIT_KHR` / `COMPUTE_SHADER` |

> 推导基于 `FGTextureDesc::usage` 标记选择具体 layout。用户也可通过 `FGTextureDesc` 自定义 usage 覆盖默认推导。TLAS 通过 `createBuffer(ACCELERATION_STRUCTURE)` 声明。

### 5.2 使用示例（迁移现有 SSAO pass）

```cpp
// 旧代码：15+ 行手工 barrier
// 新代码：
fg.addPass("SSAO", [&](FGBuilder& b) {
    b.read(m_gbufferDepth);
    b.read(m_gbufferNormalRough);
    b.write(m_ssaoOutput);

    b.setExecute([=](VkCommandBuffer cmd, const FGResources& r) {
        ssaoPass.record(cmd,
            r.getTextureView(m_gbufferDepth),
            r.getTextureView(m_gbufferNormalRough),
            r.getTextureView(m_ssaoOutput));
    });
});
```

---

## 5.3 FGResources — execute 期资源查询

`FGResources` 是 `setExecute` 回调的第二个参数，封装了 handle → 物理资源的查询。

```cpp
class FGResources {
public:
    // 根据 handle 获取 VkImageView（texture）
    VkImageView getTextureView(FGHandle handle, uint32_t mip = 0,
                               uint32_t layer = 0) const;

    // 根据 handle 获取 VkBuffer + offset
    VkBuffer getBuffer(FGHandle handle, VkDeviceSize* outOffset = nullptr) const;

    // 获取资源描述信息
    VkExtent3D extent(FGHandle handle) const;

private:
    friend class FrameGraph;
    // 内部由 FrameGraph::execute() 在执行前填充 ResourceView 缓存
    FrameGraph* m_graph = nullptr;
};
```

`FGResources` 是轻量级视图层，不持有资源。每次 `execute()` 调用时，FrameGraph 先根据 alias group 分配结果更新物理指针，然后 FGResources 直接查询 `m_views[]` 缓存返回 `VkImageView`。视图在 `execute()` 开始时一次性创建，避免 pass 内重复调用 `vkCreateImageView`。

---

## 6. 内部图表示

### 6.1 FGPassNode

```cpp
struct FGPassNode {
    std::string name;
    FGPassType passType = FGPassType::Compute;  // Pass 类型（决定 pipeline stage）
    bool enabled = true;
    bool culled = false;                        // 编译后被剔除

    struct ResourceRef {
        FGHandle handle;
        FGResourceNode* resource = nullptr;  // 编译后指向实际节点
        VkAccessFlags2 access = 0;
        VkPipelineStageFlags2 stages = 0;
        VkImageLayout requiredLayout;
    };
    std::vector<ResourceRef> reads;
    std::vector<ResourceRef> writes;

    std::function<void(VkCommandBuffer, const FGResources&)> execute;

    // 编译后填充
    uint32_t topologicalIndex = 0;
    std::vector<FGPassNode*> predecessors;  // 前驱（用于并行分析）
};
```

### 6.2 FGResourceNode

```cpp
struct FGResourceNode {
    FGHandle handle;
    FGResourceDesc desc;
    bool isImported = false;

    uint32_t firstWritePass = UINT32_MAX;   // 首次写入 pass 序号
    uint32_t lastReadPass  = 0;             // 最后读取 pass 序号

    uint32_t aliasedGroup = UINT32_MAX;     // 别名组 ID
    uint32_t aliasedOffset = 0;             // 组内偏移

    // 物理资源（execute 阶段分配）
    Image*  physicalTexture = nullptr;
    Buffer* physicalBuffer = nullptr;

    // Barrier 追踪状态
    struct BarrierState {
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkAccessFlags2 access = 0;
        FGPassNode* lastWriter = nullptr;
    };
    BarrierState state;
};
```

### 6.3 依赖图构建

```
规则 1 (RAW)：Pass B read(R), R 上一 writer = Pass A → 边 A→B
规则 2 (WAW)：Pass B write(R), R 上一 writer = Pass A → 边 A→B
规则 3 (WAR)：Pass B write(R), R 上一 reader = Pass A → 边 A→B（仅在需要保序时）
```

---

## 7. 编译阶段 (FGCompiler)

### 7.1 编译流程

```
输入: passes[] + resources[]

步骤 1 — CullPasses
  ├─ 标记 enabled=false 的 pass 为 culled
  └─ 传播：唯一消费者被 cull ⇒ 生产者也被 cull

步骤 2 — BuildEdges
  ├─ 扫描每个 pass 的 read/write 列表
  ├─ RAW: 追踪每个 resource 的 lastWriter → 当前 reader 建边
  ├─ WAW: 追踪每个 resource 的 lastWriter → 当前 writer 建边
  └─ 填充每个 pass 的 predecessors[]

步骤 3 — ComputeLifetimes
  ├─ firstWrite: 首次写该 resource 的 pass 序号
  └─ lastRead:   最后读该 resource 的 pass 序号

步骤 4 — ComputeAliasing（仅托管资源）
  ├─ 按 resource size 降序排列
  └─ 贪心分配：lifetime [firstWrite, lastRead] 不重叠 ⇒ 同 alias group
      每个 group 的 sizeBytes = max(组成员 size)

步骤 5 — TopologicalSort
  └─ Kahn 算法（BFS + 入度），输出 passOrder[]

输出: CompiledGraph { passOrder, resources, culledPasses, aliasGroups }
```

### 7.2 别名分析示例

```
  Pass0   Pass1   Pass2   Pass3   Pass4
    │       │       │       │       │
 R0[写]──[读]─┤     │       │       │     R0 寿命: 0→1
    │       │       │       │       │
    │  R1[写]───[读]─┤      │       │     R1 寿命: 1→2
    │       │       │       │       │
    │       │  R2[写]──────────[读]──┤     R2 寿命: 2→4

R0、R1、R2 互不重叠 → Alias Group 0: {R0, R1, R2}
物理内存 = max(size(R0), size(R1), size(R2))
```

### 7.3 编译验证

编译阶段检测以下问题：

- 资源被读但从未被写 → 编译错误
- 同一 pass 内对同一资源既 read 又 write 但未用 readWrite → 警告
- 导入资源 initialLayout = UNDEFINED → 警告
- 循环依赖 → 编译错误（拓扑排序检测）

---

## 8. 执行阶段 (FGExecutor)

### 8.1 执行流程

```
FrameGraph::execute(cmd):
  1. allocateAliasGroups()
     └─ 为每个 alias group 分配/复用物理 Image/Buffer
  2. for pass in compiled.passOrder:
       a. emitBarriers(cmd, pass)
          └─ 对比 resource state vs pass 需要的 layout
             不匹配 → 插入 VkImageMemoryBarrier2 / VkBufferMemoryBarrier2
       b. pass.execute(cmd, fgResources)
       c. updateResourceStates(pass)
          └─ 更新 resource.state (layout/stage/access/lastWriter)
  3. recycleUnused()
     └─ 超过 N 帧未使用的池资源回收释放
```

### 8.2 自动 Barrier 逻辑

```cpp
void emitBarriers(VkCommandBuffer cmd, const FGPassNode& pass,
                  std::vector<FGResourceNode*>& resources) {
    std::vector<VkImageMemoryBarrier2> imgBarriers;

    for (auto& ref : pass.reads) {
        auto& res = *resources[ref.handle.index];
        if (res.state.layout == ref.requiredLayout &&
            res.state.access == ref.access) continue;  // 无需 barrier

        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.oldLayout     = res.state.layout;
        b.newLayout     = ref.requiredLayout;
        b.srcStageMask  = res.state.stage;
        b.srcAccessMask = res.state.access;
        b.dstStageMask  = ref.stages;
        b.dstAccessMask = ref.access;
        b.image         = res.physicalTexture->image();
        b.subresourceRange = {aspect, 0, res.desc.texture.mipLevels, 0, 1};
        imgBarriers.push_back(b);
    }

    // 同理处理 writes...

    if (!imgBarriers.empty()) {
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = (uint32_t)imgBarriers.size();
        di.pImageMemoryBarriers = imgBarriers.data();
        vkCmdPipelineBarrier2(cmd, &di);
    }
}
```

### 8.3 资源池

```cpp
class FGExecutor {
    // Texture 池：按 format + extent 索引
    struct TextureAlloc {
        Image image;
        uint64_t lastUsedFrame;
    };
    std::vector<TextureAlloc> m_texturePool;

    // Buffer 池：按 size 索引
    struct BufferAlloc {
        Buffer buffer;
        uint64_t lastUsedFrame;
    };
    std::vector<BufferAlloc> m_bufferPool;
};
```

分配策略：先在池中查找匹配（format + size 一致且 `lastUsedFrame` 超出 recycle 阈值）的闲置资源，未命中才 `vkCreateImage`/`vkCreateBuffer`。

---

## 9. 顶层 API (FrameGraph)

```cpp
class FrameGraph {
public:
    // 资源导入
    FGHandle importTexture(const char* name, VkImage, const FGTextureDesc&, VkImageLayout initial);

    // 资源声明（托管）
    FGHandle createTexture(const char* name, const FGTextureDesc&);
    FGHandle createBuffer(const char* name, const FGBufferDesc&);

    // Pass 声明
    void addPass(const char* name, std::function<void(FGBuilder&)> setup);

    // 编译 + 执行
    void compile();
    void execute(VkCommandBuffer cmd);

    // 查询
    VkImageView getTextureView(FGHandle, uint32_t mip=0, uint32_t layer=0) const;

    // 帧管理
    void reset();

    // 调试
    const FGCompiler::CompiledGraph& compiledGraph() const;
    FGDebug& debug();

private:
    std::vector<FGPassNode>     m_passes;
    std::vector<FGResourceNode> m_resources;
    std::unordered_map<std::string, uint32_t> m_resourceNameMap;
    FGCompiler m_compiler;
    FGExecutor m_executor;
    FGDebug    m_debug;
    bool       m_compiled = false;
    std::vector<ResourceView> m_views;
};
```

---

## 10. 调试与可视化 (FGDebug)

ImGui 面板位于 Debug 窗口的 "FrameGraph" Tab，提供以下视图：

### 10.1 Pass 执行列表

表格显示所有 Pass：序号、名称、读取资源、写入资源、GPU 耗时、culled 标记。

### 10.2 别名分组

展示每个 alias group 包含的资源及节省的显存量。

### 10.3 资源寿命图

时间轴可视化（ASCII art），每行为一个资源，横轴为 pass 序号，"██" 表示存活区间。直观展示别名机会。

### 10.4 DAG 文本表示

终端打印 ASCII 依赖图，快速确认 Pass 间依赖关系。

### 10.5 Barrier 日志

编译期打印每个 barrier 的详细决策（oldLayout→newLayout, srcStage→dstStage），辅助调试同步问题。

---

## 11. 迁移策略

### 11.1 双轨并行架构

```
App::run()
    │
    ├── m_useFrameGraph == true  → FrameGraph 路径
    └── m_useFrameGraph == false → RenderPipeline 路径（现有）
```

ImGui 开关：
```
☑ Use Frame Graph (experimental)
```

### 11.2 迁移 Phase

| Phase | 内容 | Pass 数 | 验证标准 |
|---|---|---|---|
| 0 | 框架搭建：FGHandle/FGBuilder/FGCompiler/FGExecutor 空跑 | 0 | 编译通过，不影响现有流程 |
| 1 | 纯 Compute Pass：SSAO / GTAO | 2 | 输出像素一致，无 sync validation 警告 |
| 2 | 屏幕空间 Pass：SSR / SSGI / GTGI | 3 | 带 copy+barrier 的复杂流程 |
| 3 | Graphics Pass：GBuffer / Forward | 2 | MSAA resolve、depth stencil |
| 4 | GI 系统：VXGI chain / LPV / DDGI / SDFGI / ReSTIR | 10+ | 多 Pass 依赖链 |
| 5 | 余下 Pass：Shadow / Skybox / Tonemap / TAA / SMAA / Lighting | 10+ | 全 Pass 覆盖 |
| 6 | 清理：删除旧 RenderPipeline、清理 App 旧 barrier 代码 | — | 无遗留代码 |

### 11.3 回退策略

- 关闭 `m_useFrameGraph` 立即回退到旧管线
- 每个 Phase 独立提交，互不阻塞

---

## 12. 关键设计决策摘要

| 决策 | 选择 | 理由 |
|---|---|---|
| 资源模型 | 句柄模式 (Handle-based) | 支持 aliasing，与 Frostbite/UE5 对齐 |
| Pass API | Lambda + FGBuilder | 保持与现有 registerPipelineSteps 风格一致，迁移摩擦最小 |
| 迁移方式 | 双轨并行 | 安全逐步验证，风险可控 |
| 作用域 | 完整 Frame Graph | DAG + 自动 barrier + aliasing + 可视化 |
| Layout 推导 | 基于 usage 标记自动选择 | 覆盖 95% 场景，复杂情况用 usage 覆盖 |
| 内存管理 | 池化分配 + 帧热度淘汰 | 平衡分配开销和内存占用 |
| 可视化 | ImGui Tab + 终端 ASCII | 快速确认 DAG 正确性，无需外部工具 |
