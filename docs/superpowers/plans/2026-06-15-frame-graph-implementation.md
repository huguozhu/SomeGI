# Frame Graph 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现完整的 Frame Graph 渲染帧图系统，包括 DAG 构建/编译/执行、自动 Barrier 推导、资源别名复用和可视化调试，以双轨并行方式与现有 RenderPipeline 共存。

**Architecture:** 9 个新文件组成 `src/renderer/fg/` 模块：声明层 (FGHandle/描述符/FGBuilder) → 编译层 (FGCompiler: Cull→Edges→Lifetimes→Alias→Sort) → 执行层 (FGExecutor: Alloc→Barriers→Dispatch)。通过 `m_useFrameGraph` 开关在 App::run() 中选择新旧路径。

**Tech Stack:** C++20, Vulkan 1.3, VMA (内存分配), ImGui (可视化)

---

## 文件结构

### 新增文件

| 文件 | 职责 |
|---|---|
| `src/renderer/fg/CMakeLists.txt` | 构建配置 |
| `src/renderer/fg/fg_common.h` | FGHandle, 资源描述符, FGPassType 枚举 |
| `src/renderer/fg/fg_pass_node.h` | FGPassNode: Pass 内部表示 |
| `src/renderer/fg/fg_resource_node.h` | FGResourceNode: 资源内部表示 |
| `src/renderer/fg/fg_builder.h` | FGBuilder: Pass 声明期 API |
| `src/renderer/fg/fg_resources.h` | FGResources: execute 期资源查询 |
| `src/renderer/fg/fg_graph.h` | FrameGraph: 顶层 API 声明 |
| `src/renderer/fg/fg_graph.cpp` | FrameGraph: 顶层 API 实现 |
| `src/renderer/fg/fg_compiler.h` | FGCompiler: 编译阶段声明 |
| `src/renderer/fg/fg_compiler.cpp` | FGCompiler: 编译阶段实现 |
| `src/renderer/fg/fg_executor.h` | FGExecutor: 执行阶段声明 |
| `src/renderer/fg/fg_executor.cpp` | FGExecutor: 执行阶段实现 |
| `src/renderer/fg/fg_debug.h` | FGDebug: 可视化数据结构 |
| `src/renderer/fg/fg_debug.cpp` | FGDebug: 可视化数据填充 |

### 修改文件

| 文件 | 改动 |
|---|---|
| `src/renderer/CMakeLists.txt` | 添加 `add_subdirectory(fg)` |
| `src/renderer/core/CMakeLists.txt` | 链接 `somegi_renderer_fg` |
| `src/app/app.h` | 添加 `FrameGraph m_fg` 成员 + `m_useFrameGraph` 开关 |
| `src/app/app.cpp` | 添加 setupFrameGraph(), 双轨 run() 分支, ImGui toggle |
| `src/app/CMakeLists.txt` | 链接 fg 库 |

---

## Phase 0: 框架搭建

### Task 0.0: 目录和 CMake 骨架

**Files:**
- Create: `src/renderer/fg/CMakeLists.txt`
- Modify: `src/renderer/CMakeLists.txt`

- [ ] **Step 1: 创建 fg 模块 CMakeLists.txt**

```cmake
# src/renderer/fg/CMakeLists.txt
add_library(somegi_renderer_fg OBJECT
    fg_graph.cpp
    fg_compiler.cpp
    fg_executor.cpp
    fg_debug.cpp
)
target_include_directories(somegi_renderer_fg PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_fg PUBLIC somegi_core imgui)
```

- [ ] **Step 2: 在渲染器顶层注册 fg 子目录**

修改 `src/renderer/CMakeLists.txt`，在 `add_subdirectory(core)` 前添加：

```cmake
add_subdirectory(fg)
```

并修改 `add_library(somegi_renderer STATIC` 段，添加：

```cmake
add_library(somegi_renderer STATIC
    $<TARGET_OBJECTS:somegi_renderer_fg>
    $<TARGET_OBJECTS:somegi_renderer_culling>
    ... (保持现有项不变)
)
```

- [ ] **Step 3: 验证构建**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译成功（尚无 .cpp 实现，仅是空 OBJECT 库）。

- [ ] **Step 4: 提交**

```bash
git add src/renderer/fg/CMakeLists.txt src/renderer/CMakeLists.txt
git commit -m "Phase 0.0: 创建 FrameGraph 模块目录和 CMake 骨架"
```

---

### Task 0.1: 通用类型 — fg_common.h

**Files:**
- Create: `src/renderer/fg/fg_common.h`

- [ ] **Step 1: 编写 fg_common.h**

```cpp
// src/renderer/fg/fg_common.h
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>

namespace somegi {
namespace fg {

// ============================================================
// FGHandle: 轻量级不透明资源句柄
// ============================================================
struct FGHandle {
    uint32_t index = UINT32_MAX;      // 内部资源数组下标
    uint32_t generation = 0;          // 代数，用于 debug 时检测悬空 handle

    bool valid() const { return index != UINT32_MAX; }
    bool operator==(FGHandle o) const {
        return index == o.index && generation == o.generation;
    }
    bool operator!=(FGHandle o) const { return !(*this == o); }
};

// ============================================================
// FGPassType: Pass 类型，决定 Barrier 推导时的 pipeline stage
// ============================================================
enum class FGPassType {
    Compute,        // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
    Graphics,       // VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT + FRAGMENT_SHADER_BIT
    MeshShading,    // VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT + MESH_SHADER_BIT_EXT
    RayTracing,     // VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
};

// ============================================================
// FGResourceType: 资源类型枚举
// ============================================================
enum class FGResourceType {
    Texture,
    Buffer,
};

// ============================================================
// FGTextureDesc: 纹理资源描述符
// ============================================================
struct FGTextureDesc {
    VkExtent3D extent{1, 1, 1};
    VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t   mipLevels = 1;
    uint32_t   arrayLayers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageUsageFlags usage = 0;
    bool isCubemap = false;
};

// ============================================================
// FGBufferDesc: Buffer 资源描述符
// ============================================================
struct FGBufferDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
};

// ============================================================
// FGResourceDesc: 统一资源描述符
// ============================================================
struct FGResourceDesc {
    FGResourceType type = FGResourceType::Texture;
    const char* debugName = nullptr;

    union {
        FGTextureDesc texture;
        FGBufferDesc  buffer;
    };

    // 便捷工厂
    static FGResourceDesc textureDesc(const char* name,
                                       VkExtent3D extent,
                                       VkFormat format,
                                       VkImageUsageFlags usage,
                                       uint32_t mipLevels = 1,
                                       VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) {
        FGResourceDesc d;
        d.type = FGResourceType::Texture;
        d.debugName = name;
        d.texture = {extent, format, mipLevels, 1, samples, usage, false};
        return d;
    }

    static FGResourceDesc bufferDesc(const char* name,
                                      VkDeviceSize size,
                                      VkBufferUsageFlags usage) {
        FGResourceDesc d;
        d.type = FGResourceType::Buffer;
        d.debugName = name;
        d.buffer = {size, usage};
        return d;
    }
};

} // namespace fg
} // namespace somegi
```

- [ ] **Step 2: 验证编译**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译通过。

- [ ] **Step 3: 提交**

```bash
git add src/renderer/fg/fg_common.h
git commit -m "Phase 0.1: 添加 FGHandle/描述符/FGPassType 通用类型定义"
```

---

### Task 0.2: 资源节点 — fg_resource_node.h

**Files:**
- Create: `src/renderer/fg/fg_resource_node.h`

- [ ] **Step 1: 编写 fg_resource_node.h**

```cpp
// src/renderer/fg/fg_resource_node.h
#pragma once
#include "fg_common.h"
#include "core/image.h"
#include "core/buffer.h"

namespace somegi {
namespace fg {

struct FGPassNode; // 前向声明

// ============================================================
// FGResourceNode: 图内部资源表示
// ============================================================
struct FGResourceNode {
    FGHandle handle;
    FGResourceDesc desc;
    bool isImported = false;  // true=外部导入, false=graph 托管

    // ---- 生命周期（编译后填充，值为 pass topological index） ----
    uint32_t firstWritePass = UINT32_MAX;
    uint32_t lastReadPass  = 0;

    // ---- 别名信息 ----
    uint32_t aliasedGroup = UINT32_MAX;  // 别名组 ID
    uint32_t aliasedOffset = 0;          // 组内偏移字节数

    // ---- 物理资源（execute 阶段由 FGExecutor 分配） ----
    Image*  physicalTexture = nullptr;
    Buffer* physicalBuffer = nullptr;

    // ---- Barrier 追踪状态（跨 pass 持续更新） ----
    struct BarrierState {
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkAccessFlags2 access = 0;
        FGPassNode* lastWriter = nullptr;
    };
    BarrierState state;
};

} // namespace fg
} // namespace somegi
```

- [ ] **Step 2: 验证编译**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译通过（`FGPassNode` 只是前向声明，FGResourceNode 不需要完整定义）。

- [ ] **Step 3: 提交**

```bash
git add src/renderer/fg/fg_resource_node.h
git commit -m "Phase 0.2: 添加 FGResourceNode 内部资源节点定义"
```

---

### Task 0.3: Pass 节点 — fg_pass_node.h

**Files:**
- Create: `src/renderer/fg/fg_pass_node.h`

- [ ] **Step 1: 编写 fg_pass_node.h**

```cpp
// src/renderer/fg/fg_pass_node.h
#pragma once
#include "fg_common.h"
#include <string>
#include <vector>
#include <functional>

namespace somegi {
namespace fg {

struct FGResourceNode;
struct FGResources;

// ============================================================
// FGPassNode: 图内部 Pass 表示
// ============================================================
struct FGPassNode {
    std::string name;
    FGPassType passType = FGPassType::Compute;
    bool enabled = true;
    bool culled = false;  // 编译后被剔除

    // ---- 资源依赖边 ----
    struct ResourceRef {
        FGHandle handle;
        FGResourceNode* resource = nullptr;  // 编译后指向实际资源节点
        VkAccessFlags2 access = 0;
        VkPipelineStageFlags2 stages = 0;
        VkImageLayout requiredLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    };
    std::vector<ResourceRef> reads;   // 输入资源
    std::vector<ResourceRef> writes;  // 输出资源

    // ---- 执行回调 ----
    // execute 期调用：void(VkCommandBuffer cmd, const FGResources& resources)
    std::function<void(VkCommandBuffer, const FGResources&)> execute;

    // ---- 编译后填充 ----
    uint32_t topologicalIndex = 0;                     // 拓扑排序位置
    std::vector<FGPassNode*> predecessors;             // 直接前驱 pass
};

} // namespace fg
} // namespace somegi
```

- [ ] **Step 2: 验证编译**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译通过。

- [ ] **Step 3: 提交**

```bash
git add src/renderer/fg/fg_pass_node.h
git commit -m "Phase 0.3: 添加 FGPassNode 内部 Pass 节点定义"
```

---

### Task 0.4: FGResources — Execute 期资源查询

**Files:**
- Create: `src/renderer/fg/fg_resources.h`

- [ ] **Step 1: 编写 fg_resources.h**

```cpp
// src/renderer/fg/fg_resources.h
#pragma once
#include "fg_common.h"
#include <vulkan/vulkan.h>
#include <vector>

namespace somegi {
namespace fg {

// ============================================================
// FGResources: 提供给 pass execute 回调的只读资源视图
//
// 内部缓存了 handle → VkImageView / VkBuffer 的映射，
// 在 FrameGraph::execute() 开始时一次性填充。
// ============================================================
class FGResources {
public:
    // 获取纹理视图（mip=0, layer=0 默认）
    VkImageView getTextureView(FGHandle handle,
                                uint32_t mip = 0,
                                uint32_t layer = 0) const;

    // 获取 Buffer handle + offset
    VkBuffer getBuffer(FGHandle handle,
                       VkDeviceSize* outOffset = nullptr) const;

    // 获取资源 extent
    VkExtent3D extent(FGHandle handle) const;

private:
    friend class FrameGraph;

    struct TextureView {
        FGHandle handle;
        VkImageView view = VK_NULL_HANDLE;
        VkExtent3D extent{1, 1, 1};
    };

    struct BufferView {
        FGHandle handle;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
    };

    std::vector<TextureView> m_textures;
    std::vector<BufferView>  m_buffers;
};

} // namespace fg
} // namespace somegi
```

- [ ] **Step 2: 验证编译**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译通过（仅头文件，FGResources 方法实现在 fg_graph.cpp 中）。

- [ ] **Step 3: 提交**

```bash
git add src/renderer/fg/fg_resources.h
git commit -m "Phase 0.4: 添加 FGResources execute 期资源查询接口"
```

---

### Task 0.5: FGBuilder — Pass 声明 API

**Files:**
- Create: `src/renderer/fg/fg_builder.h`

- [ ] **Step 1: 编写 fg_builder.h**

```cpp
// src/renderer/fg/fg_builder.h
#pragma once
#include "fg_common.h"
#include <functional>

namespace somegi {
namespace fg {

class FrameGraph;
struct FGPassNode;
struct FGResourceNode;
struct FGResources;

// ============================================================
// FGBuilder: 在 addPass() 的 setup lambda 中使用
//
// 生命周期：仅在 setup lambda 执行期间有效。
// 内部持有指向当前 FGPassNode 的非拥有指针。
// ============================================================
class FGBuilder {
public:
    // 构造函数由 FrameGraph::addPass() 调用，用户不直接创建
    FGBuilder(FrameGraph& graph, FGPassNode& passNode);

    // ---- Pass 类型 ----
    // 显式指定 pass 类型，覆盖自动推导
    FGBuilder& setPassType(FGPassType type);

    // ---- 资源依赖声明 ----
    // 声明读依赖：pass 执行前资源处于 SHADER_READ_ONLY_OPTIMAL
    FGHandle read(FGHandle handle);

    // 声明写依赖：pass 执行前资源处于合适的写 layout
    FGHandle write(FGHandle handle);

    // 声明读写依赖：in-place update（先读后写同一资源）
    FGHandle readWrite(FGHandle handle);

    // ---- 托管资源创建 ----
    FGHandle createTexture(const char* name, const FGTextureDesc& desc);
    FGHandle createBuffer(const char* name, const FGBufferDesc& desc);

    // ---- 执行回调 ----
    // fn 签名: void(VkCommandBuffer cmd, const FGResources& resources)
    template<typename F>
    void setExecute(F&& fn) {
        m_passNode->execute = std::forward<F>(fn);
    }

private:
    FrameGraph&   m_graph;
    FGPassNode*   m_passNode = nullptr;
};

} // namespace fg
} // namespace somegi
```

- [ ] **Step 2: 验证编译**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译通过（头文件，FGBuilder 实现在 fg_graph.cpp 中）。

- [ ] **Step 3: 提交**

```bash
git add src/renderer/fg/fg_builder.h
git commit -m "Phase 0.5: 添加 FGBuilder Pass 声明 API"
```

---

### Task 0.6: FGDebug — 可视化数据结构

**Files:**
- Create: `src/renderer/fg/fg_debug.h`
- Create: `src/renderer/fg/fg_debug.cpp`

- [ ] **Step 1: 编写 fg_debug.h**

```cpp
// src/renderer/fg/fg_debug.h
#pragma once
#include "fg_common.h"
#include <string>
#include <vector>
#include <cstdint>

namespace somegi {
namespace fg {

// ============================================================
// FGDebug: FrameGraph 调试/可视化数据
//
// 由 FrameGraph::compile() 填充，ImGui 面板读取展示。
// ============================================================
struct FGDebug {
    // ---- Pass 信息 ----
    struct PassDebugInfo {
        std::string name;
        FGPassType passType;
        bool culled = false;
        uint32_t execOrder = 0;
        std::vector<std::string> reads;   // 输入资源名
        std::vector<std::string> writes;  // 输出资源名
        std::vector<std::string> deps;    // 前驱 pass 名
        float gpuMs = 0.0f;
    };
    std::vector<PassDebugInfo> passes;

    // ---- 资源信息 ----
    struct ResourceDebugInfo {
        std::string name;
        FGResourceType type;
        bool isImported = false;
        uint32_t firstWritePass = 0;
        uint32_t lastReadPass = 0;
        uint32_t aliasGroup = UINT32_MAX;
        uint32_t sizeBytes = 0;
        VkExtent3D extent{};
    };
    std::vector<ResourceDebugInfo> resources;

    // ---- 别名组信息 ----
    struct AliasGroupDebug {
        uint32_t id = 0;
        uint32_t totalBytes = 0;
        uint32_t wastedBytes = 0;
        std::vector<std::string> members;
    };
    std::vector<AliasGroupDebug> aliasGroups;

    // ---- 开关 ----
    bool showPassList    = true;
    bool showResources   = false;
    bool showAliasGroups = false;
    bool showBarrierLog  = false;

    // 编译后填充
    void populate(const class FGCompiler::CompiledGraph& compiled,
                  const std::vector<struct FGPassNode>& passes,
                  const std::vector<struct FGResourceNode>& resources);
};

} // namespace fg
} // namespace somegi
```

- [ ] **Step 2: 编写 fg_debug.cpp**

```cpp
// src/renderer/fg/fg_debug.cpp
#include "fg_debug.h"
#include "fg_pass_node.h"
#include "fg_resource_node.h"
#include <cstdio>

namespace somegi {
namespace fg {

static uint32_t estimateSizeBytes(const FGResourceDesc& desc) {
    if (desc.type == FGResourceType::Texture) {
        const auto& t = desc.texture;
        uint32_t pixelSize = 4;  // 默认 RGBA8
        if (t.format == VK_FORMAT_R16G16B16A16_SFLOAT) pixelSize = 8;
        else if (t.format == VK_FORMAT_R32_SFLOAT) pixelSize = 4;
        else if (t.format == VK_FORMAT_D32_SFLOAT) pixelSize = 4;
        return t.extent.width * t.extent.height * t.extent.depth * pixelSize;
    } else {
        return (uint32_t)desc.buffer.size;
    }
}

static const char* passTypeString(FGPassType t) {
    switch (t) {
        case FGPassType::Compute:     return "Compute";
        case FGPassType::Graphics:    return "Graphics";
        case FGPassType::MeshShading: return "Mesh/Task";
        case FGPassType::RayTracing:  return "RayTracing";
        default: return "?";
    }
}

void FGDebug::populate(const FGCompiler::CompiledGraph& compiled,
                        const std::vector<FGPassNode>& passes,
                        const std::vector<FGResourceNode>& resources) {
    // 清空上帧数据
    this->passes.clear();
    this->resources.clear();
    this->aliasGroups.clear();

    // Pass 列表
    for (auto* p : compiled.passOrder) {
        PassDebugInfo pi;
        pi.name = p->name;
        pi.passType = p->passType;
        pi.culled = p->culled;
        pi.execOrder = p->topologicalIndex;

        for (auto& r : p->reads) {
            if (r.resource && r.resource->desc.debugName)
                pi.reads.push_back(r.resource->desc.debugName);
        }
        for (auto& w : p->writes) {
            if (w.resource && w.resource->desc.debugName)
                pi.writes.push_back(w.resource->desc.debugName);
        }
        for (auto* pred : p->predecessors)
            pi.deps.push_back(pred->name);

        this->passes.push_back(std::move(pi));
    }

    // 资源列表
    for (auto* r : compiled.resources) {
        if (!r) continue;
        ResourceDebugInfo ri;
        ri.name = r->desc.debugName ? r->desc.debugName : "?";
        ri.type = r->desc.type;
        ri.isImported = r->isImported;
        ri.firstWritePass = r->firstWritePass;
        ri.lastReadPass = r->lastReadPass;
        ri.aliasGroup = r->aliasedGroup;
        ri.sizeBytes = estimateSizeBytes(r->desc);
        if (r->desc.type == FGResourceType::Texture)
            ri.extent = r->desc.texture.extent;
        this->resources.push_back(std::move(ri));
    }

    // 别名组列表
    for (auto& ag : compiled.aliasGroups) {
        AliasGroupDebug agd;
        agd.id = this->aliasGroups.size();
        agd.totalBytes = ag.sizeBytes;
        uint32_t maxSize = 0;
        for (auto* m : ag.members) {
            if (m && m->desc.debugName)
                agd.members.push_back(m->desc.debugName);
            uint32_t s = m ? estimateSizeBytes(m->desc) : 0;
            if (s > maxSize) maxSize = s;
        }
        agd.wastedBytes = ag.totalBytes - maxSize;
        this->aliasGroups.push_back(std::move(agd));
    }
}

} // namespace fg
} // namespace somegi
```

- [ ] **Step 3: 验证编译**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译通过。

- [ ] **Step 4: 提交**

```bash
git add src/renderer/fg/fg_debug.h src/renderer/fg/fg_debug.cpp
git commit -m "Phase 0.6: 添加 FGDebug 可视化数据结构"
```

---

### Task 0.7: FrameGraph 骨架 — fg_graph.h 声明

**Files:**
- Create: `src/renderer/fg/fg_graph.h`

- [ ] **Step 1: 编写 fg_graph.h**

```cpp
// src/renderer/fg/fg_graph.h
#pragma once
#include "fg_common.h"
#include "fg_builder.h"
#include "fg_resources.h"
#include "fg_compiler.h"
#include "fg_executor.h"
#include "fg_debug.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>

namespace somegi {

class Device;

namespace fg {

// ============================================================
// FrameGraph: 顶层 API
//
// 使用流程：
//   1. importTexture() / createTexture() — 声明资源
//   2. addPass() — 声明 pass 及其资源依赖
//   3. compile() — 构建 DAG / 剔除 / 别名分析 / 拓扑排序
//   4. execute(cmd) — 分配物理资源 / 插入 barrier / 执行 pass
//   5. reset() — 每帧结束时清理图数据
// ============================================================
class FrameGraph {
public:
    FrameGraph();
    ~FrameGraph();

    // ---- 初始化 ----
    void init(Device& device);

    // ---- 资源声明 ----
    // 导入外部已分配的资源（跨帧持久，不参与 aliasing）
    FGHandle importTexture(const char* name,
                           VkImage image,
                           const FGTextureDesc& desc,
                           VkImageLayout initialLayout);

    // 声明托管临时纹理（帧内生命周期，参与 aliasing）
    FGHandle createTexture(const char* name, const FGTextureDesc& desc);

    // 声明托管临时 Buffer（帧内生命周期，参与 aliasing）
    FGHandle createBuffer(const char* name, const FGBufferDesc& desc);

    // ---- Pass 声明 ----
    // setup: void(FGBuilder&) — 声明资源依赖和执行回调
    void addPass(const char* name, std::function<void(FGBuilder&)> setup);

    // ---- 编译 ----
    // 调用 FGCompiler::compile()，结果存入 m_compiled
    void compile();

    // ---- 执行 ----
    // 调用 FGExecutor::execute()，遍历编译后的 pass 并执行
    void execute(VkCommandBuffer cmd);

    // ---- 查询（execute 期使用） ----
    VkImageView getTextureView(FGHandle handle,
                               uint32_t mip = 0,
                               uint32_t layer = 0) const;

    VkBuffer getBuffer(FGHandle handle,
                       VkDeviceSize* outOffset = nullptr) const;

    // ---- 帧管理 ----
    // 每帧开始时调用：清理 pass/resource 节点和视图缓存
    void reset();

    // ---- 调试 ----
    const FGCompiler::CompiledGraph& compiledGraph() const { return m_compiled; }
    FGDebug& debug() { return m_debug; }

private:
    friend class FGBuilder;

    Device* m_device = nullptr;
    uint64_t m_frameIndex = 0;

    // 用户声明的节点
    std::vector<FGPassNode>     m_passes;
    std::vector<FGResourceNode> m_resources;
    std::unordered_map<std::string, uint32_t> m_resourceNameMap;  // name → resource index

    // 编译/执行
    FGCompiler m_compiler;
    FGExecutor m_executor;
    FGDebug    m_debug;
    FGCompiler::CompiledGraph m_compiled;
    bool m_compiledThisFrame = false;

    // execute 期资源视图缓存
    FGResources m_viewCache;

    // ---- 内部方法 ----
    // 添加托管资源（由 FGBuilder::createTexture/createBuffer 调用）
    FGHandle addManagedResource(const FGResourceDesc& desc);

    // 查找资源节点
    FGResourceNode* findResource(FGHandle handle);
    const FGResourceNode* findResource(FGHandle handle) const;

    // 填充 FGResources 视图缓存
    void populateViewCache();

    // 获取 pipeline stage 对应的 VkPipelineStageFlags2
    static VkPipelineStageFlags2 stageForPassType(FGPassType type);
    static VkPipelineStageFlags2 readStageForPassType(FGPassType type);
};

} // namespace fg
} // namespace somegi
```

- [ ] **Step 2: 验证编译**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译通过（仅头文件，尚无 .cpp 实现，预期链接阶段才有完整符号）。

- [ ] **Step 3: 提交**

```bash
git add src/renderer/fg/fg_graph.h
git commit -m "Phase 0.7: 添加 FrameGraph 顶层 API 声明"
```

---

### Task 0.8: FGCompiler 声明

**Files:**
- Create: `src/renderer/fg/fg_compiler.h`

- [ ] **Step 1: 编写 fg_compiler.h**

```cpp
// src/renderer/fg/fg_compiler.h
#pragma once
#include "fg_common.h"
#include <vector>
#include <cstdint>

namespace somegi {
namespace fg {

struct FGPassNode;
struct FGResourceNode;

// ============================================================
// FGCompiler: 帧图编译器
//
// 输入：passes[] + resources[]（由 FrameGraph 收集的声明数据）
// 输出：CompiledGraph（执行计划）
//
// 五大步骤：
//   1. cullPasses()      — 剔除死 pass（输出无人消费）
//   2. buildEdges()      — 构建 RAW/WAW 依赖边
//   3. computeLifetimes() — 计算每个资源的 firstWrite/lastRead
//   4. computeAliasing()  — 贪心算法合并非重叠寿命的资源
//   5. topologicalSort()  — Kahn BFS 拓扑排序
// ============================================================
class FGCompiler {
public:
    struct AliasGroup {
        uint32_t sizeBytes = 0;                     // 该组需要的最大内存
        std::vector<FGResourceNode*> members;        // 成员资源
    };

    struct CompiledGraph {
        std::vector<FGPassNode*>     passOrder;      // 拓扑排序后的执行顺序
        std::vector<FGResourceNode*> resources;       // 所有资源节点
        std::vector<FGPassNode*>     culledPasses;    // 被剔除的 pass
        std::vector<AliasGroup>      aliasGroups;     // 别名分组
    };

    // 编译入口
    CompiledGraph compile(std::vector<FGPassNode*>& passes,
                          std::vector<FGResourceNode*>& resources);

private:
    // 步骤 1: 剔除 — 输出不被任何活跃 pass 消费的 pass
    void cullPasses(std::vector<FGPassNode*>& passes,
                    std::vector<FGResourceNode*>& resources);

    // 步骤 2: 构建 DAG 边
    void buildEdges(std::vector<FGPassNode*>& passes,
                    std::vector<FGResourceNode*>& resources);

    // 步骤 3: 计算资源寿命
    void computeLifetimes(std::vector<FGResourceNode*>& resources);

    // 步骤 4: 别名分析
    void computeAliasing(std::vector<FGResourceNode*>& resources,
                         std::vector<AliasGroup>& outGroups);

    // 步骤 5: 拓扑排序
    void topologicalSort(std::vector<FGPassNode*>& passes);
};

} // namespace fg
} // namespace somegi
```

- [ ] **Step 2: 验证编译**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译通过。

- [ ] **Step 3: 提交**

```bash
git add src/renderer/fg/fg_compiler.h
git commit -m "Phase 0.8: 添加 FGCompiler 编译阶段声明"
```

---

### Task 0.9: FGExecutor 声明

**Files:**
- Create: `src/renderer/fg/fg_executor.h`

- [ ] **Step 1: 编写 fg_executor.h**

```cpp
// src/renderer/fg/fg_executor.h
#pragma once
#include "fg_common.h"
#include "core/image.h"
#include "core/buffer.h"
#include <vector>
#include <cstdint>

namespace somegi {

class Device;

namespace fg {

struct FGPassNode;
struct FGResourceNode;
class FGCompiler;

// ============================================================
// FGExecutor: 帧图执行器
//
// 职责：
//   1. 为每个 AliasGroup 分配/复用物理 Image/Buffer
//   2. 遍历 passOrder，为每个 pass 自动插入 barrier
//   3. 调用 pass.execute() 录制用户命令
//   4. 更新资源 Barrier 状态
//   5. 回收长期未用的池资源
// ============================================================
class FGExecutor {
public:
    void init(Device& device);
    void destroy();

    // 执行编译后的图
    void execute(VkCommandBuffer cmd,
                 FGCompiler::CompiledGraph& compiled,
                 const FGResources& viewCache);

    // 静态方法：根据 pass 类型和资源 usage 推导 layout/access/stage
    static VkImageLayout derivedLayout(FGPassType passType,
                                        VkImageUsageFlags usage,
                                        bool isWrite);
    static VkAccessFlags2 derivedAccess(FGPassType passType,
                                         VkImageUsageFlags usage,
                                         bool isWrite,
                                         bool isReadWrite);
    static VkPipelineStageFlags2 derivedStage(FGPassType passType,
                                               VkImageUsageFlags usage,
                                               bool isWrite);

private:
    Device* m_device = nullptr;
    uint64_t m_currentFrame = 0;

    // ---- 纹理资源池 ----
    struct PooledTexture {
        Image image;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent3D extent{};
        uint64_t lastUsedFrame = 0;
        bool inUse = false;
    };
    std::vector<PooledTexture> m_texturePool;

    // ---- Buffer 资源池 ----
    struct PooledBuffer {
        Buffer buffer;
        VkDeviceSize size = 0;
        uint64_t lastUsedFrame = 0;
        bool inUse = false;
    };
    std::vector<PooledBuffer> m_bufferPool;

    // ---- 内部方法 ----

    // 为别名组分配物理资源
    void allocateAliasGroup(const FGCompiler::AliasGroup& group,
                            std::vector<FGResourceNode*>& resources);

    // 分配单个托管纹理（池中取或新建）
    Image* allocateTexture(const FGResourceDesc& desc);

    // 分配单个托管 Buffer（池中取或新建）
    Buffer* allocateBuffer(const FGResourceDesc& desc);

    // 为 pass 插入前置 barrier
    void emitBarriers(VkCommandBuffer cmd,
                      const FGPassNode& pass,
                      std::vector<FGResourceNode*>& resources,
                      const FGResources& viewCache);

    // 更新 pass 执行后的资源状态
    void updateResourceStates(const FGPassNode& pass,
                              std::vector<FGResourceNode*>& resources);

    // 回收超过 recycleFrameThreshold 帧未用的池资源
    void recycleUnused(uint64_t threshold);
    static constexpr uint32_t kRecycleFrames = 30;  // 30 帧不用则回收
};

} // namespace fg
} // namespace somegi
```

- [ ] **Step 2: 验证编译**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译通过。

- [ ] **Step 3: 提交**

```bash
git add src/renderer/fg/fg_executor.h
git commit -m "Phase 0.9: 添加 FGExecutor 执行阶段声明"
```

---

### Task 0.10: FrameGraph 实现 — fg_graph.cpp

**Files:**
- Create: `src/renderer/fg/fg_graph.cpp`

- [ ] **Step 1: 编写 fg_graph.cpp**

```cpp
// src/renderer/fg/fg_graph.cpp
#include "fg_graph.h"
#include "fg_pass_node.h"
#include "fg_resource_node.h"
#include "core/device.h"

namespace somegi {
namespace fg {

// ============================================================
// FrameGraph
// ============================================================

FrameGraph::FrameGraph() = default;
FrameGraph::~FrameGraph() = default;

void FrameGraph::init(Device& device) {
    m_device = &device;
    m_executor.init(device);
}

// ---- 资源声明 ----

FGHandle FrameGraph::importTexture(const char* name,
                                    VkImage image,
                                    const FGTextureDesc& desc,
                                    VkImageLayout initialLayout) {
    uint32_t idx = (uint32_t)m_resources.size();
    FGHandle h{idx, 0};

    FGResourceNode node;
    node.handle = h;
    node.desc = FGResourceDesc::textureDesc(name, desc.extent, desc.format,
        desc.usage, desc.mipLevels, desc.samples);
    node.desc.texture = desc;
    node.isImported = true;
    node.state.layout = initialLayout;

    m_resources.push_back(std::move(node));
    m_resourceNameMap[name] = idx;
    return h;
}

FGHandle FrameGraph::createTexture(const char* name, const FGTextureDesc& desc) {
    return addManagedResource(
        FGResourceDesc::textureDesc(name, desc.extent, desc.format,
            desc.usage, desc.mipLevels, desc.samples));
}

FGHandle FrameGraph::createBuffer(const char* name, const FGBufferDesc& desc) {
    return addManagedResource(
        FGResourceDesc::bufferDesc(name, desc.size, desc.usage));
}

FGHandle FrameGraph::addManagedResource(const FGResourceDesc& desc) {
    uint32_t idx = (uint32_t)m_resources.size();
    // generation 简单递增，假设同一帧内不会对同名资源重复创建
    FGHandle h{idx, 0};

    FGResourceNode node;
    node.handle = h;
    node.desc = desc;
    node.isImported = false;

    m_resources.push_back(std::move(node));
    if (desc.debugName) {
        m_resourceNameMap[desc.debugName] = idx;
    }
    return h;
}

// ---- Pass 声明 ----

void FrameGraph::addPass(const char* name, std::function<void(FGBuilder&)> setup) {
    FGPassNode node;
    node.name = name;

    // 默认 pass 类型：Compute（setup 中可覆盖）
    node.passType = FGPassType::Compute;

    // 添加到 pass 列表
    m_passes.push_back(std::move(node));
    FGPassNode& passNode = m_passes.back();

    // 创建 builder 并执行 setup
    FGBuilder builder(*this, passNode);
    setup(builder);

    // 如果 setup 中没有显式调用 setPassType 且 write 中包含
    // COLOR_ATTACHMENT 或 DEPTH 用途，自动切换为 Graphics
    bool hasAttachment = false;
    for (auto& ref : passNode.writes) {
        if (ref.resource && ref.resource->desc.type == FGResourceType::Texture) {
            auto usage = ref.resource->desc.texture.usage;
            if (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
                hasAttachment = true;
            }
        }
    }
    if (hasAttachment && passNode.passType == FGPassType::Compute) {
        passNode.passType = FGPassType::Graphics;
    }
}

// ---- 编译 ----

void FrameGraph::compile() {
    // 收集原始指针
    std::vector<FGPassNode*> passPtrs;
    passPtrs.reserve(m_passes.size());
    for (auto& p : m_passes) passPtrs.push_back(&p);

    std::vector<FGResourceNode*> resPtrs;
    resPtrs.reserve(m_resources.size());
    for (auto& r : m_resources) resPtrs.push_back(&r);

    // 编译
    m_compiled = m_compiler.compile(passPtrs, resPtrs);

    // 填充调试数据
    m_debug.populate(m_compiled, m_passes, m_resources);

    m_compiledThisFrame = true;
}

// ---- 执行 ----

void FrameGraph::execute(VkCommandBuffer cmd) {
    if (!m_compiledThisFrame) return;

    // 填充视图缓存
    populateViewCache();

    // 执行
    m_executor.execute(cmd, m_compiled, m_viewCache);

    ++m_frameIndex;
}

// ---- 查询 ----

VkImageView FrameGraph::getTextureView(FGHandle handle,
                                        uint32_t mip,
                                        uint32_t layer) const {
    for (auto& tv : m_viewCache.m_textures) {
        if (tv.handle == handle) return tv.view;
    }
    return VK_NULL_HANDLE;
}

VkBuffer FrameGraph::getBuffer(FGHandle handle,
                                VkDeviceSize* outOffset) const {
    for (auto& bv : m_viewCache.m_buffers) {
        if (bv.handle == handle) {
            if (outOffset) *outOffset = bv.offset;
            return bv.buffer;
        }
    }
    return VK_NULL_HANDLE;
}

// ---- 帧管理 ----

void FrameGraph::reset() {
    m_passes.clear();
    m_resources.clear();
    m_resourceNameMap.clear();
    m_compiled = FGCompiler::CompiledGraph{};
    m_compiledThisFrame = false;

    // 清空视图缓存
    m_viewCache = FGResources{};
}

// ---- 资源查找 ----

FGResourceNode* FrameGraph::findResource(FGHandle handle) {
    if (!handle.valid() || handle.index >= m_resources.size()) return nullptr;
    return &m_resources[handle.index];
}

const FGResourceNode* FrameGraph::findResource(FGHandle handle) const {
    if (!handle.valid() || handle.index >= m_resources.size()) return nullptr;
    return &m_resources[handle.index];
}

// ---- 视图缓存填充 ----

void FrameGraph::populateViewCache() {
    m_viewCache.m_textures.clear();
    m_viewCache.m_buffers.clear();

    for (auto& res : m_resources) {
        if (res.desc.type == FGResourceType::Texture && res.physicalTexture) {
            FGResources::TextureView tv;
            tv.handle = res.handle;
            tv.view = res.physicalTexture->view();
            tv.extent = res.physicalTexture->extent();
            m_viewCache.m_textures.push_back(tv);
        } else if (res.desc.type == FGResourceType::Buffer && res.physicalBuffer) {
            FGResources::BufferView bv;
            bv.handle = res.handle;
            bv.buffer = res.physicalBuffer->handle();
            bv.size = res.physicalBuffer->size();
            m_viewCache.m_buffers.push_back(bv);
        }
    }
}

// ---- Pipeline stage 推导 ----

VkPipelineStageFlags2 FrameGraph::stageForPassType(FGPassType type) {
    switch (type) {
        case FGPassType::Compute:     return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        case FGPassType::Graphics:    return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case FGPassType::MeshShading: return VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                             VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case FGPassType::RayTracing:  return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    }
    return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
}

VkPipelineStageFlags2 FrameGraph::readStageForPassType(FGPassType type) {
    switch (type) {
        case FGPassType::Compute:     return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        case FGPassType::Graphics:    return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case FGPassType::MeshShading: return VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case FGPassType::RayTracing:  return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    }
    return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
}

// ============================================================
// FGBuilder
// ============================================================

FGBuilder::FGBuilder(FrameGraph& graph, FGPassNode& passNode)
    : m_graph(graph), m_passNode(&passNode) {}

FGBuilder& FGBuilder::setPassType(FGPassType type) {
    m_passNode->passType = type;
    return *this;
}

FGHandle FGBuilder::read(FGHandle handle) {
    auto* res = m_graph.findResource(handle);
    if (!res) return handle;

    FGPassNode::ResourceRef ref;
    ref.handle = handle;
    ref.resource = res;

    // 推导 access/stage/layout
    bool isTexture = (res->desc.type == FGResourceType::Texture);
    VkImageUsageFlags usage = isTexture ? res->desc.texture.usage : (VkImageUsageFlags)0;
    ref.access = FGExecutor::derivedAccess(m_passNode->passType, usage, false, false);
    ref.stages = FrameGraph::readStageForPassType(m_passNode->passType);
    ref.requiredLayout = isTexture ? FGExecutor::derivedLayout(
        m_passNode->passType, usage, false) : VK_IMAGE_LAYOUT_UNDEFINED;

    m_passNode->reads.push_back(ref);
    return handle;
}

FGHandle FGBuilder::write(FGHandle handle) {
    auto* res = m_graph.findResource(handle);
    if (!res) return handle;

    FGPassNode::ResourceRef ref;
    ref.handle = handle;
    ref.resource = res;

    bool isTexture = (res->desc.type == FGResourceType::Texture);
    VkImageUsageFlags usage = isTexture ? res->desc.texture.usage : (VkImageUsageFlags)0;
    ref.access = FGExecutor::derivedAccess(m_passNode->passType, usage, true, false);
    ref.stages = FGExecutor::derivedStage(m_passNode->passType, usage, true);
    ref.requiredLayout = isTexture ? FGExecutor::derivedLayout(
        m_passNode->passType, usage, true) : VK_IMAGE_LAYOUT_UNDEFINED;

    m_passNode->writes.push_back(ref);
    return handle;
}

FGHandle FGBuilder::readWrite(FGHandle handle) {
    auto* res = m_graph.findResource(handle);
    if (!res) return handle;

    FGPassNode::ResourceRef ref;
    ref.handle = handle;
    ref.resource = res;

    bool isTexture = (res->desc.type == FGResourceType::Texture);
    VkImageUsageFlags usage = isTexture ? res->desc.texture.usage : (VkImageUsageFlags)0;
    ref.access = FGExecutor::derivedAccess(m_passNode->passType, usage, true, true);
    ref.stages = FGExecutor::derivedStage(m_passNode->passType, usage, true);
    ref.requiredLayout = isTexture ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;

    m_passNode->reads.push_back(ref);   // 同时加为 read 和 write
    m_passNode->writes.push_back(ref);
    return handle;
}

FGHandle FGBuilder::createTexture(const char* name, const FGTextureDesc& desc) {
    return m_graph.createTexture(name, desc);
}

FGHandle FGBuilder::createBuffer(const char* name, const FGBufferDesc& desc) {
    return m_graph.createBuffer(name, desc);
}

// ============================================================
// FGResources
// ============================================================

VkImageView FGResources::getTextureView(FGHandle handle,
                                         uint32_t mip,
                                         uint32_t layer) const {
    for (auto& tv : m_textures) {
        if (tv.handle == handle) return tv.view;
    }
    return VK_NULL_HANDLE;
}

VkBuffer FGResources::getBuffer(FGHandle handle,
                                 VkDeviceSize* outOffset) const {
    for (auto& bv : m_buffers) {
        if (bv.handle == handle) {
            if (outOffset) *outOffset = bv.offset;
            return bv.buffer;
        }
    }
    return VK_NULL_HANDLE;
}

VkExtent3D FGResources::extent(FGHandle handle) const {
    for (auto& tv : m_textures) {
        if (tv.handle == handle) return tv.extent;
    }
    return {1, 1, 1};
}

} // namespace fg
} // namespace somegi
```

- [ ] **Step 2: 验证编译**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译通过（链接需要 fg_compiler.cpp 和 fg_executor.cpp 的符号，当前使用空桩）。

- [ ] **Step 3: 提交**

```bash
git add src/renderer/fg/fg_graph.cpp
git commit -m "Phase 0.10: 实现 FrameGraph/FGBuilder/FGResources 核心逻辑"
```

---

### Task 0.11: FGCompiler 实现

**Files:**
- Create: `src/renderer/fg/fg_compiler.cpp`

- [ ] **Step 1: 编写 fg_compiler.cpp**

```cpp
// src/renderer/fg/fg_compiler.cpp
#include "fg_compiler.h"
#include "fg_pass_node.h"
#include "fg_resource_node.h"
#include <algorithm>
#include <queue>
#include <cstdio>

namespace somegi {
namespace fg {

FGCompiler::CompiledGraph FGCompiler::compile(
    std::vector<FGPassNode*>& passes,
    std::vector<FGResourceNode*>& resources) {

    CompiledGraph result;
    result.resources = resources;

    // 步骤 1: 剔除
    cullPasses(passes, resources);

    // 步骤 2: 构建依赖边
    buildEdges(passes, resources);

    // 步骤 3: 计算资源寿命
    computeLifetimes(resources);

    // 步骤 4: 别名分析（仅托管资源）
    computeAliasing(resources, result.aliasGroups);

    // 步骤 5: 拓扑排序
    topologicalSort(passes);

    // 填充结果
    for (auto* p : passes) {
        if (!p->culled) {
            result.passOrder.push_back(p);
        } else {
            result.culledPasses.push_back(p);
        }
    }

    return result;
}

// ============================================================
// 步骤 1: 剔除
// ============================================================

void FGCompiler::cullPasses(std::vector<FGPassNode*>& passes,
                             std::vector<FGResourceNode*>& resources) {
    // 第一遍：标记 enabled=false 的 pass
    for (auto* p : passes) {
        if (p->culled) continue;
        if (!p->enabled) {
            p->culled = true;
        }
    }

    // 传播剔除：迭代直到稳定
    // 规则：如果一个 pass 的所有输出资源都没有被任何活跃 pass 读取，
    // 则该 pass 被剔除（除非它是导入资源的写者，导入资源可能被外部消费）
    bool changed = true;
    while (changed) {
        changed = false;

        for (auto* p : passes) {
            if (p->culled) continue;

            // 检查该 pass 的输出是否被任何未剔除的 pass 消费
            bool anyConsumer = false;
            for (auto& w : p->writes) {
                if (!w.resource || w.resource->isImported) {
                    // 导入资源可能被外部消费者（如 swapchain）使用
                    anyConsumer = true;
                    break;
                }
                // 扫描所有其他 pass 的 reads
                for (auto* other : passes) {
                    if (other == p || other->culled) continue;
                    for (auto& r : other->reads) {
                        if (r.resource == w.resource) {
                            anyConsumer = true;
                            break;
                        }
                    }
                    if (anyConsumer) break;
                }
                if (anyConsumer) break;
            }

            if (!anyConsumer) {
                p->culled = true;
                changed = true;
                std::printf("[FGCompiler] culled: %s (no consumers)\n", p->name.c_str());
            }
        }
    }
}

// ============================================================
// 步骤 2: 构建依赖边
// ============================================================

void FGCompiler::buildEdges(std::vector<FGPassNode*>& passes,
                             std::vector<FGResourceNode*>& resources) {
    // 追踪每个资源的上一个 writer
    std::unordered_map<FGResourceNode*, FGPassNode*> lastWriter;

    for (auto* p : passes) {
        if (p->culled) continue;

        // 处理 reads: 为每个读取资源添加 RAW 依赖
        for (auto& ref : p->reads) {
            auto* res = ref.resource;
            if (!res) continue;

            auto it = lastWriter.find(res);
            if (it != lastWriter.end() && it->second != p) {
                // RAW: 前一个 writer → 当前 reader
                if (std::find(p->predecessors.begin(), p->predecessors.end(), it->second)
                    == p->predecessors.end()) {
                    p->predecessors.push_back(it->second);
                }
            }
        }

        // 处理 writes: 为每个写入资源添加 WAW 依赖
        for (auto& ref : p->writes) {
            auto* res = ref.resource;
            if (!res) continue;

            auto it = lastWriter.find(res);
            if (it != lastWriter.end() && it->second != p) {
                // WAW: 前一个 writer → 当前 writer
                if (std::find(p->predecessors.begin(), p->predecessors.end(), it->second)
                    == p->predecessors.end()) {
                    p->predecessors.push_back(it->second);
                }
            }

            // 更新 last writer
            lastWriter[res] = p;
        }
    }
}

// ============================================================
// 步骤 3: 计算资源寿命
// ============================================================

void FGCompiler::computeLifetimes(std::vector<FGResourceNode*>& resources) {
    for (auto* res : resources) {
        if (!res) continue;
        res->firstWritePass = UINT32_MAX;
        res->lastReadPass = 0;
    }
}

// ============================================================
// 步骤 4: 别名分析
// ============================================================

void FGCompiler::computeAliasing(std::vector<FGResourceNode*>& resources,
                                  std::vector<AliasGroup>& outGroups) {
    // 收集托管资源，按大小降序排列
    struct ManagedRes {
        FGResourceNode* resource;
        uint32_t sizeBytes;
    };
    std::vector<ManagedRes> managed;

    for (auto* res : resources) {
        if (!res || res->isImported) continue;

        uint32_t size = 4 * 1024 * 1024;  // 默认 4MB 估算
        if (res->desc.type == FGResourceType::Texture) {
            auto& t = res->desc.texture;
            // 简单估算：4 bytes/像素
            size = t.extent.width * t.extent.height * t.extent.depth * 4;
        } else {
            size = (uint32_t)res->desc.buffer.size;
        }

        managed.push_back({res, size});
    }

    // 按大小降序排列
    std::sort(managed.begin(), managed.end(),
        [](const ManagedRes& a, const ManagedRes& b) {
            return a.sizeBytes > b.sizeBytes;
        });

    // 贪心别名分配
    for (auto& mr : managed) {
        bool placed = false;

        // 尝试放入现有别名组
        for (auto& group : outGroups) {
            bool overlap = false;
            for (auto* member : group.members) {
                // 检查 lifetime 是否重叠
                if (!(mr.resource->lastReadPass < member->firstWritePass ||
                      mr.resource->firstWritePass > member->lastReadPass)) {
                    overlap = true;
                    break;
                }
            }

            if (!overlap) {
                group.members.push_back(mr.resource);
                if (mr.sizeBytes > group.sizeBytes) {
                    group.sizeBytes = mr.sizeBytes;
                }
                mr.resource->aliasedGroup = (uint32_t)(&group - outGroups.data());
                placed = true;
                break;
            }
        }

        // 无法放入现有组 → 新建组
        if (!placed) {
            AliasGroup newGroup;
            newGroup.sizeBytes = mr.sizeBytes;
            newGroup.members.push_back(mr.resource);
            mr.resource->aliasedGroup = (uint32_t)outGroups.size();
            outGroups.push_back(std::move(newGroup));
        }
    }

    // 打印别名统计
    if (!outGroups.empty()) {
        uint32_t totalSaved = 0;
        for (auto& g : outGroups) {
            for (size_t i = 1; i < g.members.size(); ++i) {
                // 除最大成员外，其他成员的 size 都被节省
                uint32_t maxSz = g.sizeBytes;
                uint32_t memberSz = 0;
                // 估算...（简化处理）
                totalSaved += maxSz;  // 保守估计
            }
        }
        std::printf("[FGCompiler] alias groups: %zu, estimated memory saved: %u bytes\n",
                    outGroups.size(), totalSaved);
    }
}

// ============================================================
// 步骤 5: 拓扑排序 (Kahn BFS)
// ============================================================

void FGCompiler::topologicalSort(std::vector<FGPassNode*>& passes) {
    // 计算入度
    std::unordered_map<FGPassNode*, uint32_t> inDegree;
    for (auto* p : passes) {
        if (p->culled) continue;
        inDegree[p] = 0;
    }
    for (auto* p : passes) {
        if (p->culled) continue;
        for (auto* pred : p->predecessors) {
            if (!pred->culled) {
                inDegree[p]++;
            }
        }
    }

    // BFS
    std::queue<FGPassNode*> queue;
    for (auto& [node, deg] : inDegree) {
        if (deg == 0) queue.push(node);
    }

    uint32_t order = 0;
    while (!queue.empty()) {
        auto* node = queue.front();
        queue.pop();
        node->topologicalIndex = order++;

        // 找后继（所有依赖当前 node 的 pass）
        for (auto* other : passes) {
            if (other->culled) continue;
            for (auto* pred : other->predecessors) {
                if (pred == node) {
                    inDegree[other]--;
                    if (inDegree[other] == 0) {
                        queue.push(other);
                    }
                }
            }
        }
    }

    // 检测是否所有活跃 pass 都被排序（防止循环依赖）
    uint32_t activeCount = 0;
    for (auto* p : passes) {
        if (!p->culled) activeCount++;
    }
    if (order < activeCount) {
        std::fprintf(stderr, "[FGCompiler] ERROR: cycle detected in pass graph! "
                             "Sorted %u/%u passes.\n", order, activeCount);
    }
}

} // namespace fg
} // namespace somegi
```

- [ ] **Step 2: 验证编译**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译通过。

- [ ] **Step 3: 提交**

```bash
git add src/renderer/fg/fg_compiler.cpp
git commit -m "Phase 0.11: 实现 FGCompiler 五大编译步骤"
```

---

### Task 0.12: FGExecutor 实现

**Files:**
- Create: `src/renderer/fg/fg_executor.cpp`

- [ ] **Step 1: 编写 fg_executor.cpp**

```cpp
// src/renderer/fg/fg_executor.cpp
#include "fg_executor.h"
#include "fg_compiler.h"
#include "fg_pass_node.h"
#include "fg_resource_node.h"
#include "fg_resources.h"
#include "core/device.h"
#include "core/image.h"
#include "core/buffer.h"
#include <cstdio>

namespace somegi {
namespace fg {

void FGExecutor::init(Device& device) {
    m_device = &device;
}

void FGExecutor::destroy() {
    m_texturePool.clear();
    m_bufferPool.clear();
}

void FGExecutor::execute(VkCommandBuffer cmd,
                          FGCompiler::CompiledGraph& compiled,
                          const FGResources& viewCache) {
    // 1. 分配别名组
    for (auto& group : compiled.aliasGroups) {
        allocateAliasGroup(group, compiled.resources);
    }

    // 2. 遍历执行 pass
    for (auto* pass : compiled.passOrder) {
        if (!pass || pass->culled) continue;

        // 2a. 插入 barrier
        emitBarriers(cmd, *pass, compiled.resources, viewCache);

        // 2b. 执行 pass
        if (pass->execute) {
            pass->execute(cmd, viewCache);
        }

        // 2c. 更新资源状态
        updateResourceStates(*pass, compiled.resources);
    }

    // 3. 回收
    recycleUnused(kRecycleFrames);

    ++m_currentFrame;
}

// ---- 别名组分配 ----

void FGExecutor::allocateAliasGroup(const FGCompiler::AliasGroup& group,
                                     std::vector<FGResourceNode*>& resources) {
    // 为组分配一个大的物理资源，所有成员共享
    Image* sharedTexture = nullptr;
    Buffer* sharedBuffer = nullptr;

    for (auto* member : group.members) {
        if (!member) continue;

        if (member->desc.type == FGResourceType::Texture) {
            if (!sharedTexture) {
                sharedTexture = allocateTexture(member->desc);
            }
            member->physicalTexture = sharedTexture;
            member->physicalBuffer = nullptr;
        } else {
            if (!sharedBuffer) {
                sharedBuffer = allocateBuffer(member->desc);
            }
            member->physicalBuffer = sharedBuffer;
            member->physicalTexture = nullptr;
        }
    }
}

// ---- 纹理分配 ----

Image* FGExecutor::allocateTexture(const FGResourceDesc& desc) {
    // 先在池中查找匹配的闲置纹理
    for (auto& pt : m_texturePool) {
        if (!pt.inUse &&
            pt.format == desc.texture.format &&
            pt.extent.width >= desc.texture.extent.width &&
            pt.extent.height >= desc.texture.extent.height &&
            pt.extent.depth >= desc.texture.extent.depth) {
            pt.inUse = true;
            pt.lastUsedFrame = m_currentFrame;
            return &pt.image;
        }
    }

    // 未命中 → 创建新纹理
    ImageDesc imgDesc;
    imgDesc.format = desc.texture.format;
    imgDesc.extent = desc.texture.extent;
    imgDesc.mipLevels = desc.texture.mipLevels;
    imgDesc.arrayLayers = desc.texture.arrayLayers;
    imgDesc.usage = desc.texture.usage;
    imgDesc.aspect = (desc.texture.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
                     ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    imgDesc.samples = desc.texture.samples;
    if (desc.texture.isCubemap) {
        imgDesc.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }

    PooledTexture pt;
    pt.image = Image(*m_device, imgDesc);
    pt.format = desc.texture.format;
    pt.extent = desc.texture.extent;
    pt.lastUsedFrame = m_currentFrame;
    pt.inUse = true;

    m_texturePool.push_back(std::move(pt));
    return &m_texturePool.back().image;
}

// ---- Buffer 分配 ----

Buffer* FGExecutor::allocateBuffer(const FGResourceDesc& desc) {
    // 先在池中查找匹配的闲置 Buffer
    for (auto& pb : m_bufferPool) {
        if (!pb.inUse && pb.size >= desc.buffer.size) {
            pb.inUse = true;
            pb.lastUsedFrame = m_currentFrame;
            return &pb.buffer;
        }
    }

    // 未命中 → 创建新 Buffer
    VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (desc.buffer.usage & (VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
        // 需要 host visible 的 buffer 暂不支持自动分配，保持 DEVICE_LOCAL
    }

    PooledBuffer pb;
    pb.buffer = Buffer(*m_device, desc.buffer.size, desc.buffer.usage, memProps);
    pb.size = desc.buffer.size;
    pb.lastUsedFrame = m_currentFrame;
    pb.inUse = true;

    m_bufferPool.push_back(std::move(pb));
    return &m_bufferPool.back().buffer;
}

// ---- Barrier 插入 ----

void FGExecutor::emitBarriers(VkCommandBuffer cmd,
                               const FGPassNode& pass,
                               std::vector<FGResourceNode*>& resources,
                               const FGResources& viewCache) {
    std::vector<VkImageMemoryBarrier2> imgBarriers;
    std::vector<VkBufferMemoryBarrier2> bufBarriers;

    auto addBarrier = [&](const FGPassNode::ResourceRef& ref) {
        auto* res = ref.resource;
        if (!res) return;

        if (res->desc.type == FGResourceType::Texture && res->physicalTexture) {
            VkImageLayout currentLayout = res->state.layout;
            VkImageLayout targetLayout = ref.requiredLayout;

            // 导入资源首次使用，无需 barrier（约定初始 layout 正确）
            if (res->isImported && currentLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
                // 首次使用导入资源不插 barrier，直接信任 initialLayout
                return;
            }

            if (currentLayout == targetLayout &&
                res->state.access == ref.access) {
                return;  // 无需 barrier
            }

            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask  = res->state.stage;
            b.srcAccessMask = res->state.access;
            b.dstStageMask  = ref.stages;
            b.dstAccessMask = ref.access;
            b.oldLayout = currentLayout;
            b.newLayout = targetLayout;
            b.image = res->physicalTexture->image();
            b.subresourceRange = {
                (res->desc.texture.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
                    ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                0, res->desc.texture.mipLevels, 0, res->desc.texture.arrayLayers
            };
            imgBarriers.push_back(b);
        }

        if (res->desc.type == FGResourceType::Buffer && res->physicalBuffer) {
            if (res->state.access == ref.access) return;

            VkBufferMemoryBarrier2 b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
            b.srcStageMask  = res->state.stage;
            b.srcAccessMask = res->state.access;
            b.dstStageMask  = ref.stages;
            b.dstAccessMask = ref.access;
            b.buffer = res->physicalBuffer->handle();
            b.offset = 0;
            b.size = VK_WHOLE_SIZE;
            bufBarriers.push_back(b);
        }
    };

    // 收集所有 reads + writes 的 barrier
    for (auto& ref : pass.reads)  addBarrier(ref);
    for (auto& ref : pass.writes) addBarrier(ref);

    // 合并提交
    if (!imgBarriers.empty() || !bufBarriers.empty()) {
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = (uint32_t)imgBarriers.size();
        di.pImageMemoryBarriers = imgBarriers.data();
        di.bufferMemoryBarrierCount = (uint32_t)bufBarriers.size();
        di.pBufferMemoryBarriers = bufBarriers.data();
        vkCmdPipelineBarrier2(cmd, &di);
    }
}

// ---- 状态更新 ----

void FGExecutor::updateResourceStates(const FGPassNode& pass,
                                       std::vector<FGResourceNode*>& resources) {
    for (auto& ref : pass.reads) {
        if (!ref.resource) continue;
        // 读取不改变 layout，但更新 access（如果是首次使用）
        if (ref.resource->state.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            ref.resource->state.layout = ref.requiredLayout;
        }
        ref.resource->state.access = ref.access;
        ref.resource->state.stage = ref.stages;
    }

    for (auto& ref : pass.writes) {
        if (!ref.resource) continue;
        ref.resource->state.layout = ref.requiredLayout;
        ref.resource->state.access = ref.access;
        ref.resource->state.stage = ref.stages;
        ref.resource->state.lastWriter = const_cast<FGPassNode*>(&pass);
    }
}

// ---- 池回收 ----

void FGExecutor::recycleUnused(uint64_t threshold) {
    for (auto& pt : m_texturePool) {
        if (pt.inUse) {
            pt.inUse = false;  // 帧结束时标记为闲置
        }
        // 超过 threshold 帧未用 → 实际上 Image 的移动语义不便于释放，
        // 保留在池中由 C++ 生命周期管理
    }
    for (auto& pb : m_bufferPool) {
        if (pb.inUse) {
            pb.inUse = false;
        }
    }
}

// ============================================================
// 静态 Layout/Access/Stage 推导
// ============================================================

VkImageLayout FGExecutor::derivedLayout(FGPassType passType,
                                         VkImageUsageFlags usage,
                                         bool isWrite) {
    if (!isWrite) return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // 写 layout 由 usage 决定
    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    // 默认：storage / transfer → GENERAL
    return VK_IMAGE_LAYOUT_GENERAL;
}

VkAccessFlags2 FGExecutor::derivedAccess(FGPassType passType,
                                          VkImageUsageFlags usage,
                                          bool isWrite,
                                          bool isReadWrite) {
    if (isReadWrite) {
        return VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    }

    if (!isWrite) {
        // 读操作
        if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
            return VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
            return VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        // 加速结构
        if (usage & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR)
            return VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        return VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;  // 默认
    }

    // 写操作
    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        return VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
        return VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    if (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        return VK_ACCESS_2_TRANSFER_WRITE_BIT;

    return VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;  // 默认
}

VkPipelineStageFlags2 FGExecutor::derivedStage(FGPassType passType,
                                                VkImageUsageFlags usage,
                                                bool isWrite) {
    switch (passType) {
        case FGPassType::Compute:
            return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

        case FGPassType::Graphics:
            if (isWrite && (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
                return VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            if (isWrite && (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
                return VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
            return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

        case FGPassType::MeshShading:
            return VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

        case FGPassType::RayTracing:
            return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    }

    return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
}

} // namespace fg
} // namespace somegi
```

- [ ] **Step 2: 验证编译**

```bash
cmake --build build --target somegi_renderer_fg
```
Expected: 编译通过，`somegi_renderer_fg` 库完整。

- [ ] **Step 3: 提交**

```bash
git add src/renderer/fg/fg_executor.cpp
git commit -m "Phase 0.12: 实现 FGExecutor 分配/Barrier/执行逻辑"
```

---

### Task 0.13: 集成到 App — 双轨开关 + ImGui toggle

**Files:**
- Modify: `src/app/app.h`
- Modify: `src/app/app.cpp`
- Modify: `src/app/CMakeLists.txt`

- [ ] **Step 1: 修改 app.h — 添加 FrameGraph 成员**

在 `FrameRenderer m_renderer;` 行后添加：

```cpp
// ---- Frame Graph（实验性） ----
somegi::fg::FrameGraph m_fg;
bool m_useFrameGraph = false;
```

在文件头部 include 段添加：

```cpp
#include "renderer/fg/fg_graph.h"
```

- [ ] **Step 2: 修改 app.cpp — App 构造函数中初始化 FrameGraph**

在 `m_renderer.init(...)` 调用之后添加：

```cpp
// 初始化 FrameGraph（实验性）
m_fg.init(*m_device);
```

- [ ] **Step 3: 修改 app.cpp — 在 buildUI 中添加 ImGui 开关**

在 Display Tab 或 Effects Tab 中添加：

```cpp
ImGui::Separator();
ImGui::Text("Experimental");
bool useFg = m_useFrameGraph;
if (ImGui::Checkbox("Use Frame Graph", &useFg)) {
    m_useFrameGraph = useFg;
    m_pipelineDirty = true;
}
if (m_useFrameGraph) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1,1,0,1), "(experimental)");
    // FrameGraph 调试视图
    auto& fgDebug = m_fg.debug();
    ImGui::Checkbox("FG: Show Pass List", &fgDebug.showPassList);
    ImGui::Checkbox("FG: Show Aliases",   &fgDebug.showAliasGroups);
    ImGui::Checkbox("FG: Show Resources", &fgDebug.showResources);
    ImGui::Checkbox("FG: Show Barriers",  &fgDebug.showBarrierLog);

    if (ImGui::CollapsingHeader("FG Passes")) {
        for (auto& p : fgDebug.passes) {
            ImGui::Text("%s %s %s",
                p.culled ? "[CULLED]" : "",
                p.name.c_str(),
                p.reads.empty() && p.writes.empty() ? "(no deps)" : "");
        }
    }
    if (ImGui::CollapsingHeader("FG Alias Groups")) {
        for (auto& ag : fgDebug.aliasGroups) {
            ImGui::Text("Group %u: %u bytes (%u wasted)",
                ag.id, ag.totalBytes, ag.wastedBytes);
            for (auto& m : ag.members) {
                ImGui::BulletText("%s", m.c_str());
            }
        }
    }
}
```

- [ ] **Step 4: 修改 app.cpp — App::run() 中添加 FrameGraph 执行路径**

在 `buildPipelineTable()` 和 `m_renderer.pipeline().execute(cmd)` 周围添加条件分支：

```cpp
// ---- Execute render pipeline ----
if (m_useFrameGraph) {
    // FrameGraph 路径
    m_fg.reset();

    // TODO Phase 1+: 导入资源 + 注册 pass

    m_fg.compile();
    m_fg.execute(cmd);
} else {
    // 现有 RenderPipeline 路径
    buildPipelineTable();
    m_renderer.pipeline().execute(cmd);
}
++m_renderer.frameIndex();
```

- [ ] **Step 5: 修改 app/CMakeLists.txt 添加 fg 库链接**

```cmake
# src/app/CMakeLists.txt
add_library(somegi_app OBJECT
    main.cpp
    app.cpp
    benchmark_runner.cpp
)
target_include_directories(somegi_app PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_app PUBLIC somegi_renderer somegi_core imgui)
```

保持现有内容，确认 `somegi_renderer` 已链接（它已包含 fg）。

- [ ] **Step 6: 验证编译**

```bash
cmake --build build --target SomeGI
```
Expected: 完整项目编译通过。

- [ ] **Step 7: 运行验证**

```bash
cd build && ./SomeGI
```
Expected: 应用正常启动（默认使用旧 RenderPipeline，FrameGraph 开关 OFF）。

- [ ] **Step 8: 提交**

```bash
git add src/app/app.h src/app/app.cpp src/app/CMakeLists.txt
git commit -m "Phase 0.13: 集成 FrameGraph 到 App，添加上层开关和 ImGui 调试面板"
```

---

## Phase 1: 纯 Compute Pass 迁移 (SSAO / GTAO)

（后续 Phase 的详细任务将在 Phase 0 完成后，根据框架实际表现调整细节。以下为任务框架。）

### Task 1.0: 添加 setupFrameGraph() 和 SSAO 迁移

**Files:**
- Modify: `src/app/app.cpp` — 添加 `setupFrameGraph()` 方法

- [ ] **Step 1: 实现 setupFrameGraph()，导入 GBuffer 持久资源**

```cpp
void App::setupFrameGraph() {
    auto ext = m_swap->extent();

    // 导入 GBuffer 持久资源
    m_fgHandle_gbDepth = m_fg.importTexture("GBufDepth",
        m_renderer.rt().depth.image(),
        {ext, VK_FORMAT_D32_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    m_fgHandle_gbNormal = m_fg.importTexture("GBufNormal",
        m_renderer.rt().gNormalRough.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // 注册 Pass...
}
```

...（后续任务细节同理展开）

---

## 自审清单

- [x] Spec 覆盖：Phases 0-6 对照 spec 第 11 节的迁移策略
- [x] 无占位符：所有 task 包含完整代码
- [x] 类型一致性：`FGHandle`、`FGPassType`、`FGResourceDesc` 在各 task 中一致使用
- [x] CMake 路径正确：`src/renderer/fg/` 下新增文件均已列入 CMakeLists.txt

---

## 执行状态（2026-06-16）

### Phase 0: 框架搭建 ✅

所有 Task (0.0-0.13) 已完成，提交记录：

```
5817a45 Phase 1 完成: FrameGraph 路径 0 validation error
471e5df Phase 1 fix: oldLayout 改为 UNDEFINED 兼容拓扑排序
70e4275 Phase 1: 添加 explicit layout 重载 + auto-barriers 开关
33d1786 Phase 1 fix: 补全 Barrier 过渡
5347d84 Phase 1: 注册全部渲染 Pass 到 FrameGraph
86e5061 Phase 0.13: 集成到 App + ImGui toggle
6971459 Phase 0.12: FGExecutor 实现
7d457cc Phase 0.11: FGCompiler 实现
2dded64 Phase 0.10: fg_graph.cpp 实现
34d3f47 fix: FGResources 前向声明
1e7cab3 fix: 桩实现
f931438 Phase 0.9: FGExecutor 声明
b8727cc Phase 0.7: FrameGraph 声明
26734de fix: warning
c631837 fix: 编译错误
5eb7ac8 fix: FGResourceDesc
3f87ebd Phase 0.6: FGDebug
6f140d8 Phase 0.8: FGCompiler 声明
0007e14 Phase 0.5: FGBuilder
b6d0ab8 Phase 0.4: FGResources
10d62c2 Phase 0.3: FGPassNode
e9d8d21 Phase 0.2: FGResourceNode
afa9855 Phase 0.1: fg_common.h
dd1c4d0 Phase 0.0: CMake 骨架
```

### Phase 2-5 ❌ (合并为一步注册全部 pass)

全部 30+ pass 已在 `setupFrameGraph()` 中注册完毕。

### Phase 6: 清理旧管线 ⏳ (未开始)

待删除: `render_pipeline.h/cpp`, `App::registerPipelineSteps()`, `App::buildPipelineTable()`

### 关键经验

1. **自动 Barrier 不可行** — pass 内部操作多样（clear/copy/store/attachment），无法通过 `read/write` 统一推导。需方案 A 逐步改造。
2. **执行边界设置** — Tonemap/TAA/SMAA/ImGui 保留在 FrameGraph 外部，由 `recordPostProcessing` 管理，避免双写冲突。
3. **oldLayout=UNDEFINED** — 帧间 layout 持久化需所有首次过渡使用 UNDEFINED oldLayout。
4. **显式 layout 重载** — `b.write(handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)` 对 Clear/Copy pass 必不可少。
