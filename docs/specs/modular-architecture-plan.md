# SomeGI 模块化架构方案

## 现状问题

当前 `somegi_renderer` 是一个单体静态库，包含 85 个源文件：
- 核心管线（GBuffer/Lighting/Tonemap/AA）与 13 种 GI 技术混在一起
- 所有 Pass 在 `FrameRenderer::init()` 中无条件初始化（无论是否使用）
- 新增/移除 GI 技术需要修改 FrameRenderer、CMakeLists.txt、App
- 无法独立编译或测试单个 GI 技术

## 目标架构

```
src/
├── core/           # 不变 — Vulkan 封装层
├── scene/          # 不变 — 场景加载层
├── gi/             # 不变 — GI 抽象接口 + IBL
├── renderer/
│   ├── core/       # 核心渲染管线（新拆分）
│   ├── screenspace/# SSR, SSGI, GTGI
│   ├── ao/         # SSAO, GTAO
│   └── gi/         # 所有 GI 技术（每技术一个子目录）
│       ├── rsm/
│       ├── lpv/
│       ├── vxgi/
│       ├── ddgi/
│       ├── ndgi/
│       ├── sdfgi/
│       ├── prt/
│       ├── restir/
│       ├── rt/
│       └── lumen/
└── app/            # 不变 — 应用入口
```

每个模块是独立的 `OBJECT` 库，最终链接到 `somegi_renderer`。

## 方案细节

### 1. 目录重组

```
src/renderer/
├── CMakeLists.txt              # 聚合 OBJECT 库 → somegi_renderer
├── core/
│   ├── CMakeLists.txt          # somegi_renderer_core OBJECT 库
│   ├── render_targets.h/cpp
│   ├── barrier_manager.h/cpp
│   ├── render_pipeline.h/cpp
│   ├── gbuffer_pass.h/cpp
│   ├── forward_pass.h/cpp
│   ├── lighting_pass.h/cpp
│   ├── skybox_pass.h/cpp
│   ├── tonemap_pass.h/cpp
│   ├── taa_pass.h/cpp
│   ├── smaa_pass.h/cpp
│   ├── imgui_pass.h/cpp
│   └── frame_renderer.h/cpp
├── ao/
│   ├── CMakeLists.txt          # somegi_renderer_ao OBJECT 库
│   ├── ssao_pass.h/cpp
│   └── gtao_pass.h/cpp
├── screenspace/
│   ├── CMakeLists.txt          # somegi_renderer_screenspace OBJECT 库
│   ├── ssr_pass.h/cpp
│   ├── ssgi_pass.h/cpp
│   └── gtgi_pass.h/cpp
└── gi/
    ├── CMakeLists.txt          # 聚合所有 GI OBJECT 库
    ├── rsm/
    │   ├── CMakeLists.txt
    │   ├── rsm_geometry_pass.h/cpp
    │   └── rsm_sample_pass.h/cpp
    ├── lpv/
    │   ├── CMakeLists.txt
    │   ├── lpv_grid.h/cpp
    │   ├── lpv_inject_pass.h/cpp
    │   └── lpv_propagate_pass.h/cpp
    ├── vxgi/
    │   ├── CMakeLists.txt
    │   ├── vxgi_resources.h/cpp
    │   ├── vxgi_voxelize_pass.h/cpp
    │   ├── vxgi_inject_pass.h/cpp
    │   ├── vxgi_mipmap_pass.h/cpp
    │   ├── vxgi_aniso_pass.h/cpp
    │   ├── vxgi_relight_pass.h/cpp
    │   └── vxgi_resolve_6axis_pass.h/cpp
    ├── ddgi/  (同上模式)
    ├── ndgi/
    ├── sdfgi/
    ├── prt/
    ├── restir/
    ├── rt/      (scene_rt_as + rt_gi_pass)
    └── lumen/   (lumen_resources + probe + filter + gather)
```

### 2. CMake 结构

每个子目录定义一个 `OBJECT` 库，顶层 `somegi_renderer` 聚合所有 OBJECT 库：

```cmake
# src/renderer/core/CMakeLists.txt
add_library(somegi_renderer_core OBJECT
    render_targets.cpp barrier_manager.cpp render_pipeline.cpp
    gbuffer_pass.cpp forward_pass.cpp lighting_pass.cpp
    skybox_pass.cpp tonemap_pass.cpp taa_pass.cpp smaa_pass.cpp
    imgui_pass.cpp frame_renderer.cpp
)
target_include_directories(somegi_renderer_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_core PUBLIC somegi_core somegi_scene somegi_gi imgui)

# src/renderer/ao/CMakeLists.txt
add_library(somegi_renderer_ao OBJECT ssao_pass.cpp gtao_pass.cpp)
target_include_directories(somegi_renderer_ao PUBLIC ${CMAKE_SOURCE_DIR}/src)

# src/renderer/gi/rsm/CMakeLists.txt
add_library(somegi_renderer_rsm OBJECT rsm_geometry_pass.cpp rsm_sample_pass.cpp)
target_include_directories(somegi_renderer_rsm PUBLIC ${CMAKE_SOURCE_DIR}/src)

# src/renderer/CMakeLists.txt (顶层，聚合)
add_library(somegi_renderer STATIC
    $<TARGET_OBJECTS:somegi_renderer_core>
    $<TARGET_OBJECTS:somegi_renderer_ao>
    $<TARGET_OBJECTS:somegi_renderer_screenspace>
    $<TARGET_OBJECTS:somegi_renderer_rsm>
    $<TARGET_OBJECTS:somegi_renderer_lpv>
    $<TARGET_OBJECTS:somegi_renderer_vxgi>
    $<TARGET_OBJECTS:somegi_renderer_ddgi>
    $<TARGET_OBJECTS:somegi_renderer_ndgi>
    $<TARGET_OBJECTS:somegi_renderer_sdfgi>
    $<TARGET_OBJECTS:somegi_renderer_prt>
    $<TARGET_OBJECTS:somegi_renderer_restir>
    $<TARGET_OBJECTS:somegi_renderer_rt>
    $<TARGET_OBJECTS:somegi_renderer_lumen>
)
target_link_libraries(somegi_renderer
    PUBLIC somegi_core somegi_scene somegi_gi imgui
)
```

OBJECT 库的优点：
- 每个模块独立编译（修改一个 GI 技术不影响其他模块的 object 文件）
- 最终链接为一个 `somegi_renderer`（不增加 DLL/SO 数量）
- CMake 自动管理依赖传递

### 3. 依赖关系

```
somegi_renderer_core
    ← somegi_core, somegi_scene, somegi_gi, imgui

somegi_renderer_ao
    ← somegi_renderer_core (需要 RenderTargets, GBufferPass::frameUboHandle)

somegi_renderer_screenspace
    ← somegi_renderer_core (需要 RenderTargets, GBufferPass::frameUboHandle)

GI 技术模块（每个独立）:
    ← somegi_renderer_core (需要 RenderTargets, GBufferPass::frameUboHandle)
    ← 彼此独立，无交叉依赖

例外依赖（跨模块资源引用，通过 VkBuffer/VkImageView 句柄传递，非 include）:
    rsm ← lpv (RSM position/normal/flux image → lpv inject)
    rsm ← vxgi (RSM position/flux image → vxgi inject)
    vxgi ← ddgi, sdfgi, restir, lumen, prt (voxel grid 共享)
```

### 4. FrameRenderer 使用模块注册表

当前 FrameRenderer 硬编码所有 Pass 的初始化。模块化后改为注册表模式：

```cpp
// frame_renderer.h
class FrameRenderer {
public:
    // 注册一个 GI 技术模块（调用方在 App 初始化时注册所有需要的模块）
    void registerGiModule(std::unique_ptr<IGiModule> module);

    // init 只初始化核心管线；GI 模块在 registerGiModule 时自行初始化
    void init(Device& d, VkCommandPool pool, VkExtent2D extent, ...);

private:
    std::vector<std::unique_ptr<IGiModule>> m_giModules;
};
```

### 5. IGiModule 接口（可选，远期）

```cpp
// src/renderer/gi/gi_module.h
class IGiModule {
public:
    virtual ~IGiModule() = default;
    virtual const char* name() const = 0;
    virtual int giIndex() const = 0;        // 对应 kGis[] 中的索引

    // 生命周期
    virtual void init(Device& d, GiModuleContext& ctx) = 0;
    virtual void destroy() = 0;

    // Scene 绑定
    virtual void bindScene(Device& d, const SceneGpu& gpu, uint32_t textureCount) {}

    // 每帧录制（由 FrameRenderer 调用）
    virtual void record(VkCommandBuffer cmd, const GiFrameContext& ctx) {}

    // 在 RenderPipeline 中注册自己的 steps
    virtual void registerPipelineSteps(RenderPipeline& pipeline) = 0;

    // 启用/禁用
    virtual void setEnabled(bool enabled) = 0;
    virtual bool isEnabled() const = 0;
};
```

### 6. 实施步骤

| 步骤 | 内容 | 改动量 |
|---|---|---|
| Step 1 | 创建子目录结构 + 移动文件（git mv 保留历史） | ~85 个文件移动 |
| Step 2 | 为每个子目录创建 CMakeLists.txt（OBJECT 库） | 13 个新 CMakeLists |
| Step 3 | 更新 include 路径（`"renderer/xxx.h"` → `"renderer/core/xxx.h"` 等） | ~200 处修改 |
| Step 4 | 更新 FrameRenderer 的 include 列表 | 1 个文件 |
| Step 5 | 编译验证 + 修复 | 多轮 |

每个 Step 后编译通过再继续下一步。

### 7. 收益

| 维度 | 重构前 | 重构后 |
|---|---|---|
| 新增 GI 技术 | 修改 5+ 文件（CMakeLists, frame_renderer.h/cpp, app.h/cpp） | 新建子目录 + 在 CMake 中加一行 |
| 编译时间 | 改任意 Pass 重编整个 renderer | 只重编改动的模块 |
| 依赖关系 | 隐式（所有 Pass 互相可见 header） | 显式（CMake target_link 声明） |
| 测试 | 无法独立测试单个 GI | 可以链接单个 OBJECT 库 + mock |
| 可选编译 | 不支持 | 可通过 CMake option 裁剪不需要的 GI |

### 8. 验证

1. 每步 `cmake --build` 零错误
2. 最终运行：所有 13 种 GI 切换正常，画面无变化
3. `git mv` 保留所有文件的 Git 历史
