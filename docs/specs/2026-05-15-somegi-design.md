# SomeGI — Vulkan 全局光照实验平台 设计文档

日期: 2026-05-15
最后修订: 2026-05-20（M9 / M10 已实现；GPU 升级至 RTX 4060，支持 HW RT）
状态: Draft, 待评审

## 1. 目标

构建一个使用 C++/Vulkan 1.3 的桌面渲染程序，能加载 glTF v2 模型并以可切换方式演示八种全局光照（GI）技术：IBL、SSGI、RSM、LPV、VXGI、PRT、Ray Tracing、ReSTIR。GI 算法之间共享场景、材质、相机与公共后处理，仅 GI 部分通过统一基类做插件化替换。

非目标：
- 不做编辑器（不实现属性面板编辑材质）。
- 不做 mobile / Vulkan 1.0。
- 不做完整离线渲染基线（路径追踪只做实时近似，作为参考但非地面真值）。

## 2. 关键决策（已与用户确认）

| 项 | 选择 |
|----|------|
| 构建 | CMake + MSVC + Vulkan SDK |
| Vulkan | 1.3，必须支持 KHR_ray_tracing |
| GI 组织 | 单可执行 + ImGui 下拉框切换 |
| Shader | Slang → SPIR-V，CMake build-time 编译 |
| 第三方库 | git submodule 放 `external/` |

## 3. 目录结构

```
SomeGI/
├── CMakeLists.txt
├── external/                      # git submodules
│   ├── glfw/
│   ├── glm/
│   ├── imgui/
│   ├── cgltf/                     # header-only glTF v2 loader
│   ├── stb/                       # stb_image
│   └── vk-bootstrap/              # 简化设备/物理设备初始化
├── assets/
│   └── gltf/{cube,Sponza}/
├── docs/specs/
├── shaders/                       # *.slang 源
│   ├── common/
│   ├── forward/
│   └── gi/{ibl,ssgi,rsm,lpv,vxgi,prt,rt,restir}/
├── src/
│   ├── core/                      # vk 设备/swap/cmd/buffer/image/sync
│   ├── scene/                     # gltf_loader, scene, material, camera
│   ├── renderer/                  # gbuffer, forward pass, post, imgui_pass
│   ├── gi/
│   │   ├── gi_technique.h         # 基类 IGITechnique
│   │   ├── gi_technique.cpp       # 工具/共享实现
│   │   ├── ibl_technique.{h,cpp}
│   │   ├── ssgi_technique.{h,cpp}
│   │   ├── rsm_technique.{h,cpp}
│   │   ├── lpv_technique.{h,cpp}
│   │   ├── vxgi_technique.{h,cpp}
│   │   ├── prt_technique.{h,cpp}
│   │   ├── rt_technique.{h,cpp}
│   │   └── restir_technique.{h,cpp}
│   └── app/                       # main, App, ui_panel
└── build/                         # 由 CMake 生成
```

## 4. 模块边界与职责

### 4.1 core/
封装所有 Vulkan 原语，对上层只暴露 RAII 句柄与录制接口：
- `Device` — instance / physical / logical / queues / VMA-like 简易分配器（先用 VkAllocateMemory，后续可换 VMA）
- `Swapchain` — 多帧 in-flight、sync2、dynamic rendering
- `CommandPool/Buffer` — per-frame、per-thread
- `Buffer / Image / Sampler` — 资源 + 描述符
- `Pipeline / Shader` — 加载 spv，反射出布局
- `DescriptorAllocator` — 简化绑定
- `RTHelpers` — BLAS/TLAS 构建、SBT（仅在 RT 可用时）

依赖：vulkan, glfw, vk-bootstrap, glm。

### 4.2 scene/
- `GltfLoader` — 用 cgltf，转成内部表示
- `Scene` — meshes、nodes、lights、camera、materials
- `Material` — 完整 PBR metallic-roughness + 全部 glTF v2 扩展槽位（baseColor, metallicRoughness, normal, occlusion, emissive, KHR_materials_clearcoat, _sheen, _transmission, _ior, _specular, _volume, _emissive_strength, _unlit）
- `EnvProbe` — HDR 环境贴图（默认 cubemap，含预滤波/BRDF LUT 在 IBL 模块复用）
- `Camera` — flying-camera + 鼠标右键拖拽 + WASD

依赖：core, cgltf, stb, glm。

### 4.3 renderer/
- `GBufferPass` — 输出 albedo / normal+roughness / metallic+occlusion / depth / motion
- `ForwardPass` — 透明物体直接前向（与 GI 模块协作）
- `LightingPass` — direct lighting (analytic light) + 调用 GI::evaluateIndirect
- `PostPass` — tonemap (ACES) + sRGB
- `ImGuiPass` — 单独 pass，最后绘制

GI 不是一个 pass，而是被 LightingPass 调用的一组资源 + shader 片段（见 4.4）。

### 4.4 gi/ — 核心抽象

```cpp
// gi_technique.h
class IGITechnique {
public:
    virtual ~IGITechnique() = default;
    virtual const char* name() const = 0;

    // 一次性：在 GI 切换/场景加载时调用
    virtual void onAttach(const GIContext& ctx) = 0;
    virtual void onDetach() = 0;

    // 每帧：在主帧 cmd 中预计算 (voxelize / inject / build TLAS / spatial-temporal reuse)
    virtual void prepare(VkCommandBuffer cmd, const FrameContext& fc) = 0;

    // 在 LightingPass 中绑定 GI 描述符 / 调度 indirect lighting compute
    virtual void evaluate(VkCommandBuffer cmd, const FrameContext& fc) = 0;

    // ImGui 自定义参数面板
    virtual void drawUI() = 0;
};
```

`GIContext` 提供：device、shared layouts、scene 句柄、gbuffer image views、env probe。
`FrameContext` 提供：frame index、camera、view/proj、jitter、prev matrices、HDR 输出 image。

每种实现独立 cpp + 独立 slang。`App` 持 `std::unique_ptr<IGITechnique>`，UI 切换时 detach 旧的、attach 新的。

#### 8 种实现要点
- **IBL**: 启动时离线预滤波 env map（split-sum）+ 生成 BRDF LUT。`evaluate` 只是采样 + GGX 镜面/兰伯特漫反射。无需每帧 prepare。
- **SSGI**: 屏幕空间一次反弹漫反射，基于 cosine 半球 ray march + hdrPrev 取色。配套 SSAO + SSR。compute pipeline。
- **RSM**: 从 sun 视角渲扩展 shadow map（depth + worldPos + worldNormal + flux=albedo·sunColor），屏幕像素在 RSM 上以 N 个 disk 样本抽 VPL 累积一次反弹间接光。`prepare` 重渲 RSM（sun 方向 / 场景动时）。
- **LPV**: 体素化场景到 32³ 网格，每个格子用 SH 系数表示入射 radiance；`prepare` = inject（从 RSM 当源）+ propagate（N 次 6 邻居 SH 转移）；`evaluate` = SH·N 还原 diffuse。低频但稳定。
- **VXGI**: clip-map 体素（256³ × 6 cascades，先做 128³ 单 cascade）；`prepare` 做 voxelize + light injection + mipmap；`evaluate` 做 cone tracing。
- **PRT**: 启动时 / 离线烘焙 SH9 transfer 系数到 vertex 或 lightmap UV；`evaluate` 做 SH 重建。环境光投影到 SH。先实现 vertex baking 版本。
- **Ray Tracing**: 用 KHR_ray_tracing 跑 1 spp diffuse + spec 反射 + 时序累积。`prepare` 更新 TLAS 与 SBT。
- **ReSTIR**: 在 RT 之上做 spatial-temporal 重采样（GRIS/RIS），只取 DI（先做）+ GI（再做）。

## 5. 数据流（每帧）

```
FrameStart
 → Camera.update (CPU)
 → Scene.uploadDirty (transfer)
 → GBufferPass (graphics)
 → CurrentGI.prepare (compute / RT, depends)
 → LightingPass (compute, calls CurrentGI.evaluate)
 → ForwardPass (透明)
 → PostPass (compute)
 → ImGuiPass (graphics, render-to-swapchain)
 → Present
```

帧内全部使用 sync2 + dynamic rendering，避免 RenderPass 对象。

## 6. Shader 工具链

- 源在 `shaders/`，扩展名 `.slang`。
- CMake 自定义 target `compile_shaders`：扫描所有 `.slang`，调 `slangc -profile spirv_1_5 -target spirv -fvk-use-entrypoint-name` 编译到 `${CMAKE_BINARY_DIR}/shaders/...spv`。
- 主程序运行时按相对路径加载 spv。
- 公共代码放 `shaders/common/*.slang`，通过 `import` 复用。

## 7. 错误处理与诊断

- 启用 VK_LAYER_KHRONOS_validation（debug 配置）。
- 自带 debug callback，VkResult != SUCCESS 抛 `std::runtime_error`。
- ImGui 面板含 "GPU Stats / Frame Time / Pipeline Toggle / Shader Reload" 按钮（Shader hot-reload 用 std::filesystem 监视文件 mtime）。

## 8. 测试与验证

- 单元层：scene 解析（cube：1 mesh、6 face、24 vertex；Sponza：~25 meshes、有 normalmap）。
- 渲染验证（人工肉眼 + 截图 diff）：
  1. 加载 cube 显示 (vertex color / albedo)。
  2. 加载 Sponza 直接光显示。
  3. 每接入一个 GI，做 "GI off vs on" 对比截图。
- 暂不引入图像比对自动测试；有需要后续用 stb_image_write 输出帧 + Python 脚本算 SSIM。

## 9. 里程碑（与你给出的步骤对齐）

| M | 范围 | 验证 |
|---|------|------|
| M0 | 项目脚手架（CMake、submodule、Vulkan/GLFW 初始化、清屏） | 出现窗口，clear color 正常 |
| M1 | glTF 加载 + GBuffer + 直接光 + tonemap | cube 与 Sponza 显示正确 |
| M2 | ImGui 集成 + 相机 / 场景 / debug UI | UI 可切窗口、调相机、显示 FPS |
| M3 | GI 基类 + IBL 实现 | env map IBL 正确（金属球反射、漫反射环境色） |
| M4 | SSGI（含 GBuffer + Deferred Lighting + SSAO + SSR） | SSGI 看到色溢；SSAO 拐角变暗；SSR 反射场景 |
| **M5** | **RSM**（Reflective Shadow Maps，单光源一次反弹间接光） | sun 朝向变化时受光面附近间接光跟随；与 IBL 叠加合理 |
| **M6** | **LPV**（Light Propagation Volumes，体素 + SH 传播） | 红/绿色帘附近白柱有色溢；穿过几何漏光在可接受范围 |
| M7 | VXGI | 单 cascade voxel 可视化 + 间接光（旧 M5） |
| M8 | PRT | SH9 烘焙 + 旋转环境光实时变化（旧 M6） |
| M9 | Ray Tracing GI | 1 spp + 时序累积，与 IBL 比对（旧 M7） |
| M10 | ReSTIR DI（再 GI） | 含多光源场景的 noise 显著降低（旧 M8）；HW RT 可见性 |

每个里程碑都需要 cube + Sponza 两个场景跑通才能进入下一阶段。

## 10. 风险与缓解

- **Slang 工具链问题** → 退路：把对应 shader 写成 GLSL，由 glslangValidator 编译，不影响整体架构。
- **RT 硬件不可用** → 启动检测，UI 中 e/f 选项灰色 + tooltip。
- **VXGI 显存压力** → 先 128³ 单 cascade，跑通再扩 256³ + cascade。
- **PRT 烘焙慢** → 先做 vertex 级、低样本数 (64 rays/vertex)，明确仅作演示。
- **RSM sun frustum 覆盖不到的物体** → 先接受（用 IBL diffuse 当 fallback），后续可加多张 RSM（cascade）。
- **RSM 单光源限制** → M5 只接 directional sun；点 / 聚光的 RSM 是后续扩展。
- **LPV 漏光（光穿墙）** → SH 表示天然低频；M6 接受漏光，加 occlusion volume 可缓解但不是 M6 范围。
- **LPV 网格精度** → 32³ 起步省显存，跑通后可扩到 64³ 或加 cascade。

## 11. YAGNI 明确移除

- 不做编辑器、不做物理、不做骨骼动画（即使 glTF 含 animations 也只静态加载第一帧）。
- 不做 multi-GPU、不做 Vulkan 1.0/1.1 兼容。
- IBL 不做实时环境捕获，启动时离线一次。
- PRT 不做 wavelet / volumetric，只 SH9 vertex 级。

## 12. 待定 / 默认值（可在实现期调整）

- 窗口尺寸 1600×900。
- HDR env：内置 1 张 .hdr（papermill / ennis）随源码提交（小尺寸 512²）。
- 相机：FPS 风格，鼠标右键持续拖动旋转，WASD 平移，QE 升降。
- 选择 cgltf 而非 tinygltf：纯 C 头文件、零依赖、生命周期清晰。

