#pragma once
#include "core/vk_common.h"
#include "core/screenshot.h"
#include "scene/scene.h"
#include "scene/camera.h"
#include "scene/draw_list.h"
#include "renderer/core/frame_ubo.h"      // for FrameUBO type
#include "renderer/core/frame_context.h"
#include "renderer/core/frame_renderer.h"
#include "renderer/fg/fg_graph.h"
#include "rhi/base/context.h"
#include "app/benchmark_runner.h"
#include <map>
#include <memory>
#include <filesystem>
#include <string>

namespace somegi {

class Window;
class Device;
class Swapchain;
class IGITechnique;

// Per-scene persisted view + lighting.
struct SceneState {
    glm::vec3 camPos{0};
    float yaw = 0, pitch = 0, fov = 60.0f;
    glm::vec3 sunDir{-0.4f, -1.0f, -0.3f};
    float sunIntensity = 3.0f;
    glm::vec3 ambient{0.10f, 0.12f, 0.15f};
    float taaBlendAlpha = 0.92f;
    bool camValid = false;
};

class App {
public:
    App();
    // D3D12 专用构造标签
    struct ForD3D12 {};
    explicit App(ForD3D12);
    ~App();
    void run();
    // D3D12 独立渲染循环（后端为 D3D12 时自动调用）
    void runD3D12();

    // 由 main() 在构造后、run() 前调用，传入 CLI 截图配置
    void setScreenshotConfig(int interval, int oneFrame, const char* dir);
    void setInitialShadowMethod(int method);   // --shadow-method CLI
    void setExitAfterCapture(bool v) { m_exitAfterCapture = v; }

    // 图形 API 后端（--backend vulkan|d3d12）
    void setBackend(const char* name);
    const char* backendName() const { return m_backendName.c_str(); }
    bool backendIs(const char* name) const { return m_backendName == name; }

    // D3D12 专用构造函数（跳过 Vulkan 初始化）
    static App createForD3D12();
    // 回归测试（--capture-ref / --capture-compare）
    void setCaptureRefMode(bool v)      { m_captureRef = v; }
    void setCaptureCompareMode(bool v, double thresh) { m_captureCompare = v; m_refThreshold = thresh; }

private:
    void onSwapchainResized();
    bool loadAndUploadScene(const std::filesystem::path& gltfPath, std::string& outErr);
    void applySceneSelection();
    SceneState captureSceneState() const;
    void applyState(const SceneState& s);
    void persistAllStates();
    void cleanup();

    // ---- Platform ----
    std::unique_ptr<Window> m_window;
    std::unique_ptr<Device> m_device;
    std::unique_ptr<Swapchain> m_swap;
    // RHI 命令上下文（管理命令池 + cmd buffer + fence，替代直接 Vulkan 调用）
    std::unique_ptr<rhi::RHIContext> m_context;

    // ---- Scene + Camera ----
    SceneCpu m_scene;
    SceneGpu m_sceneGpu;
    Camera m_camera;
    FlyController m_flyer;

    // ---- Rendering (all passes owned by FrameRenderer) ----
    FrameRenderer m_renderer;

    // ---- Frame Graph（实验性） ----
    somegi::fg::FrameGraph m_fg;
    bool m_useFrameGraph = true;

    // FrameGraph 资源句柄缓存（每帧 setupFrameGraph 填充）
    struct FGH {
        // GBuffer (resolved, single-sample)
        somegi::fg::FGHandle gAlbedoMetal;
        somegi::fg::FGHandle gNormalRough;
        somegi::fg::FGHandle gEmissiveAO;
        somegi::fg::FGHandle depth;

        // GBuffer MSAA
        somegi::fg::FGHandle gAlbedoMetalMs;
        somegi::fg::FGHandle gNormalRoughMs;
        somegi::fg::FGHandle gEmissiveAOMs;
        somegi::fg::FGHandle depthMs;

        // AO / SSR / SSGI
        somegi::fg::FGHandle ssao;
        somegi::fg::FGHandle ssr;
        somegi::fg::FGHandle ssgi;
        somegi::fg::FGHandle ssgiPrev;

        // HDR
        somegi::fg::FGHandle hdrColor;
        somegi::fg::FGHandle hdrPrev;

        // LDR output
        somegi::fg::FGHandle ldrTonemap;

        // GI outputs
        somegi::fg::FGHandle rtGI;
        somegi::fg::FGHandle restir;
        somegi::fg::FGHandle rsmGI;
        somegi::fg::FGHandle lumenGI;

        // AA intermediates
        somegi::fg::FGHandle aaHdr;
        somegi::fg::FGHandle aaHistory;

        // Shadow mask
        somegi::fg::FGHandle shadowMask;

        // Swapchain（每帧导入，初始布局 UNDEFINED）
        somegi::fg::FGHandle swapImage;
    };
    FGH m_fgh = {};

    Buffer m_indirectBuf;
    Buffer m_indirectBufSun;
    Buffer m_countBuf;
    uint32_t m_drawCount = 0;
    uint32_t m_culledDrawCount = 0;
    bool m_useGpuCulling = false;   // 默认关闭，FrameGraph 测试用
    bool m_useHiZOcclusion = false;
    std::vector<DrawEntry> m_drawEntries;

    // ---- ImGui debug window ----
    std::unique_ptr<Window> m_imguiWin;
    std::unique_ptr<Swapchain> m_imguiSwap;
    VkCommandPool m_imguiPool = VK_NULL_HANDLE;
    VkCommandBuffer m_imguiCmds[kFramesInFlight]{};
    VkFence m_imguiFence = VK_NULL_HANDLE;

    // ---- Live lighting values ----
    glm::vec3 m_sunDir{-0.4f, -1.0f, -0.3f};
    float m_sunIntensity = 3.0f;
    glm::vec3 m_ambient{0.10f, 0.12f, 0.15f};

    // ---- UI / Settings ----
    enum class AOMethod : int { None = 0, SSAO = 1, GTAO = 2 };
    AOMethod m_aoMethod = AOMethod::SSAO;
    enum class AAMethod : int { None = 0, MSAA = 1, TAA = 2, SMAA = 3 };
    AAMethod m_aaMethod = AAMethod::MSAA;
    enum class RenderingMode : int { Deferred = 0, Forward = 1 };
    RenderingMode m_renderingMode = RenderingMode::Deferred;
    VkSampleCountFlagBits m_msaaSamples = VK_SAMPLE_COUNT_4_BIT;
    bool m_useMipmaps = true;

    int m_currentSceneIndex = 1;
    int m_sceneIndexApplied = -1;
    std::string m_sceneLoadError;
    int m_currentGiIndex = 1;
    int m_giIndexApplied = -1;

    int m_currentShadowIndex = 1;   // 默认 Hard SM
    int m_shadowIndexApplied = -1;

    // 管线表脏标记：GI/AO/渲染模式切换时置 true，buildPipelineTable() 重建后清除
    bool m_pipelineDirty = true;

    // TAA
    glm::vec2 m_jitter{};
    glm::vec2 m_prevJitter{};
    float m_taaBlendAlpha = 0.92f;
    glm::mat4 m_prevViewProj{1.0f};

    // SH 投影缓存：仅在 sunDir 或 intensity 变化时重算球谐系数
    struct CachedSH {
        glm::vec3 lastSunDir{0.0f};
        float lastSunIntensity = -1.0f;          // -1 确保首帧一定计算
        // SH4（l=0,1 共 4 系数 / RGB 通道）
        glm::vec4 prtLightSH_R, prtLightSH_G, prtLightSH_B;
        // SH9（l=2 追加 5 系数）
        glm::vec4 prtLightSH9_R0, prtLightSH9_R1;
        glm::vec4 prtLightSH9_G0, prtLightSH9_G1;
        glm::vec4 prtLightSH9_B0, prtLightSH9_B1;
        // SH16（l=3 追加 7 系数）
        glm::vec4 prtLightSH16_R0, prtLightSH16_R1;
        glm::vec4 prtLightSH16_G0, prtLightSH16_G1;
        glm::vec4 prtLightSH16_B0, prtLightSH16_B1;
    };
    CachedSH m_cachedSH;

    // ---- Per-frame temp state ----
    RenderFrame m_frameCtx;

    // ---- Benchmark ----
    BenchmarkRunner m_benchmark;

    // ---- Screenshot ----
    ScreenshotCapture m_screenshot;
    bool m_exitAfterCapture = false;  // --exit-after-capture CLI

    // ---- Backend ----
    std::string m_backendName = "Vulkan";
    // D3D12 设备（仅 D3D12 路径使用，Vulkan 路径为 nullptr）
    class rhi::RHIDevice* m_d3d12Device = nullptr;

    // ---- Regression test ----
    bool   m_captureRef     = false;  // --capture-ref：生成参考图到 tests/ref/
    bool   m_captureCompare = false;  // --capture-compare：截帧 + 对比回归测试
    double m_refThreshold   = 40.0;   // PSNR 阈值（dB）

    // ---- Stats ----
    float m_fpsAvg = 0.0f;
    float m_dtMs = 0.0f;

    // ---- Scene persistence ----
    std::map<std::string, SceneState> m_sceneStates;

    // ---- Methods ----
    void buildUI();
    void applyGiSelection();
    void applyShadowSelection();
    void bakePrt();
    void bakeEnvIbl();
    void bootstrapHdrPrev();
    void bootstrapSsgiTemporal();
    void rebuildDemoLights();

    void buildPipelineTable();
    void setupFrameGraph();
    void setupFgImports(VkExtent3D ext, bool aaEnabled);
    // Frame loop helpers (extracted from run())
    void buildFrameUBO(FrameUBO& ubo);
    void recordIndirectDraws(VkCommandBuffer cmd, uint32_t frameInFlight, const glm::mat4& viewProj);
    void recordPostProcessing(VkCommandBuffer cmd);
    void renderDebugWindow();

    void startBenchmark();
    void tickBenchmark(float dt);
    void applyBenchSettings();

    static constexpr uint32_t kMaxTextures = 128;
};

} // namespace somegi
