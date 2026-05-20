#pragma once
#include "core/vk_common.h"
#include "scene/scene.h"
#include "scene/camera.h"
#include "gi/ibl_baker.h"
#include "renderer/render_targets.h"
#include "renderer/forward_pass.h"      // for FrameUBO type
#include "renderer/gbuffer_pass.h"
#include "renderer/rsm_geometry_pass.h"
#include "renderer/rsm_sample_pass.h"
#include "renderer/lpv_grid.h"
#include "renderer/lpv_inject_pass.h"
#include "renderer/lpv_propagate_pass.h"
#include "renderer/vxgi_resources.h"
#include "renderer/vxgi_voxelize_pass.h"
#include "renderer/vxgi_inject_pass.h"
#include "renderer/vxgi_mipmap_pass.h"
#include "renderer/vxgi_aniso_pass.h"
#include "renderer/vxgi_relight_pass.h"
#include "renderer/vxgi_resolve_6axis_pass.h"
#include "renderer/sdfgi_resources.h"
#include "renderer/sdfgi_pass.h"
#include "renderer/restir_resources.h"
#include "renderer/restir_pass.h"
#include "renderer/prt_resources.h"
#include "renderer/prt_bake_pass.h"
#include "renderer/ddgi_resources.h"
#include "renderer/ddgi_pass.h"
#include "renderer/lighting_pass.h"
#include "renderer/ssao_pass.h"
#include "renderer/gtao_pass.h"
#include "renderer/ssr_pass.h"
#include "renderer/ssgi_pass.h"
#include "renderer/gtgi_pass.h"
#include "renderer/skybox_pass.h"
#include "renderer/tonemap_pass.h"
#include "renderer/imgui_pass.h"
#include "renderer/scene_rt_as.h"
#include "renderer/rt_gi_pass.h"
#include "renderer/lumen_resources.h"
#include "renderer/lumen_probe_pass.h"
#include "renderer/lumen_gather_pass.h"
#include "renderer/lumen_filter_pass.h"
#include <map>
#include <memory>
#include <filesystem>
#include <string>

namespace somegi {

class Window;
class Device;
class Swapchain;
class IGITechnique;

// Per-scene persisted view + lighting. Defaults match the old hard-coded
// global lighting; defaults() is what we use the first time a scene is
// entered (no entry in scene_state.ini yet).
struct SceneState {
    glm::vec3 camPos{0};
    float yaw = 0, pitch = 0, fov = 60.0f;
    glm::vec3 sunDir{-0.4f, -1.0f, -0.3f};
    float sunIntensity = 3.0f;
    glm::vec3 ambient{0.10f, 0.12f, 0.15f};
    bool camValid = false;   // false = use auto-framing for camera
};

class App {
public:
    App();
    ~App();
    void run();

private:
    void onSwapchainResized();
    void bootstrapHdrPrev();           // one-shot clear hdrPrev to zero, layout SHADER_READ_ONLY
    void bootstrapSsgiTemporal();      // B.4 一次性清 ssgi + ssgiPrev 为 0，layout SHADER_READ_ONLY
    void loadAndUploadScene(const std::filesystem::path& gltfPath);
    void applySceneSelection();        // honor m_currentSceneIndex changes from UI
    SceneState captureSceneState() const;
    void applyState(const SceneState& s);
    void persistAllStates();

    std::unique_ptr<Window> m_window;
    std::unique_ptr<Device> m_device;
    std::unique_ptr<Swapchain> m_swap;

    VkCommandPool m_pool = VK_NULL_HANDLE;
    VkCommandBuffer m_cmds[kFramesInFlight]{};

    // A.2: GPU 时间戳。每帧 2 个 query (TOP + BOTTOM)；下一次同 in-flight
    // 帧消费上次的结果。timestampPeriod 来自 device limits。
    VkQueryPool m_timestampPool = VK_NULL_HANDLE;
    bool m_timestampValid[kFramesInFlight]{};
    float m_gpuMs = 0.0f;       // 平滑后的最近 GPU 帧时

    SceneCpu m_scene;
    SceneGpu m_sceneGpu;
    Camera m_camera;
    FlyController m_flyer;

    RenderTargets m_rt;
    GBufferPass m_gbuffer;     // M4: writes albedo/normal/roughness/etc.
    RsmGeometryPass m_rsmGeom; // M5.0: sun-view 4-RT (pos/normal/flux/depth) for RSM
    RsmSamplePass m_rsmSample; // M5.1: per-pixel disk gather over RSM → rsmGI
    LpvResources m_lpv;        // M6: 32³ ping-pong SH grid (R/G/B 各 4 系数)
    LpvInjectPass m_lpvInject; // M6.0: RSM → LPV grid 注入
    LpvPropagatePass m_lpvProp;// M6.1: 6 邻居 SH 转移
    VxgiResources m_vxgi;            // M7: 128³ voxel grid + mipchain
    VxgiVoxelizePass m_vxgiVoxelize; // M7.0: 三角形 scatter 体素化
    VxgiInjectPass m_vxgiInject;     // M7.1: RSM → voxel RGB 注入
    VxgiMipmapPass m_vxgiMipmap;     // M7.2: 各级 mip 下采样
    VxgiAnisoPass m_vxgiAniso;       // B.6: 各向异性 alpha mipchain
    VxgiRelightPass m_vxgiRelight;   // C.2: multi-bounce voxel relight (Lumen-lite)
    VxgiResolve6AxisPass m_vxgiResolve6Axis;  // L.3b 6-axis resolve
    bool m_vxgiRelightEnabled = false;
    bool m_vxgiSixAxisInited  = false;
    float m_vxgiRelightStrength = 1.0f;

    // C.3 SDFGI-lite：JFA 构 UDF + sphere-trace。kGis[9] = SDFGI。
    SdfgiResources m_sdfgi;
    SdfgiPass      m_sdfgiPass;
    bool           m_sdfgiBootstrapped = false;
    static constexpr uint32_t kSdfgiResolution = 128;

    // C.4 ReSTIR DI 软件版：CPU 维护一组 demo point lights，每帧上传。
    RestirResources              m_restir;
    RestirPass                   m_restirPass;
    bool                         m_restirBootstrapped = false;     // reservoirA/B layout init
    bool                         m_restirOutInited    = false;     // rt.restir 首次 transition
    std::vector<PointLightCpu>   m_demoLights;
    int                          m_demoLightCount     = 8;
    float                        m_demoLightIntensity = 8.0f;
    static constexpr uint32_t    kRestirMaxLights     = 64;
    void rebuildDemoLights();
    PrtResources m_prt;              // M8: 32³ visibility transfer SH4
    PrtBakePass m_prtBake;           // M8.1: 一次性烘焙
    DdgiResources m_ddgi;            // M11: 8×4×8 probe atlas + ray buffer
    DdgiPass m_ddgiPass;             // M11: per-frame update + blend
    LightingPass m_lighting;   // M4: deferred direct + IBL (compute)
    SsaoPass m_ssao;           // M4.1: AO compute, modulates IBL diffuse
    GtaoPass m_gtao;           // B.1: GTAO 替代版（与 SSAO 共用 rt.ssao 输出）

    // AO 方法选择：用户可在 ImGui 切换。两种方法共用 rt.ssao R8 输出，
    // 一帧只跑其中一个；None 走 clear-to-1.0 路径。
    enum class AOMethod : int { None = 0, SSAO = 1, GTAO = 2 };
    AOMethod m_aoMethod = AOMethod::SSAO;
    SsrPass m_ssr;             // M4.2: screen-space reflections, replaces IBL specular
    SsgiPass m_ssgi;           // M4.3: screen-space 1-bounce diffuse, replaces IBL diffuse
    GtgiPass m_gtgi;           // C.1: GTGI（horizon-based GI，Sucker Punch 2024）
    SkyboxPass m_skybox;
    SceneRtAS m_rtAS;      // M9 HW RT acceleration structure (BLAS + TLAS)
    RtGiPass  m_rtGiPass;  // M9 HW RT GI compute pass
    TonemapPass m_tonemap;
    ImGuiPass m_imgui;

    // Pre-baked env from skybox.hdr — owned by App, lives across GI changes.
    // Used by SkyboxPass and (when active) borrowed by IBLTechnique.
    IblResources m_envIbl;
    void bakeEnvIbl();

    // GI technique (M3: IBL only)
    std::unique_ptr<IGITechnique> m_giTech;
    void applyGiSelection();           // honor m_currentGiIndex changes from UI

    // Stats / UI state
    float m_fpsAvg = 0.0f;
    float m_dtMs = 0.0f;
    int m_currentSceneIndex = 1;       // 0 = cube, 1 = Sponza (default)
    int m_sceneIndexApplied = -1;      // last scene actually loaded
    int m_currentGiIndex = 0;          // default = None
    int m_giIndexApplied = -1;         // last index actually attached

    // Per-scene persisted view + lighting (loaded from / written to scene_state.ini)
    std::map<std::string, SceneState> m_sceneStates;

    // Live lighting values (mirror of the active scene's SceneState fields)
    glm::vec3 m_sunDir{-0.4f, -1.0f, -0.3f};
    float m_sunIntensity = 3.0f;
    glm::vec3 m_ambient{0.10f, 0.12f, 0.15f};

    // M6 LPV：grid 几何参数（按 scene AABB 算）。M6.2 由 GI 下拉切换打开。
    bool      m_lpvEnabled = false;
    glm::vec3 m_lpvGridMin{0};
    float     m_lpvCellSize = 1.0f;
    static constexpr uint32_t kLpvResolution = 32;

    // M7 VXGI：128³ voxel grid 几何（同样按 scene AABB 算，但更细）。
    bool      m_vxgiEnabled = false;
    glm::vec3 m_vxgiGridMin{0};
    float     m_vxgiCellSize = 1.0f;
    static constexpr uint32_t kVxgiResolution = 128;

    // M8 PRT：32³ 体素 visibility transfer。bake 一次性，后续运行时不变。
    bool      m_prtEnabled = false;
    bool      m_prtBaked   = false;   // scene 切换后置 false 重 bake
    int       m_prtShOrder = 0;       // 0=SH4, 1=SH9, 2=SH16（驱动 prtCounts.z）
    glm::vec3 m_prtGridMin{0};
    float     m_prtCellSize = 1.0f;
    static constexpr uint32_t kPrtResolution = 32;
    void bakePrt();

    // M11 DDGI：probe grid 摆位 + 是否启用。
    bool      m_ddgiEnabled = false;
    bool      m_ddgiAtlasInited = false;   // 初次 transition 用
    glm::vec3 m_ddgiOrigin{0};
    glm::vec3 m_ddgiSpacing{1};
    uint32_t  m_frameIndex = 0;

    // M9 RT GI：硬件支持标志 + 初始化完成标志
    bool m_rtSupported = false;
    bool m_rtGiInited  = false;    // m_rtGiPass.init() 成功
    bool m_rtGiBound   = false;    // bindFrame 已调用

    // L.2 Lumen-lite：屏幕 probe atlas + ray buffer + probe pass
    LumenResources  m_lumen;
    LumenProbePass  m_lumenProbePass;
    LumenFilterPass m_lumenFilterPass;
    LumenGatherPass m_lumenGatherPass;
    bool            m_lumenEnabled       = false;
    bool            m_lumenAtlasInited   = false;
    bool            m_lumenProbeInited   = false;
    bool            m_lumenFilterInited  = false;
    bool            m_lumenGatherInited  = false;
    bool            m_lumenOutInited     = false;
    int             m_lumenDebugMode     = 0;   // 0=normal, 1=DC only, 2=probe color, 3=const radiance, 4=fixed SH, 5=clear only

    // B.4 SSGI 时序累积：保存上一帧 viewProj 用于 reproject。
    glm::mat4 m_prevViewProj{1.0f};

    void buildUI();

    static constexpr uint32_t kMaxTextures = 128;
};

}
