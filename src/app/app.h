#pragma once
#include "core/vk_common.h"
#include "scene/scene.h"
#include "scene/camera.h"
#include "scene/draw_list.h"
#include "renderer/core/forward_pass.h"      // for FrameUBO type
#include "renderer/core/frame_renderer.h"
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
    ~App();
    void run();

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
    VkCommandPool m_pool = VK_NULL_HANDLE;
    VkCommandBuffer m_cmds[kFramesInFlight]{};

    // ---- Scene + Camera ----
    SceneCpu m_scene;
    SceneGpu m_sceneGpu;
    Camera m_camera;
    FlyController m_flyer;

    // ---- Rendering (all passes owned by FrameRenderer) ----
    FrameRenderer m_renderer;
    Buffer m_indirectBuf;
    Buffer m_indirectBufSun;
    Buffer m_countBuf;
    uint32_t m_drawCount = 0;
    uint32_t m_culledDrawCount = 0;
    bool m_useGpuCulling = false;
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

    // TAA
    glm::vec2 m_jitter{};
    glm::vec2 m_prevJitter{};
    float m_taaBlendAlpha = 0.92f;
    glm::mat4 m_prevViewProj{1.0f};

    // ---- Per-frame temp state ----
    uint32_t m_currentFrameInFlight = 0;
    VkImageView m_currentSwapView = VK_NULL_HANDLE;
    VkImage     m_currentSwapImage = VK_NULL_HANDLE;
    VkExtent2D  m_currentSwapExtent{};
    glm::mat4   m_currentProj{1.0f};
    glm::mat4   m_currentView{1.0f};
    glm::mat4   m_currentInvViewProj{1.0f};

    // ---- Stats ----
    float m_fpsAvg = 0.0f;
    float m_dtMs = 0.0f;

    // ---- Scene persistence ----
    std::map<std::string, SceneState> m_sceneStates;

    // ---- Methods ----
    void buildUI();
    void applyGiSelection();
    void bakePrt();
    void bakeEnvIbl();
    void bootstrapHdrPrev();
    void bootstrapSsgiTemporal();
    void rebuildDemoLights();

    void registerPipelineSteps();
    void buildPipelineTable();
    void writeTimestamp(VkCommandBuffer cmd, uint32_t slot);

    void startBenchmark();
    void tickBenchmark(float dt);
    void applyBenchSettings();

    static constexpr uint32_t kMaxTextures = 128;
};

} // namespace somegi
