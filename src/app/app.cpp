#include "app.h"
#include "scene/draw_list.h"
#include "core/window.h"
#include "core/device.h"
#include "core/swapchain.h"
#include "scene/gltf_loader.h"
#include "scene/scene_gpu.h"
#include "scene/env_loader.h"
#include "scene/upload.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace somegi {

namespace {
struct SceneEntry { const char* name; const char* relPath; };
constexpr SceneEntry kScenes[] = {
    { "cube",           "gltf/cube/cube.gltf" },
    { "Sponza",         "gltf/Sponza/Sponza.gltf" },
    { "DamagedHelmet",  "gltf/DamagedHelmet/DamagedHelmet.gltf" },
    { "Bistro",         "gltf/Bistro/Bistro.gltf" },
};

// 用 implemented=false 占位的下拉项当 roadmap 用：选中后 applyGiSelection
// 把 effective 落到 IBL 兜底，但 dropdown 标签仍显示原 name + " (未实现)"。
struct GiEntry { const char* name; bool implemented; bool requiresRt = false; };
GiEntry kGis[] = {
    {"None (direct only)", true},
    {"IBL",                true},
    {"SSGI",               true},   // M4.3
    {"RSM",                true},   // M5
    {"LPV",                true},   // M6
    {"VXGI",               true},   // M7
    {"PRT",                true},   // M8
    {"DDGI",               true},   // M11（M9/M10 软件版兼并的实用替代）
    {"GTGI",               true},   // C.1 Sucker Punch 2024 horizon-based GI
    {"SDFGI",              true},   // C.3 Godot 4 风格 SDFGI-lite（JFA + sphere-trace）
    {"RayTracing",         false, true},  // M9 deferred (no HW RT on Intel UHD 770)
    {"ReSTIR DI",          true},   // C.4 软件版（reservoir resampling on point lights）
    {"Lumen-lite",         false, true},  // L 阶段：UE5 Lumen 简化复刻（Phase L1）
};
constexpr int kGiCount = (int)(sizeof(kGis)/sizeof(kGis[0]));

inline const char* giLabel(int i, char* buf, size_t bufSize) {
    if (i < 0 || i >= kGiCount) return "?";
    if (kGis[i].implemented && !kGis[i].requiresRt) return kGis[i].name;
    if (kGis[i].implemented && kGis[i].requiresRt) {
        std::snprintf(buf, bufSize, "%s (RT)", kGis[i].name);
        return buf;
    }
    std::snprintf(buf, bufSize, "%s (未实现)", kGis[i].name);
    return buf;
}

constexpr const char* kStatePath = "assets/scene_state.ini";

struct PersistedAll {
    std::map<std::string, SceneState> scenes;
    std::string lastScene;   // scene that was active at last shutdown
};

// Parse scene_state.ini — global `key value` lines before any section, then
// one [section] per scene with `key value` lines inside.
PersistedAll loadAllSceneStates() {
    PersistedAll out;
    std::ifstream f(kStatePath);
    if (!f) return out;
    std::string current;
    std::string line;
    while (std::getline(f, line)) {
        // strip leading/trailing whitespace
        size_t b = line.find_first_not_of(" \t\r");
        if (b == std::string::npos) continue;
        size_t e = line.find_last_not_of(" \t\r");
        std::string s = line.substr(b, e - b + 1);
        if (s.empty() || s[0] == '#') continue;
        if (s.front() == '[' && s.back() == ']') {
            current = s.substr(1, s.size() - 2);
            out.scenes[current];   // create empty entry for the section
            continue;
        }
        std::istringstream is(s);
        std::string key;
        if (!(is >> key)) continue;
        if (current.empty()) {
            // Pre-section global keys
            if (key == "last_scene") is >> out.lastScene;
            continue;
        }
        float v;
        if (!(is >> v)) continue;
        SceneState& st = out.scenes[current];
        if      (key == "cam.pos.x")     { st.camPos.x = v; st.camValid = true; }
        else if (key == "cam.pos.y")     { st.camPos.y = v; st.camValid = true; }
        else if (key == "cam.pos.z")     { st.camPos.z = v; st.camValid = true; }
        else if (key == "cam.yaw")       { st.yaw = v;      st.camValid = true; }
        else if (key == "cam.pitch")     { st.pitch = v;    st.camValid = true; }
        else if (key == "cam.fov")       { st.fov = v;      st.camValid = true; }
        else if (key == "sun.dir.x")       st.sunDir.x = v;
        else if (key == "sun.dir.y")       st.sunDir.y = v;
        else if (key == "sun.dir.z")       st.sunDir.z = v;
        else if (key == "sun.intensity")   st.sunIntensity = v;
        else if (key == "amb.x")           st.ambient.x = v;
        else if (key == "amb.y")           st.ambient.y = v;
        else if (key == "amb.z")           st.ambient.z = v;
        else if (key == "taa.blend")       st.taaBlendAlpha = v;
    }
    return out;
}

void saveAllSceneStates(const std::map<std::string, SceneState>& states,
                        const std::string& lastScene) {
    std::ofstream f(kStatePath);
    if (!f) return;
    if (!lastScene.empty()) f << "last_scene " << lastScene << "\n\n";
    for (const auto& [name, s] : states) {
        f << "[" << name << "]\n";
        if (s.camValid) {
            f << "cam.pos.x " << s.camPos.x << "\n";
            f << "cam.pos.y " << s.camPos.y << "\n";
            f << "cam.pos.z " << s.camPos.z << "\n";
            f << "cam.yaw "   << s.yaw << "\n";
            f << "cam.pitch " << s.pitch << "\n";
            f << "cam.fov "   << s.fov << "\n";
        }
        f << "sun.dir.x "     << s.sunDir.x << "\n";
        f << "sun.dir.y "     << s.sunDir.y << "\n";
        f << "sun.dir.z "     << s.sunDir.z << "\n";
        f << "sun.intensity " << s.sunIntensity << "\n";
        f << "amb.x "         << s.ambient.x << "\n";
        f << "amb.y "         << s.ambient.y << "\n";
        f << "amb.z "         << s.ambient.z << "\n";
        f << "taa.blend "     << s.taaBlendAlpha << "\n";
        f << "\n";
    }
}
}

App::App() {
    WindowDesc wd; wd.title = "SomeGI"; wd.width = 800; wd.height = 450;
    m_window = std::make_unique<Window>(wd);
    m_device = std::make_unique<Device>(*m_window, /*validation=*/true);

    try {
    // Clamp default MSAA to device-supported sample counts
    {
        VkSampleCountFlags supp = m_device->supportedSampleCounts();
        if (!(m_msaaSamples & supp)) {
            if (supp & VK_SAMPLE_COUNT_8_BIT)      m_msaaSamples = VK_SAMPLE_COUNT_8_BIT;
            else if (supp & VK_SAMPLE_COUNT_4_BIT) m_msaaSamples = VK_SAMPLE_COUNT_4_BIT;
            else if (supp & VK_SAMPLE_COUNT_2_BIT) m_msaaSamples = VK_SAMPLE_COUNT_2_BIT;
            else                                    m_msaaSamples = VK_SAMPLE_COUNT_1_BIT;
        }
    }

    m_swap   = std::make_unique<Swapchain>(*m_device, *m_window);

    // ImGui debug window (600x900)
    {
        WindowDesc iwd; iwd.title = "SomeGI Debug"; iwd.width = 600; iwd.height = 900;
        m_imguiWin = std::make_unique<Window>(iwd);
        m_imguiSwap = std::make_unique<Swapchain>(*m_device, *m_imguiWin);
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = m_device->graphicsQueueFamily();
        VK_CHECK(vkCreateCommandPool(m_device->device(), &pci, nullptr, &m_imguiPool));
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = m_imguiPool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = kFramesInFlight;
        VK_CHECK(vkAllocateCommandBuffers(m_device->device(), &ai, m_imguiCmds));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
        VK_CHECK(vkCreateFence(m_device->device(), &fci, nullptr, &m_imguiFence));
    }

    // M9：检测 HW RT 支持并更新 dropdown 实现状态。
    bool rtSupported = m_device->features().rayQuery && m_device->features().accelStruct;
    if (rtSupported) {
        kGis[10].implemented = true;
        kGis[12].implemented = true;   // Lumen-lite 同样依赖 HW RT
        std::printf("[init] HW RT supported — RayTracing GI + Lumen-lite enabled\n");
    } else {
        std::printf("[init] HW RT not supported — RayTracing GI + Lumen-lite unavailable\n");
    }

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = m_device->graphicsQueueFamily();
    VK_CHECK(vkCreateCommandPool(m_device->device(), &pci, nullptr, &m_pool));

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = m_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = kFramesInFlight;
    VK_CHECK(vkAllocateCommandBuffers(m_device->device(), &ai, m_cmds));

    // Delegate all rendering setup to FrameRenderer
    m_renderer.init(*m_device, m_pool, m_swap->extent(), m_msaaSamples,
                    rtSupported, m_swap->format(), m_window->handle());

    // 初始化 FrameGraph（实验性）
    m_fg.init(*m_device);
    m_fg.initTimestamps(*m_device, 32);  // GPU timestamp profiling: 最多 32 个 active pass

    // Init ImGui on the separate debug window
    m_renderer.imgui().init(*m_device, m_imguiWin->handle(), m_imguiSwap->format(), kFramesInFlight);

    // Register all rendering steps into the pipeline table
    registerPipelineSteps();

    // Load all scenes' persisted view + lighting before the first scene apply.
    {
        PersistedAll p = loadAllSceneStates();
        m_sceneStates = std::move(p.scenes);
        if (!p.lastScene.empty()) {
            constexpr int kSceneCount = (int)(sizeof(kScenes)/sizeof(kScenes[0]));
            for (int i = 0; i < kSceneCount; ++i) {
                if (p.lastScene == kScenes[i].name) { m_currentSceneIndex = i; break; }
            }
        }
    }

    // Initial scene load (camera framed inside applySceneSelection).
    applySceneSelection();

    // Bake skybox.hdr, then bind IBL resources to lighting pass
    std::printf("[init] env (skybox.hdr) load + bake...\n");
    bakeEnvIbl();
    std::printf("[init] env bake done.\n");
    m_renderer.skybox().bindEnv(*m_device, m_renderer.envIbl().envCube.view(),
                                m_renderer.envIbl().linear);
    m_renderer.lighting().bindIblResources(*m_device, m_renderer.envIbl());
    m_renderer.forward().bindIblResources(*m_device, m_renderer.envIbl());
    if (m_renderer.useMeshShader())
        m_renderer.forward().setMeshShaderEnabled(true);

    std::printf("[init] apply GI selection...\n");
    applyGiSelection();

    std::printf("[init] apply shadow selection...\n");
    applyShadowSelection();

    // Tonemap with scene sampler
    m_renderer.tonemap().init(*m_device, m_sceneGpu.linearSampler);
    m_renderer.tonemap().bindTargets(*m_device, m_renderer.rt());

    std::printf("[init] all set up, entering main loop.\n");
    } catch (...) {
        cleanup();
        throw;
    }
}

void App::bakeEnvIbl() {
    EnvCpu env;
    std::string err;
    auto hdrPath = std::filesystem::path(SOMEGI_ASSET_DIR) / "env" / "skybox.hdr";
    if (!loadHdrEquirect(hdrPath, env, err)) {
        std::fprintf(stderr, "[env] %s — using fallback sky\n", err.c_str());
        makeFallbackSky(env);
    }
    IblBaker baker;
    baker.bake(*m_device, m_pool, env, m_renderer.envIbl());
}

bool App::loadAndUploadScene(const std::filesystem::path& gltfPath, std::string& outErr) {
    if (!loadGltf(gltfPath, m_scene, outErr)) {
        return false;
    }
    std::printf("[scene] %s\n  vertices=%zu  indices=%zu  meshes=%zu  materials=%zu  textures=%zu\n  AABB min=(%.2f,%.2f,%.2f) max=(%.2f,%.2f,%.2f)\n",
        gltfPath.string().c_str(),
        m_scene.vertices.size(), m_scene.indices.size(),
        m_scene.meshes.size(), m_scene.materials.size(), m_scene.textures.size(),
        m_scene.aabbMin.x, m_scene.aabbMin.y, m_scene.aabbMin.z,
        m_scene.aabbMax.x, m_scene.aabbMax.y, m_scene.aabbMax.z);

    std::printf("[scene] uploading to GPU...\n");
    uploadScene(*m_device, m_pool, m_scene, m_sceneGpu, m_useMipmaps);
    std::printf("[scene] GPU upload done.\n");
    return true;
}

void App::applySceneSelection() {
    if (m_currentSceneIndex == m_sceneIndexApplied) return;
    constexpr int kSceneCount = (int)(sizeof(kScenes)/sizeof(kScenes[0]));
    if (m_currentSceneIndex < 0 || m_currentSceneIndex >= kSceneCount) {
        m_currentSceneIndex = m_sceneIndexApplied >= 0 ? m_sceneIndexApplied : 1;
        return;
    }

    // Remember which scene we came from, in case the new one fails to load.
    int      prevIndex    = m_sceneIndexApplied;
    SceneCpu prevSceneCpu = std::move(m_scene);
    SceneGpu prevSceneGpu = std::move(m_sceneGpu);

    if (prevIndex >= 0) {
        m_sceneStates[kScenes[prevIndex].name] = captureSceneState();
    }

    auto path = std::filesystem::path(SOMEGI_ASSET_DIR) / kScenes[m_currentSceneIndex].relPath;
    m_scene = SceneCpu{};
    m_sceneGpu = SceneGpu{};

    std::string loadErr;
    if (!loadAndUploadScene(path, loadErr)) {
        std::fprintf(stderr, "[scene] failed to load '%s': %s\n",
                     kScenes[m_currentSceneIndex].name, loadErr.c_str());
        m_sceneLoadError = std::string(kScenes[m_currentSceneIndex].name)
                         + " 加载失败：\n" + loadErr;

        // Restore previous scene
        if (prevIndex >= 0) {
            m_scene = std::move(prevSceneCpu);
            m_sceneGpu = std::move(prevSceneGpu);
            m_currentSceneIndex = prevIndex;
        } else {
            // First load failed — try each remaining scene until one works.
            bool recovered = false;
            for (int i = 0; i < kSceneCount; ++i) {
                if (i == m_currentSceneIndex) continue;
                auto fallbackPath = std::filesystem::path(SOMEGI_ASSET_DIR) / kScenes[i].relPath;
                if (loadAndUploadScene(fallbackPath, loadErr)) {
                    m_currentSceneIndex = i;
                    recovered = true;
                    break;
                }
            }
            if (!recovered) {
                throw std::runtime_error("no usable scene found: " + loadErr);
            }
        }
        ImGui::OpenPopup("Scene load failed");
    } else {
        if (prevIndex >= 0) {
            destroySceneSamplers(*m_device, prevSceneGpu);
        }
    }

    // M9：场景已加载，构建加速结构 + bindFrame（仅 HW 支持时）。
    if (m_renderer.rtSupported()) {
        m_renderer.rtAS().build(*m_device, m_pool, m_scene, m_sceneGpu);
        m_renderer.rtGiBound() = (m_renderer.rtAS().instanceCount() > 0);
        if (m_renderer.rtGiBound()) {
            m_renderer.rtGi().bindFrame(*m_device, m_renderer.rt(), m_renderer.gbuffer().frameUboHandle(), m_renderer.rtAS(), m_sceneGpu);
        }
        // 绑定 TLAS 到 ShadowPass（RT shadow 用）
        m_renderer.shadow().bindTLAS(*m_device, m_renderer.rtAS().tlas());
        // M10：TLAS 就绪，绑定到 ReSTIR RT shade pipeline
        if (m_renderer.rtAS().instanceCount() > 0) {
            m_renderer.restirPass().bindResourcesRt(*m_device, m_renderer.restir(), m_renderer.rt(),
                m_renderer.gbuffer().frameUboHandle(), m_renderer.rtAS().tlas());
        }
        m_renderer.ndgiPass().bindResources(*m_device, m_renderer.ndgi(), m_renderer.rtAS(), m_sceneGpu, m_renderer.rt(),
            m_renderer.gbuffer().frameUboHandle());
        // 如果 MLP 已经初始化过（从之前的场景），重新 init
        if (!m_renderer.ndgiInited()) {
            m_renderer.ndgiInited() = true;
            // initWeights 需要在 command buffer 中执行，延迟到下一帧 pipeline
        }
    }

    // GPU-driven: build draw list + indirect buffers
    {
        buildDrawList(m_scene, m_drawEntries);
        m_drawCount = (uint32_t)m_drawEntries.size();
        m_sceneGpu.drawCount = m_drawCount;
        std::printf("[scene] draw list: %u entries\n", m_drawCount);
        // 构建 mesh workgroup 映射（大 draw 拆分到多个 group）
        m_renderer.gbuffer().buildMeshGroups(m_drawEntries);
        m_renderer.forward().buildMeshGroups(m_drawEntries);
        m_sceneGpu.drawDataBuffer.reset();
        m_sceneGpu.drawDataBuffer = Buffer(*m_device, m_drawCount * sizeof(DrawEntry),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        std::memcpy(m_sceneGpu.drawDataBuffer.mapped(), m_drawEntries.data(), m_drawCount * sizeof(DrawEntry));
        m_indirectBuf.reset();
        m_indirectBuf = Buffer(*m_device, m_drawCount * sizeof(VkDrawIndexedIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        m_indirectBufSun.reset();
        m_indirectBufSun = Buffer(*m_device, m_drawCount * sizeof(VkDrawIndexedIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        m_countBuf.reset();
        m_countBuf = Buffer(*m_device, sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        m_renderer.gbuffer().bindDrawData(*m_device, m_sceneGpu.drawDataBuffer.handle());
        m_renderer.forward().bindDrawData(*m_device, m_sceneGpu.drawDataBuffer.handle());
        m_renderer.rsmGeom().bindDrawData(*m_device, m_sceneGpu.drawDataBuffer.handle());
    }

    m_renderer.gbuffer().bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size());
    m_renderer.forward().bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size());
    m_renderer.rsmGeom().bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size());
    m_renderer.vxgiVoxelize().bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size(), m_renderer.vxgi());

    // 绑定场景数据到阴影 pass
    m_renderer.shadow().bindScene(*m_device, m_sceneGpu);
    m_renderer.shadow().setSceneAabb(m_scene.aabbMin, m_scene.aabbMax);
    m_renderer.shadow().setSunDir(m_sunDir);
    m_renderer.lighting().bindShadowMask(*m_device, m_renderer.shadow().shadowMask().view());

    if (m_sceneIndexApplied >= 0) {
        // Tonemap pass cached the old sampler; old one was destroyed above.
        m_renderer.tonemap().destroy();
        m_renderer.tonemap().init(*m_device, m_sceneGpu.linearSampler);
        m_renderer.tonemap().bindTargets(*m_device, m_renderer.rt());
        if (m_aaMethod == AAMethod::TAA || m_aaMethod == AAMethod::SMAA) {
            m_renderer.rt().ensureAaResources(*m_device);
            m_renderer.taa().bindResources(*m_device, m_renderer.rt(), 0);
            m_renderer.smaa().bindResources(*m_device, m_renderer.rt());
        }
    }

    // Camera framing.
    // Interior-shaped scenes (much wider than tall, e.g. Sponza): stand inside
    // the AABB at one end of the longest horizontal axis, eye-height ~15% above
    // the floor, looking toward the far end.
    // Object-shaped scenes (roughly cubical, e.g. cube): stand outside the AABB
    // looking at its center — otherwise backface culling makes the geometry
    // invisible from inside.
    glm::vec3 c = (m_scene.aabbMin + m_scene.aabbMax) * 0.5f;
    glm::vec3 d = m_scene.aabbMax - m_scene.aabbMin;

    // GI 网格参数：由 FrameRenderer 根据 AABB 统一计算
    m_renderer.setupGiGrids(m_scene.aabbMin, m_scene.aabbMax);
    // ReSTIR demo lights
    m_renderer.rebuildDemoLights(m_scene);

    bool interiorLike = std::max(d.x, d.z) > d.y * 2.0f;
    if (interiorLike) {
        float eyeY = m_scene.aabbMin.y + d.y * 0.15f;
        if (d.x >= d.z) {
            m_camera.position = glm::vec3(m_scene.aabbMin.x + d.x * 0.05f, eyeY, c.z);
            m_camera.yaw = 0.0f;       // look toward +X
        } else {
            m_camera.position = glm::vec3(c.x, eyeY, m_scene.aabbMin.z + d.z * 0.05f);
            m_camera.yaw = 90.0f;      // look toward +Z
        }
    } else {
        float radius = glm::length(d) * 0.5f;
        float dist = std::max(1.0f, radius * 2.5f);
        m_camera.position = c + glm::vec3(0.0f, d.y * 0.3f, dist);
        m_camera.yaw = -90.0f;         // look toward -Z (back at center)
    }
    m_camera.pitch = 0.0f;
    m_camera.farZ = std::max(200.0f, glm::length(d) * 4.0f);
    // Scale near plane with scene size: a fixed 0.05 near with a ~10000 far on
    // the cube scene gives terrible z-precision; ~0.1% of the diagonal keeps
    // the far/near ratio bounded while staying well in front of any geometry.
    m_camera.nearZ = std::max(0.05f, glm::length(d) * 0.001f);

    // Scale fly-cam speed to scene size so movement is usable in both a 30-unit
    // Sponza and a 1400-unit cube. Shift still multiplies by 4×.
    m_flyer.moveSpeed = std::max(3.0f, glm::length(d) * 0.1f);

    // SSAO radius scales with scene: ~0.5% of the AABB diagonal works for
    // both Sponza (~40u → 0.2) and the scaled cube (~2400u → 12). Capped to
    // a min so tiny scenes still get a usable kernel.
    m_renderer.ssao().radius = std::max(0.1f, glm::length(d) * 0.005f);

    // SSR maxDist also scales with scene: about half the diagonal lets a
    // reflection ray cross the frame's worth of geometry. Cube ~1200,
    // Sponza ~20.
    m_renderer.ssr().maxDist = std::max(2.0f, glm::length(d) * 0.5f);

    // SSGI maxDist: shorter — diffuse contributions decay quickly with
    // distance, so a quarter of the diagonal is plenty. Reduces wasted
    // march steps. Cube ~600, Sponza ~10.
    m_renderer.ssgi().maxDist = std::max(1.0f, glm::length(d) * 0.25f);

    // Apply persisted state for this scene if present (camera + lighting).
    // First reset lighting to defaults so unsaved scenes don't inherit the
    // previous scene's tweaks.
    {
        SceneState defaults;
        m_sunDir = defaults.sunDir;
        m_sunIntensity = defaults.sunIntensity;
        m_ambient = defaults.ambient;
    }
    auto it = m_sceneStates.find(kScenes[m_currentSceneIndex].name);
    if (it != m_sceneStates.end()) {
        applyState(it->second);
        std::printf("[state] restored '%s' from %s\n",
                    kScenes[m_currentSceneIndex].name, kStatePath);
    }

    m_sceneIndexApplied = m_currentSceneIndex;
    std::printf("[scene] applied index=%d (%s)\n", m_sceneIndexApplied, kScenes[m_sceneIndexApplied].name);
    persistAllStates();   // persist with last_scene = the now-active scene
}

SceneState App::captureSceneState() const {
    SceneState s;
    s.camPos = m_camera.position;
    s.yaw = m_camera.yaw;
    s.pitch = m_camera.pitch;
    s.fov = m_camera.fovDeg;
    s.camValid = true;
    s.sunDir = m_sunDir;
    s.sunIntensity = m_sunIntensity;
    s.ambient = m_ambient;
    s.taaBlendAlpha = m_taaBlendAlpha;
    return s;
}

void App::applyState(const SceneState& s) {
    if (s.camValid) {
        m_camera.position = s.camPos;
        m_camera.yaw = s.yaw;
        m_camera.pitch = s.pitch;
        m_camera.fovDeg = s.fov;
    }
    m_sunDir = s.sunDir;
    m_sunIntensity = s.sunIntensity;
    m_ambient = s.ambient;
    m_taaBlendAlpha = s.taaBlendAlpha;
}

void App::persistAllStates() {
    constexpr int kSceneCount = (int)(sizeof(kScenes)/sizeof(kScenes[0]));
    std::string last;
    if (m_sceneIndexApplied >= 0 && m_sceneIndexApplied < kSceneCount) {
        last = kScenes[m_sceneIndexApplied].name;
    }
    saveAllSceneStates(m_sceneStates, last);
}

App::~App() {
    if (m_sceneIndexApplied >= 0) {
        m_sceneStates[kScenes[m_sceneIndexApplied].name] = captureSceneState();
        persistAllStates();
    }
    cleanup();
}

void App::cleanup() {
    // 统一由 FrameRenderer::destroy() 清理所有 render pass 资源
    m_renderer.destroy();
    // FrameGraph 资源池清理
    m_fg.destroy();
    // App 自有的 Vulkan 资源
    if (m_device) destroySceneSamplers(*m_device, m_sceneGpu);
    if (m_pool) vkDestroyCommandPool(m_device->device(), m_pool, nullptr);
    if (m_imguiFence) vkDestroyFence(m_device->device(), m_imguiFence, nullptr);
    if (m_imguiCmds[0]) vkFreeCommandBuffers(m_device->device(), m_imguiPool, kFramesInFlight, m_imguiCmds);
    if (m_imguiPool) vkDestroyCommandPool(m_device->device(), m_imguiPool, nullptr);
    m_imguiSwap.reset();
    m_imguiWin.reset();
}

void App::applyGiSelection() {
    // Effective index：用 kGis[i].implemented 兜底。未实现的下拉项
    // （LPV/VXGI/PRT/RT/ReSTIR）回落到 IBL，让画面有兜底而不是继承上
    // 一次的状态；dropdown 显示仍是用户的原始选择，不改 m_currentGiIndex。
    int effective = m_currentGiIndex;
    if (effective < 0 || effective >= kGiCount || !kGis[effective].implemented) {
        effective = 1;   // fallback = IBL
    }

    if (effective == m_giIndexApplied) return;

    m_pipelineDirty = true;  // GI 模式切换，下一帧重建管线执行表
    m_renderer.applyGiSelection(effective);

    m_giIndexApplied = effective;
    std::printf("[GI] applied technique index=%d (UI=%d)\n",
                m_giIndexApplied, m_currentGiIndex);
}

void App::applyShadowSelection() {
    if (m_currentShadowIndex == m_shadowIndexApplied) return;
    if (m_currentShadowIndex < 0 || m_currentShadowIndex >= kShadowCount) {
        m_currentShadowIndex = 1; return;
    }
    if (!kShadows[m_currentShadowIndex].implemented) {
        m_currentShadowIndex = 1; return;
    }
    // RT 阴影需要硬件支持，不可用时 fallback 到 PCF
    if (kShadows[m_currentShadowIndex].requiresRt && !m_renderer.rtSupported()) {
        m_currentShadowIndex = 2; return;
    }
    m_renderer.applyShadowSelection(m_currentShadowIndex);
    m_shadowIndexApplied = m_currentShadowIndex;
    std::printf("[Shadow] applied method index=%d (%s)\n",
                m_shadowIndexApplied, kShadows[m_shadowIndexApplied].name);
}

void App::startBenchmark() {
    auto applySettings = [this](int gi, int aa, int ao) {
        // GI
        m_currentGiIndex = gi;
        m_giIndexApplied = -1;
        applyGiSelection();
        // AA
        AAMethod newAa = (AAMethod)aa;
        if (newAa != m_aaMethod) {
            m_aaMethod = newAa;
            if (m_aaMethod == AAMethod::TAA || m_aaMethod == AAMethod::SMAA) {
                m_renderer.rt().ensureAaResources(*m_device);
                m_renderer.taa().bindResources(*m_device, m_renderer.rt(), 0);
                m_renderer.smaa().bindResources(*m_device, m_renderer.rt());
                m_renderer.aaHistoryNeedsInit() = true;
            } else {
                m_renderer.rt().destroyAaResources();
                m_renderer.tonemap().bindOutput(*m_device, m_renderer.rt().ldrTonemap.view(), 0);
            }
        }
        // AO
        m_aoMethod = (AOMethod)ao;
    };
    m_benchmark.start(applySettings);
}

void App::applyBenchSettings() { /* 逻辑已移入 BenchmarkRunner + startBenchmark 的 applySettings 回调 */ }

void App::tickBenchmark(float dt) {
    m_benchmark.tick(dt, m_renderer.gpuMs(), [this](int gi, int aa, int ao) {
        m_currentGiIndex = gi;
        m_giIndexApplied = -1;
        applyGiSelection();
        AAMethod newAa = (AAMethod)aa;
        if (newAa != m_aaMethod) {
            m_aaMethod = newAa;
            if (m_aaMethod == AAMethod::TAA || m_aaMethod == AAMethod::SMAA) {
                m_renderer.rt().ensureAaResources(*m_device);
                m_renderer.taa().bindResources(*m_device, m_renderer.rt(), 0);
                m_renderer.smaa().bindResources(*m_device, m_renderer.rt());
                m_renderer.aaHistoryNeedsInit() = true;
            } else {
                m_renderer.rt().destroyAaResources();
                m_renderer.tonemap().bindOutput(*m_device, m_renderer.rt().ldrTonemap.view(), 0);
            }
        }
        m_aoMethod = (AOMethod)ao;
    });
}

void App::onSwapchainResized() {
    m_renderer.rt().destroy();
    m_renderer.rt().create(*m_device, m_swap->extent(), m_msaaSamples);
    m_renderer.lighting().bindFrame(*m_device, m_renderer.rt(), m_renderer.gbuffer().frameUboHandle(),
                         m_renderer.lpv().current(), m_renderer.vxgi(), m_renderer.prt(), m_renderer.ddgi(),
                         m_renderer.ddgi().probeStates().handle());
    m_renderer.ssao().bindFrame(*m_device, m_renderer.rt());
    m_renderer.gtao().bindFrame(*m_device, m_renderer.rt(), m_renderer.gbuffer().frameUboHandle());
    m_renderer.ssr().bindFrame(*m_device, m_renderer.rt(), m_renderer.gbuffer().frameUboHandle());
    m_renderer.ssgi().bindFrame(*m_device, m_renderer.rt(), m_renderer.gbuffer().frameUboHandle());
    m_renderer.gtgi().bindFrame(*m_device, m_renderer.rt(), m_renderer.gbuffer().frameUboHandle());
    // SDFGI trace 也读 rt.gNormalRough/depth/ssgi → resize 后重绑。
    m_renderer.sdfgiPass().bindResources(*m_device, m_renderer.sdfgi(), m_renderer.vxgi(), m_renderer.rt(), m_renderer.gbuffer().frameUboHandle());
    // ReSTIR：reservoir image 跟 swapchain，需重建；lightBuffer 跨帧持久。
    m_renderer.restir().resize(*m_device, m_swap->extent());
    m_renderer.restirPass().bindResources(*m_device, m_renderer.restir(), m_renderer.vxgi(), m_renderer.rt(), m_renderer.gbuffer().frameUboHandle());
    m_renderer.restirOutInited() = false;
    m_renderer.restirBootstrapped() = false;
    m_renderer.rsmSample().bindFrame(*m_device, m_renderer.rt(),
        m_renderer.gbuffer().frameUboHandle(),
        m_renderer.rsmGeom().frameUboHandle(),
        m_renderer.rsmGeom().position(), m_renderer.rsmGeom().normal(), m_renderer.rsmGeom().flux());
    // M9：swapchain resize 后 rtGI 换新 view → 重绑 RT pass。
    if (m_renderer.rtSupported() && m_renderer.rtGiBound()) {
        m_renderer.rtGi().bindFrame(*m_device, m_renderer.rt(), m_renderer.gbuffer().frameUboHandle(), m_renderer.rtAS(), m_sceneGpu);
        // M10：resize 后重绑 ReSTIR RT shade 的 TLAS
        m_renderer.restirPass().bindResourcesRt(*m_device, m_renderer.restir(), m_renderer.rt(),
            m_renderer.gbuffer().frameUboHandle(), m_renderer.rtAS().tlas());
    }
    // L.2：swapchain resize 后 probe atlas 重建。
    if (m_renderer.rtSupported()) {
        m_renderer.lumen().destroy();
        m_renderer.lumen().create(*m_device, m_swap->extent());
        m_renderer.lumenAtlasInited() = false;
        m_renderer.lumenOutInited()  = false;
        if (m_renderer.lumenProbeInited()) {
            m_renderer.lumenProbe().bindResources(*m_device, m_renderer.lumen(), m_renderer.rtAS(), m_sceneGpu,
                                            m_renderer.vxgi(), m_renderer.rt(), m_renderer.gbuffer().frameUboHandle(),
                                            m_renderer.vxgiSixAxisInited());
        }
        if (m_renderer.lumenFilterInited()) {
            m_renderer.lumenFilter().bindResources(*m_device, m_renderer.lumen(), m_renderer.rt(),
                                             m_renderer.gbuffer().frameUboHandle());
        }
        if (m_renderer.lumenGatherInited()) {
            m_renderer.lumenGather().bindResources(*m_device, m_renderer.lumen(), m_renderer.rt(),
                                             m_renderer.gbuffer().frameUboHandle(), true);
        }
    }
    m_renderer.tonemap().bindTargets(*m_device, m_renderer.rt());
    // AA resources were destroyed by m_renderer.rt().destroy(), recreate if needed
    if (m_aaMethod == AAMethod::TAA || m_aaMethod == AAMethod::SMAA) {
        m_renderer.rt().ensureAaResources(*m_device);
        m_renderer.taa().bindResources(*m_device, m_renderer.rt(), 0);
        m_renderer.smaa().destroy();
        m_renderer.smaa().init(*m_device, m_swap->extent());
        m_renderer.smaa().bindResources(*m_device, m_renderer.rt());
    }
    m_renderer.bootstrapHdrPrev();   // fresh hdrPrev image — clear it before SSR can read
    m_renderer.bootstrapSsgiTemporal();
}

void App::rebuildDemoLights() {
    // demo lights：按 scene AABB 摆若干 point light。
    // 位置：4 个 ceiling 角 + 4 个 floor 角（默认 8 盏）。当 m_renderer.demoLightCount()
    // 改了，只取前 N 盏。颜色：HSV 环绕；intensity：m_renderer.demoLightIntensity()。
    m_renderer.demoLights().clear();
    glm::vec3 mn = m_scene.aabbMin;
    glm::vec3 mx = m_scene.aabbMax;
    if (glm::length(mx - mn) < 1e-3f) {
        // scene 还没 load，给一个 unit cube 的 fallback
        mn = glm::vec3(-2.0f); mx = glm::vec3(2.0f);
    }
    glm::vec3 d = mx - mn;
    float floorY  = mn.y + d.y * 0.10f;   // 略离地
    float ceilY   = mn.y + d.y * 0.85f;   // 略离顶
    glm::vec3 corners[8] = {
        {mn.x + d.x*0.20f, ceilY,  mn.z + d.z*0.20f},
        {mn.x + d.x*0.80f, ceilY,  mn.z + d.z*0.20f},
        {mn.x + d.x*0.20f, ceilY,  mn.z + d.z*0.80f},
        {mn.x + d.x*0.80f, ceilY,  mn.z + d.z*0.80f},
        {mn.x + d.x*0.30f, floorY, mn.z + d.z*0.50f},
        {mn.x + d.x*0.70f, floorY, mn.z + d.z*0.50f},
        {mn.x + d.x*0.50f, floorY, mn.z + d.z*0.30f},
        {mn.x + d.x*0.50f, floorY, mn.z + d.z*0.70f},
    };
    glm::vec3 colors[8] = {
        {1.0f, 0.4f, 0.4f},   // 红
        {0.4f, 1.0f, 0.4f},   // 绿
        {0.4f, 0.4f, 1.0f},   // 蓝
        {1.0f, 1.0f, 0.5f},   // 黄
        {1.0f, 0.5f, 1.0f},   // 紫
        {0.5f, 1.0f, 1.0f},   // 青
        {1.0f, 0.8f, 0.4f},   // 橙
        {0.8f, 0.4f, 1.0f},   // 玫红
    };
    int n = std::clamp(m_renderer.demoLightCount(), 0, 8);
    for (int i = 0; i < n; ++i) {
        PointLightCpu L{};
        L.pos = corners[i];
        L.radius = 0.0f;
        L.color = colors[i];
        L.intensity = m_renderer.demoLightIntensity();
        m_renderer.demoLights().push_back(L);
    }
}

void App::bakePrt() {
    // 流程：oneShot 内 voxelize 一次场景 → mipmap 一次 → prt_bake →
    // prtTransfer 转 SHADER_READ_ONLY 给 lighting 用。
    // voxel grid 的 layout 不保证留给 main loop 复用 —— main loop 自己
    // 每帧 reset（VXGI on 时 voxelize 重做；off 时 UNDEFINED→SHADER_READ_ONLY）。
    auto barrierAllVxgiMips = [&](VkCommandBuffer cmd, VkImageLayout oldL, VkImageLayout newL,
                                   VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                   VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
        b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
        b.oldLayout = oldL; b.newLayout = newL;
        b.image = m_renderer.vxgi().image().image();
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, m_renderer.vxgi().mipLevels(), 0, 1};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    };

    oneShotSubmit(*m_device, m_pool, [&](VkCommandBuffer cmd) {
        // 1. voxel grid all mips: UNDEFINED → TRANSFER_DST → clear → GENERAL
        barrierAllVxgiMips(cmd,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkClearColorValue zero{};
        VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT, 0, m_renderer.vxgi().mipLevels(), 0, 1};
        vkCmdClearColorImage(cmd, m_renderer.vxgi().image().image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &rg);
        barrierAllVxgiMips(cmd,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        // 2. voxelize：写 mip 0
        m_renderer.vxgiVoxelize().record(cmd, m_scene, m_sceneGpu,
            m_renderer.vxgiGridMin(), m_renderer.vxgiCellSize(), m_renderer.kVxgiResolution);

        // 3. mipmap：内部把 src mip 转 SHADER_READ_ONLY，dst 保持 GENERAL。
        m_renderer.vxgiMipmap().record(cmd, m_renderer.vxgi());

        // 4. mipmap 结束后状态：mip 0..mipLevels-2 已是 SHADER_READ_ONLY
        //    （mipmap 内部 barrier 转过），mip mipLevels-1 还在 GENERAL。
        //    只把最后一级转过去即可。
        {
            VkImageMemoryBarrier2 mb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            mb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            mb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            mb.image = m_renderer.vxgi().image().image();
            mb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                   m_renderer.vxgi().mipLevels() - 1, 1, 0, 1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cmd, &di);
        }

        // 5. prtTransfer A/B/C/D/E: UNDEFINED → GENERAL（bake 写 SH16 五张 atlas）
        VkImage prtImgs[5] = {m_renderer.prt().image().image(),
                              m_renderer.prt().imageB().image(),
                              m_renderer.prt().imageC().image(),
                              m_renderer.prt().imageD().image(),
                              m_renderer.prt().imageE().image()};
        VkImageMemoryBarrier2 pb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        pb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo pdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        pdi.imageMemoryBarrierCount = 1; pdi.pImageMemoryBarriers = &pb;
        for (auto img : prtImgs) {
            pb.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            pb.srcAccessMask = 0;
            pb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            pb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            pb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            pb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            pb.image = img;
            vkCmdPipelineBarrier2(cmd, &pdi);
        }

        // 6. prt_bake compute（写 SH16 16 系数 → 5 atlas）
        m_renderer.prtBake().record(cmd,
            m_renderer.prtGridMin(), m_renderer.prtCellSize(), m_renderer.kPrtResolution,
            m_renderer.vxgiGridMin(), m_renderer.vxgiCellSize(), m_renderer.kVxgiResolution,
            64);

        // 7. prtTransfer A/B/C/D/E: GENERAL → SHADER_READ_ONLY（lighting 读）
        for (auto img : prtImgs) {
            pb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            pb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            pb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            pb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            pb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            pb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            pb.image = img;
            vkCmdPipelineBarrier2(cmd, &pdi);
        }
    });
    std::printf("[PRT] bake complete (%u^3 cells, 64 rays/cell)\n", m_renderer.kPrtResolution);
}

void App::bootstrapSsgiTemporal() {
    // 把 ssgi 与 ssgiPrev 都清成 0 + 转 SHADER_READ_ONLY，让第一帧 SSGI on
    // 时 copy / sample 都有合法 layout。两张 image 同形 + 同初始内容。
    oneShotSubmit(*m_device, m_pool, [&](VkCommandBuffer cmd) {
        VkImage imgs[2] = {m_renderer.rt().ssgi.image(), m_renderer.rt().ssgiPrev.image()};
        for (auto img : imgs) {
            transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, img,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &r);
            transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });
}

void App::bootstrapHdrPrev() {
    oneShotSubmit(*m_device, m_pool, [&](VkCommandBuffer cmd) {
        transitionImage(cmd, m_renderer.rt().hdrPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkClearColorValue zero{};
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, m_renderer.rt().hdrPrev.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &r);
        transitionImage(cmd, m_renderer.rt().hdrPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    });
}

void App::buildUI() {
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::Begin("SomeGI Debug")) {
        ImGui::Text("Frame: %.2f ms (%.0f fps)  GPU: %.2f ms",
                    m_dtMs, m_fpsAvg, m_renderer.gpuMs());
        ImGui::SameLine(); ImGui::TextDisabled("  F2: benchmark");

        if (m_benchmark.running()) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1,1,0,1), "  [%d/156] benchmarking...",
                (int)m_benchmark.results().size());
        }

        if (ImGui::BeginTabBar("MainTabs")) {

        // ===== Tab 1: Scene =====
        if (ImGui::BeginTabItem("Scene")) {

        ImGui::Text("Scene");
        {
            const char* curName = (m_currentSceneIndex >= 0 &&
                                   m_currentSceneIndex < (int)(sizeof(kScenes)/sizeof(kScenes[0])))
                                  ? kScenes[m_currentSceneIndex].name : "?";
            if (ImGui::BeginCombo("glTF", curName)) {
                for (int i = 0; i < (int)(sizeof(kScenes)/sizeof(kScenes[0])); ++i) {
                    bool sel = (i == m_currentSceneIndex);
                    if (ImGui::Selectable(kScenes[i].name, sel)) m_currentSceneIndex = i;
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }
        ImGui::Text("  vertices: %zu", m_scene.vertices.size());
        ImGui::Text("  indices : %zu", m_scene.indices.size());
        ImGui::Text("  meshes  : %zu", m_scene.meshes.size());
        ImGui::Text("  mats    : %zu", m_scene.materials.size());
        ImGui::Text("  texs    : %zu", m_scene.textures.size());
        ImGui::Text("  AABB min: %.1f %.1f %.1f", m_scene.aabbMin.x, m_scene.aabbMin.y, m_scene.aabbMin.z);
        ImGui::Text("  AABB max: %.1f %.1f %.1f", m_scene.aabbMax.x, m_scene.aabbMax.y, m_scene.aabbMax.z);
        ImGui::Separator();

        ImGui::Text("Camera");
        ImGui::DragFloat3("position", &m_camera.position.x, 0.5f);
        ImGui::DragFloat("yaw", &m_camera.yaw, 0.5f);
        ImGui::DragFloat("pitch", &m_camera.pitch, 0.5f, -89.0f, 89.0f);
        ImGui::DragFloat("fov", &m_camera.fovDeg, 0.2f, 20.0f, 110.0f);
        glm::vec3 fwd = m_camera.forward();
        ImGui::Text("forward: %.2f %.2f %.2f", fwd.x, fwd.y, fwd.z);
        ImGui::Separator();

        ImGui::Text("Lighting");
        ImGui::DragFloat3("sun dir", &m_sunDir.x, 0.05f, -1.0f, 1.0f);
        ImGui::DragFloat("sun intensity", &m_sunIntensity, 0.05f, 0.0f, 20.0f);
        ImGui::ColorEdit3("ambient", &m_ambient.x);
        ImGui::Separator();

        ImGui::Separator();
        ImGui::Text("Shadow");
        {
            int idx = m_currentShadowIndex;
            if (idx < 0 || idx >= kShadowCount) idx = 0;
            if (ImGui::BeginCombo("Shadow Method", kShadows[idx].name)) {
                for (int i = 0; i < kShadowCount; ++i) {
                    if (!kShadows[i].implemented) continue;
                    if (kShadows[i].requiresRt && !m_renderer.rtSupported()) continue;
                    bool sel = (i == m_currentShadowIndex);
                    if (ImGui::Selectable(kShadows[i].name, sel)) m_currentShadowIndex = i;
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }
        // RT Soft 参数（仅在选中时显示）
        if (m_currentShadowIndex == (int)ShadowMethod::RTSoft && m_renderer.rtSupported()) {
            auto& sh = m_renderer.shadow();
            ImGui::SliderInt("RT samples", &sh.rtRayCount(), 4, 32);
            ImGui::SliderFloat("sun radius", &sh.rtSunRadius(), 0.01f, 0.10f, "%.3f rad");
        }
        ImGui::Separator();
        ImGui::Checkbox("GPU Frustum Culling", &m_useGpuCulling);
        ImGui::SameLine();
        if (m_useGpuCulling) {
            ImGui::TextColored(ImVec4(0.3f,1,0.3f,1), "ON");
            ImGui::Checkbox("Hi-Z Occlusion", &m_useHiZOcclusion);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(m_useHiZOcclusion?0.3f:0.6f, m_useHiZOcclusion?1.0f:0.6f, m_useHiZOcclusion?0.3f:0.6f, 1),
                m_useHiZOcclusion ? "ON" : "OFF");
        } else
            ImGui::TextDisabled("OFF (CPU Fill)");
        ImGui::Text("  Draws: %u total | indirect (%u culled)", m_drawCount, m_culledDrawCount);
        ImGui::TextDisabled("  3 indirect calls/frame");

        // Mesh Shader toggle（根据 GPU 实际能力显示可用特性）
        bool meshSupported = m_renderer.meshShaderSupported();
        bool taskSupported = m_renderer.taskShaderSupported();
        if (meshSupported || taskSupported) {
            bool useMs = m_renderer.useMeshShader();
            const char* label = taskSupported ? "Mesh Shader (Mesh + Task)" : "Mesh Shader (Mesh only)";
            if (ImGui::Checkbox(label, &useMs)) {
                m_renderer.setUseMeshShader(useMs);
                m_pipelineDirty = true;
            }
        } else {
            ImGui::TextDisabled("Mesh Shader (GPU not supported)");
        }

        ImGui::Text("GPU Profile");
        {
            uint32_t fi = 0; // show most recent frame's data
            float* ms = m_renderer.passTimes(fi);
            float maxMs = 0.02f; // min bar width
            for (uint32_t i = m_renderer.kTsGBuffer; i <= m_renderer.kTsEnd; ++i)
                if (ms[i] > maxMs) maxMs = ms[i];

            for (uint32_t i = m_renderer.kTsGBuffer; i <= m_renderer.kTsAA; ++i) {
                const char* name = m_renderer.passNames()[i];
                float t = ms[i];
                ImGui::Text("%-10s", name); ImGui::SameLine(80);
                ImGui::ProgressBar(t / maxMs, ImVec2(120, 0), "");
                ImGui::SameLine(); ImGui::Text("%5.2f ms", t);
            }
        }

        ImGui::Text("Mouse RMB: rotate  WASD/QE: move  Shift: fast");
        ImGui::EndTabItem();
        }

        // ===== Tab 2: Display =====
        if (ImGui::BeginTabItem("Display")) {

        ImGui::Text("MSAA");
        {
            VkSampleCountFlags supported = m_device->supportedSampleCounts();
            struct MsaaOption { const char* label; VkSampleCountFlagBits value; };
            MsaaOption all[] = {
                {"Off", VK_SAMPLE_COUNT_1_BIT},
                {"2x",  VK_SAMPLE_COUNT_2_BIT},
                {"4x",  VK_SAMPLE_COUNT_4_BIT},
                {"8x",  VK_SAMPLE_COUNT_8_BIT},
                {"16x", VK_SAMPLE_COUNT_16_BIT},
            };
            std::vector<MsaaOption> opts;
            int curIdx = -1;
            for (int i = 0; i < 5; ++i) {
                if ((all[i].value & supported) || all[i].value == VK_SAMPLE_COUNT_1_BIT) {
                    if (m_msaaSamples == all[i].value) curIdx = (int)opts.size();
                    opts.push_back(all[i]);
                }
            }
            // 如果当前 m_msaaSamples 不在支持列表中（例如从别的 GPU 迁移配置），
            // fallback 到最高支持的 sample count。
            if (curIdx < 0 && !opts.empty()) {
                m_msaaSamples = opts.back().value;
                curIdx = (int)opts.size() - 1;
            }
            if (ImGui::BeginCombo("MSAA samples", opts[curIdx].label)) {
                for (int i = 0; i < (int)opts.size(); ++i) {
                    bool sel = (curIdx == i);
                    if (ImGui::Selectable(opts[i].label, sel)) {
                        if (m_msaaSamples != opts[i].value) {
                            m_msaaSamples = opts[i].value;
                            m_renderer.rt().recreateMsaa(*m_device, m_msaaSamples);
                            m_renderer.gbuffer().setMsaaSamples(m_msaaSamples);
                        }
                    }
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            bool prevMip = m_useMipmaps;
            ImGui::Checkbox("Mipmap", &m_useMipmaps);
            if (m_useMipmaps != prevMip) {
                destroySceneSamplers(*m_device, m_sceneGpu);
                m_sceneGpu.vertexBuffer.reset();
                m_sceneGpu.indexBuffer.reset();
                m_sceneGpu.materialBuffer.reset();
                uploadScene(*m_device, m_pool, m_scene, m_sceneGpu, m_useMipmaps);
                m_renderer.gbuffer().bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size());
                m_renderer.rsmGeom().bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size());
                m_renderer.vxgiVoxelize().bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size(), m_renderer.vxgi());
            }
        }
        if (m_swap->hdrAvailable()) {
            bool hdrOn = m_swap->hdrEnabled();
            if (ImGui::Checkbox("HDR (scRGB)", &hdrOn)) {
                m_swap->setHdrEnabled(hdrOn);
            }
        }
        ImGui::Separator();

        ImGui::Text("Anti-Aliasing");
        {
            const char* items[] = {"None", "MSAA", "TAA", "SMAA"};
            int cur = (int)m_aaMethod;
            if (ImGui::BeginCombo("AA method", items[cur])) {
                for (int i = 0; i < 4; ++i) {
                    bool sel = (cur == i);
                    if (ImGui::Selectable(items[i], sel)) {
                        if (i != (int)m_aaMethod) {
                            m_aaMethod = (AAMethod)i;
                            if (m_aaMethod == AAMethod::TAA || m_aaMethod == AAMethod::SMAA) {
                                m_renderer.rt().ensureAaResources(*m_device);
                                m_renderer.taa().bindResources(*m_device, m_renderer.rt(), 0);
                                m_renderer.smaa().bindResources(*m_device, m_renderer.rt());
                            } else {
                                m_renderer.rt().destroyAaResources();
                                m_renderer.tonemap().bindOutput(*m_device, m_renderer.rt().ldrTonemap.view(), 0);
                            }
                        }
                    }
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }
        if (m_aaMethod == AAMethod::TAA) {
            ImGui::SliderFloat("TAA blend", &m_taaBlendAlpha, 0.5f, 0.98f, "%.3f");
        }
        ImGui::Separator();

        ImGui::Text("Ambient Occlusion");
        const char* aoLabels[] = {"None", "SSAO", "GTAO"};
        int aoIdx = (int)m_aoMethod;
        if (ImGui::BeginCombo("AO method", aoLabels[aoIdx])) {
            for (int i = 0; i < 3; ++i) {
                if (ImGui::Selectable(aoLabels[i], aoIdx == i)) { m_aoMethod = (AOMethod)i; m_pipelineDirty = true; }
            }
            ImGui::EndCombo();
        }
        if (m_aoMethod == AOMethod::SSAO) {
            ImGui::DragFloat("SSAO radius",  &m_renderer.ssao().radius, 0.05f, 0.05f, 100.0f, "%.3f");
            ImGui::DragFloat("SSAO bias",    &m_renderer.ssao().bias,   0.005f, 0.0f, 0.5f);
            ImGui::SliderInt("SSAO samples", &m_renderer.ssao().sampleCount, 4, 64);
        } else if (m_aoMethod == AOMethod::GTAO) {
            ImGui::SliderInt("GTAO slices",   &m_renderer.gtao().sliceCount, 1, 8);
            ImGui::SliderInt("GTAO samples/slice", &m_renderer.gtao().samplesPerSlice, 2, 16);
            ImGui::DragFloat("GTAO radius (px)", &m_renderer.gtao().radiusPixels, 1.0f, 4.0f, 256.0f);
            ImGui::DragFloat("GTAO falloff", &m_renderer.gtao().falloff, 0.1f, 0.5f, 50.0f);
        }

        ImGui::Separator();
        {
            const char* modeLabels[] = {"Deferred", "Forward"};
            int modeIdx = (int)m_renderingMode;
            if (ImGui::BeginCombo("Rendering", modeLabels[modeIdx])) {
                for (int i = 0; i < 2; ++i) {
                    bool sel = (modeIdx == i);
                    if (ImGui::Selectable(modeLabels[i], sel)) {
                        if ((RenderingMode)i != m_renderingMode) {
                            m_renderingMode = (RenderingMode)i;
                            m_pipelineDirty = true;
                        }
                    }
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }

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

            auto& fgDebug = m_fg.debug();
            if (ImGui::TreeNode("FrameGraph Debug")) {
                auto& compiled = m_fg.compiledGraph();
                ImGui::Text("Passes: %u active / %zu total | Resources: %zu | Alias Groups: %zu",
                    (uint32_t)compiled.passOrder.size(), fgDebug.passes.size(),
                    fgDebug.resources.size(), fgDebug.aliasGroups.size());

                // ---- Pass 执行列表 ----
                if (ImGui::CollapsingHeader("Pass List", ImGuiTreeNodeFlags_DefaultOpen)) {
                    if (ImGui::BeginTable("##fgpasstable", 7,
                            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit)) {
                        ImGui::TableSetupColumn("#");
                        ImGui::TableSetupColumn("Name");
                        ImGui::TableSetupColumn("T");
                        ImGui::TableSetupColumn("Reads");
                        ImGui::TableSetupColumn("Writes");
                        ImGui::TableSetupColumn("Deps");
                        ImGui::TableSetupColumn("GPU ms");
                        ImGui::TableHeadersRow();

                        for (auto& p : fgDebug.passes) {
                            ImGui::TableNextRow();
                            if (p.culled) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, 0x22AA2222);

                            ImGui::TableNextColumn();
                            if (!p.culled) ImGui::Text("%u", p.execOrder);
                            else ImGui::TextColored(ImVec4(1,0.3f,0.3f,1), "[X]");

                            ImGui::TableNextColumn();
                            ImGui::Text("%s", p.name.c_str());

                            ImGui::TableNextColumn();
                            const char* t = "?";
                            switch (p.passType) {
                                case somegi::fg::FGPassType::Compute: t = "C"; break;
                                case somegi::fg::FGPassType::Graphics: t = "G"; break;
                                case somegi::fg::FGPassType::MeshShading: t = "M"; break;
                                case somegi::fg::FGPassType::RayTracing: t = "R"; break;
                            }
                            ImGui::Text("%s", t);

                            ImGui::TableNextColumn();
                            ImGui::Text("%zu", p.reads.size());
                            if (ImGui::IsItemHovered()) {
                                ImGui::BeginTooltip();
                                for (auto& r : p.reads) ImGui::BulletText("%s", r.c_str());
                                ImGui::EndTooltip();
                            }

                            ImGui::TableNextColumn();
                            ImGui::Text("%zu", p.writes.size());
                            if (ImGui::IsItemHovered()) {
                                ImGui::BeginTooltip();
                                for (auto& w : p.writes) ImGui::BulletText("%s", w.c_str());
                                ImGui::EndTooltip();
                            }

                            ImGui::TableNextColumn();
                            ImGui::Text("%zu", p.deps.size());
                            if (ImGui::IsItemHovered() && !p.deps.empty()) {
                                ImGui::BeginTooltip();
                                for (auto& d : p.deps) ImGui::BulletText("%s", d.c_str());
                                ImGui::EndTooltip();
                            }

                            ImGui::TableNextColumn();
                            if (p.gpuMs > 0.001f) ImGui::Text("%.3f", p.gpuMs);
                            else ImGui::TextDisabled("-");
                        }
                        ImGui::EndTable();
                    }
                }

                // ---- 资源寿命图 ----
                if (ImGui::CollapsingHeader("Resources")) {
                    if (ImGui::BeginTable("##fgrestable", 6,
                            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
                        ImGui::TableSetupColumn("Name");
                        ImGui::TableSetupColumn("Type");
                        ImGui::TableSetupColumn("Size");
                        ImGui::TableSetupColumn("First");
                        ImGui::TableSetupColumn("Last");
                        ImGui::TableSetupColumn("Imported");
                        ImGui::TableHeadersRow();

                        uint32_t maxPass = 0;
                        for (auto& p : fgDebug.passes)
                            if (!p.culled && p.execOrder > maxPass) maxPass = p.execOrder;

                        for (auto& r : fgDebug.resources) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn(); ImGui::Text("%s", r.name.c_str());
                            ImGui::TableNextColumn();
                            ImGui::Text("%s", r.type == somegi::fg::FGResourceType::Texture ? "Tex" : "Buf");
                            ImGui::TableNextColumn();
                            if (r.sizeBytes >= 1024*1024) ImGui::Text("%.1f MB", r.sizeBytes / (1024.0f*1024.0f));
                            else ImGui::Text("%u KB", r.sizeBytes / 1024);
                            ImGui::TableNextColumn();
                            ImGui::Text("%u", r.firstWritePass);
                            ImGui::TableNextColumn();
                            ImGui::Text("%u", r.lastReadPass);
                            ImGui::TableNextColumn();
                            ImGui::Text("%s", r.isImported ? "Y" : "");
                        }
                        ImGui::EndTable();
                    }
                }

                // ---- 别名组 ----
                if (!fgDebug.aliasGroups.empty() && ImGui::CollapsingHeader("Alias Groups")) {
                    for (auto& ag : fgDebug.aliasGroups) {
                        ImGui::Text("Group %u: %u KB (%u KB wasted)",
                            ag.id, ag.totalBytes / 1024, ag.wastedBytes / 1024);
                        for (auto& m : ag.members)
                            ImGui::BulletText("%s", m.c_str());
                    }
                } else if (ImGui::CollapsingHeader("Alias Groups")) {
                    ImGui::TextDisabled("No alias groups (all lifetimes overlap, or no managed resources)");
                }

                ImGui::TreePop();
            }
        }

        ImGui::EndTabItem();
        }

        // ===== Tab 3: Effects =====
        if (ImGui::BeginTabItem("Effects")) {

        ImGui::Text("SSR (Screen-Space Reflections)");
        ImGui::Checkbox("SSR enabled",     &m_renderer.ssr().enabled);
        ImGui::SliderInt("SSR steps",      &m_renderer.ssr().maxSteps, 8, 128);
        ImGui::DragFloat("SSR max dist",   &m_renderer.ssr().maxDist,  0.5f, 0.1f, 1000.0f);
        ImGui::DragFloat("SSR thickness",  &m_renderer.ssr().thickness, 0.005f, 0.001f, 0.5f);
        ImGui::DragFloat("SSR rough threshold", &m_renderer.ssr().roughThreshold, 0.01f, 0.0f, 1.0f);
        ImGui::Separator();

        ImGui::Text("SSGI (Screen-Space GI)");
        ImGui::Checkbox("SSGI enabled",    &m_renderer.ssgi().enabled);
        ImGui::SliderInt("SSGI samples",   &m_renderer.ssgi().sampleCount, 2, 32);
        ImGui::SliderInt("SSGI steps",     &m_renderer.ssgi().maxSteps, 8, 64);
        ImGui::DragFloat("SSGI max dist",  &m_renderer.ssgi().maxDist,  0.2f, 0.1f, 500.0f);
        ImGui::DragFloat("SSGI thickness", &m_renderer.ssgi().thickness, 0.005f, 0.001f, 0.5f);
        ImGui::Separator();

        ImGui::Text("GTGI (Horizon-Based GI)");
        ImGui::Checkbox("GTGI enabled",   &m_renderer.gtgi().enabled);
        ImGui::SliderInt("GTGI slices",   &m_renderer.gtgi().sliceCount, 1, 8);
        ImGui::SliderInt("GTGI samples/slice", &m_renderer.gtgi().samplesPerSlice, 2, 16);
        ImGui::DragFloat("GTGI radius (px)", &m_renderer.gtgi().radiusPixels, 1.0f, 4.0f, 256.0f);
        ImGui::DragFloat("GTGI falloff", &m_renderer.gtgi().falloff, 0.1f, 0.5f, 50.0f);

        if (m_renderer.rtSupported()) {
            ImGui::Separator();
            ImGui::Text("RT GI (Hardware Ray Tracing)");
            ImGui::Text("RT GI %s (switch GI to 'RayTracing')",
                        m_giIndexApplied == 10 ? "active" : "off");
            if (m_renderer.rtAS().instanceCount() > 0) {
                ImGui::Text("TLAS instances: %u", m_renderer.rtAS().instanceCount());
            }
        }

        ImGui::EndTabItem();
        }

        // ===== Tab 4: GI =====
        if (ImGui::BeginTabItem("GI")) {

        ImGui::Text("GI Technique");
        {
            char curBuf[64];
            const char* curLabel = giLabel(m_currentGiIndex, curBuf, sizeof(curBuf));
            if (ImGui::BeginCombo("GI", curLabel)) {
                // Only show implemented options. Unimplemented entries stay
                // in kGis as a roadmap marker; flip their `implemented` flag
                // when the corresponding milestone lands (M5+).
                for (int i = 0; i < kGiCount; ++i) {
                    if (!kGis[i].implemented) continue;
                    char itemBuf[64];
                    const char* itemLabel = giLabel(i, itemBuf, sizeof(itemBuf));
                    bool sel = (i == m_currentGiIndex);
                    if (ImGui::Selectable(itemLabel, sel)) {
                        if (kGis[i].requiresRt && !m_renderer.rtSupported()) {
                            ImGui::OpenPopup("RT not supported");
                        } else {
                            m_currentGiIndex = i;
                        }
                    }
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }
        // IBL intensity slider — 拖动时即时写入 host-coherent UBO
        float iblI = m_renderer.lighting().iblIntensity();
        if (ImGui::SliderFloat("IBL intensity", &iblI, 0.0f, 4.0f)) {
            m_renderer.lighting().setIblIntensity(iblI);
        }
        if (m_renderer.envIbl().specularMipCount > 0)
            ImGui::Text("specular mips: %u", m_renderer.envIbl().specularMipCount);
        ImGui::Separator();

        ImGui::Text("RSM (Reflective Shadow Maps)");
        ImGui::Checkbox("RSM enabled",     &m_renderer.rsmSample().enabled);
        ImGui::SliderInt("RSM samples",    &m_renderer.rsmSample().sampleCount, 4, 128);
        ImGui::DragFloat("RSM radius",     &m_renderer.rsmSample().radius,    0.005f, 0.001f, 0.5f);
        ImGui::DragFloat("RSM intensity",  &m_renderer.rsmSample().intensity, 0.05f,  0.0f, 20.0f);
        ImGui::Separator();

        ImGui::Text("LPV (Light Propagation Volumes, 32^3)");
        ImGui::Checkbox("LPV enabled",     &m_renderer.lpvEnabled());
        ImGui::SliderInt("LPV iterations", &m_renderer.lpvProp().iterations, 0, 16);
        ImGui::DragFloat("LPV amplifier",  &m_renderer.lpvProp().occlusionAmplifier, 0.05f, 0.0f, 5.0f);
        ImGui::DragFloat("LPV GV occlusion", &m_renderer.lpvProp().gvOcclusionStrength, 0.05f, 0.0f, 5.0f);
        ImGui::Text("LPV cell=%.2f gridMin=(%.1f %.1f %.1f)",
                    m_renderer.lpvCellSize(), m_renderer.lpvGridMin().x, m_renderer.lpvGridMin().y, m_renderer.lpvGridMin().z);
        ImGui::Separator();

        ImGui::Text("VXGI (Voxel Cone Tracing, 128^3)");
        ImGui::Checkbox("VXGI enabled", &m_renderer.vxgiEnabled());
        ImGui::Text("VXGI cell=%.3f mipLevels=%u",
                    m_renderer.vxgiCellSize(), m_renderer.vxgi().mipLevels());
        ImGui::Checkbox("VXGI multi-bounce relight", &m_renderer.vxgiRelightEnabled());
        ImGui::SliderFloat("Relight bounce strength", &m_renderer.vxgiRelightStrength(),
                           0.0f, 4.0f, "%.2f");
        ImGui::Separator();

        ImGui::Text("DDGI (Dynamic Diffuse GI)");
        ImGui::Checkbox("DDGI enabled", &m_renderer.ddgiEnabled());
        if (m_renderer.rtSupported()) {
            ImGui::Checkbox("NDGI enabled (neural GI)", &m_renderer.ndgiEnabled());
        }
        ImGui::Text("spacing=(%.1f %.1f %.1f) origin=(%.1f %.1f %.1f)",
                    m_renderer.ddgiSpacing().x, m_renderer.ddgiSpacing().y, m_renderer.ddgiSpacing().z,
                    m_renderer.ddgiOrigin().x, m_renderer.ddgiOrigin().y, m_renderer.ddgiOrigin().z);
        ImGui::Separator();

        ImGui::Text("SDFGI (Signed Distance Field GI)");
        ImGui::Text("SDFGI %s (switch GI to 'SDFGI')",
                    m_renderer.sdfgiPass().enabled ? "active" : "off");
        ImGui::SliderInt("SDFGI rays", &m_renderer.sdfgiPass().numRays, 1, 16);
        ImGui::SliderInt("SDFGI maxSteps", &m_renderer.sdfgiPass().maxSteps, 8, 96);
        ImGui::DragFloat("SDFGI rayMax (cells)", &m_renderer.sdfgiPass().rayMaxCells, 1.0f, 8.0f, 256.0f);
        ImGui::DragFloat("SDFGI hitEps (cells)", &m_renderer.sdfgiPass().hitEpsCells, 0.05f, 0.1f, 2.0f);
        ImGui::DragFloat("SDFGI seedThr", &m_renderer.sdfgiPass().seedThreshold, 0.005f, 0.0f, 0.5f);
        ImGui::Separator();

        ImGui::Text("PRT (Precomputed Radiance Transfer, 32^3)");
        ImGui::Checkbox("PRT enabled", &m_renderer.prtEnabled());
        const char* shLabels[] = {
            "SH4  (order-1, 4 coefs)",
            "SH9  (order-2, 9 coefs)",
            "SH16 (order-3, 16 coefs)"
        };
        ImGui::Combo("PRT SH order", &m_renderer.prtShOrder(), shLabels, 3);
        if (m_renderer.prtShOrder() < 0) m_renderer.prtShOrder() = 0;
        if (m_renderer.prtShOrder() > 2) m_renderer.prtShOrder() = 2;
        ImGui::Text("PRT cell=%.2f baked=%s", m_renderer.prtCellSize(), m_renderer.prtBaked() ? "yes" : "no");
        if (ImGui::Button("Re-bake PRT")) m_renderer.prtBaked() = false;
        ImGui::Separator();

        bool restirUsingRt = m_renderer.rtSupported() && m_renderer.rtGiBound() && m_renderer.restirPass().enabled;
        ImGui::Text("ReSTIR DI (%s)",
                    restirUsingRt ? "HW RT visibility" : "voxel visibility");
        ImGui::Text("ReSTIR %s (switch GI to 'ReSTIR DI')",
                    m_renderer.restirPass().enabled ? "active" : "off");
        if (ImGui::SliderInt("ReSTIR demo lights", &m_renderer.demoLightCount(), 0, 8)) {
            m_renderer.rebuildDemoLights(m_scene);
        }
        if (ImGui::DragFloat("ReSTIR light intensity", &m_renderer.demoLightIntensity(), 0.1f, 0.0f, 50.0f)) {
            m_renderer.rebuildDemoLights(m_scene);
        }
        ImGui::SliderInt("ReSTIR M (candidates)", &m_renderer.restirPass().numCandidates, 1, 32);
        ImGui::SliderInt("ReSTIR K (neighbors)", &m_renderer.restirPass().numNeighbors, 0, 8);
        ImGui::DragFloat("ReSTIR spatial radius (px)", &m_renderer.restirPass().spatialRadius, 1.0f, 4.0f, 96.0f);
        ImGui::SliderInt("ReSTIR shadow steps", &m_renderer.restirPass().shadowSteps, 0, 16);
        ImGui::DragFloat("ReSTIR intensity scale", &m_renderer.restirPass().intensityScale, 0.05f, 0.0f, 8.0f);
        ImGui::Separator();

        ImGui::Text("Lumen-lite (Screen Probes)");
        ImGui::Text("Lumen %s (switch GI to 'Lumen-lite')",
                    m_renderer.lumenEnabled() ? "active" : "off");
        if (m_renderer.lumenEnabled()) {
            ImGui::Text("Probe grid: %d x %d (%d probes, %d rays each)",
                        m_renderer.lumen().probeGridW(), m_renderer.lumen().probeGridH(),
                        m_renderer.lumen().probeCount(), (int)LumenResources::kRaysPerProbe);
            ImGui::SliderFloat("Filter sigmaDepth", &m_renderer.lumenFilter().sigmaDepth,
                               0.01f, 1.0f, "%.3f");
            ImGui::SliderFloat("Filter normalPower", &m_renderer.lumenFilter().normalPower,
                               1.0f, 128.0f, "%.1f");
            ImGui::SliderFloat("Filter sigmaDist", &m_renderer.lumenFilter().sigmaDist,
                               1.0f, 500.0f, "%.1f");
            ImGui::SliderFloat("Temporal alpha", &m_renderer.lumenFilter().temporalAlpha,
                               0.0f, 0.98f, "%.2f");
            ImGui::Combo("Debug mode", &m_renderer.lumenDebugMode(),
                         "Normal\0SH DC only\0Probe colors\0Const radiance\0Fixed SH\0Clear only\0");
        }

        ImGui::EndTabItem();
        }

        // ===== Tab 5: Benchmark (only if data available) =====
        if (!m_benchmark.results().empty() && ImGui::BeginTabItem("Benchmark")) {

        static const char* kGiNames[] = {
            "None", "IBL", "SSGI", "RSM", "LPV", "VXGI", "PRT",
            "DDGI", "GTGI", "SDFGI", "RT GI", "ReSTIR", "Lumen"
        };
        static const char* kAaNames[] = {"None", "MSAA", "TAA", "SMAA"};

        // GI comparison: average FPS per GI technique (across AA/AO)
        ImGui::Text("FPS by GI Technique");
        {
            std::vector<float> fpsByGi(13, 0);
            std::vector<int>   cntByGi(13, 0);
            float maxFps = 1;
            for (auto& r : m_benchmark.results()) {
                fpsByGi[r.gi] += r.fps;
                cntByGi[r.gi]++;
            }
            for (int i = 0; i < 13; ++i) {
                if (cntByGi[i] > 0) { fpsByGi[i] /= (float)cntByGi[i]; maxFps = std::max(maxFps, fpsByGi[i]); }
            }
            for (int i = 0; i < 13; ++i) {
                if (cntByGi[i] == 0) continue;
                ImGui::Text("%-8s", kGiNames[i]); ImGui::SameLine(90);
                ImGui::ProgressBar(fpsByGi[i] / maxFps, ImVec2(120, 0), "");
                ImGui::SameLine(); ImGui::Text("%.0f", fpsByGi[i]);
            }
        }
        ImGui::Separator();

        // AA comparison: average FPS per AA method (across GI/AO)
        ImGui::Text("FPS by AA Method");
        {
            std::vector<float> fpsByAa(4, 0);
            std::vector<int>   cntByAa(4, 0);
            float maxFps = 1;
            for (auto& r : m_benchmark.results()) {
                fpsByAa[r.aa] += r.fps;
                cntByAa[r.aa]++;
            }
            for (int i = 0; i < 4; ++i) {
                if (cntByAa[i] > 0) { fpsByAa[i] /= (float)cntByAa[i]; maxFps = std::max(maxFps, fpsByAa[i]); }
            }
            for (int i = 0; i < 4; ++i) {
                if (cntByAa[i] == 0) continue;
                ImGui::Text("%-8s", kAaNames[i]); ImGui::SameLine(90);
                ImGui::ProgressBar(fpsByAa[i] / maxFps, ImVec2(120, 0), "");
                ImGui::SameLine(); ImGui::Text("%.0f", fpsByAa[i]);
            }
        }
        ImGui::Separator();

        // GPU time by GI (inverted: lower is better)
        ImGui::Text("GPU Time by GI Technique (ms)");
        {
            std::vector<float> gpuByGi(13, 0);
            std::vector<int>   cntByGi(13, 0);
            float maxGpu = 1;
            for (auto& r : m_benchmark.results()) {
                gpuByGi[r.gi] += r.gpuMs;
                cntByGi[r.gi]++;
            }
            for (int i = 0; i < 13; ++i) {
                if (cntByGi[i] > 0) { gpuByGi[i] /= (float)cntByGi[i]; maxGpu = std::max(maxGpu, gpuByGi[i]); }
            }
            for (int i = 0; i < 13; ++i) {
                if (cntByGi[i] == 0) continue;
                ImGui::Text("%-8s", kGiNames[i]); ImGui::SameLine(90);
                ImGui::ProgressBar(gpuByGi[i] / maxGpu, ImVec2(120, 0), "");
                ImGui::SameLine(); ImGui::Text("%.2f", gpuByGi[i]);
            }
        }

        ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
        }

        // Global modal popups (outside tab bar)
        if (ImGui::BeginPopupModal("RT not supported", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("This GI requires hardware Ray Tracing,\nnot supported on this device.");
            ImGui::Spacing();
            if (ImGui::Button("OK", ImVec2(120, 0))) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
        if (ImGui::BeginPopupModal("Scene load failed", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("%s", m_sceneLoadError.c_str());
            ImGui::Spacing();
            if (ImGui::Button("OK", ImVec2(120, 0))) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
    }
    ImGui::End();
    (void)io;
}

// ============================================================
// RenderPipeline 辅助方法
// ============================================================

void App::writeTimestamp(VkCommandBuffer cmd, uint32_t slot) {
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                         m_renderer.timestampPool(),
                         m_frameCtx.frameInFlight * m_renderer.kTimestampSlots + slot);
}

void App::buildPipelineTable() {
    // 仅在管线状态变化时重建执行表，避免每帧无效的 setEnabled + build
    if (!m_pipelineDirty) return;
    m_pipelineDirty = false;

    // 将 App 层状态打包传给 FrameRenderer，由其内部根据 GI/AO 标志完成 setEnabled
    m_renderer.buildPipelineTable({
        .forwardMode = (m_renderingMode == RenderingMode::Forward),
        .aoMethod    = (int)m_aoMethod,
        .giIndex     = m_giIndexApplied,
    });
}

void App::registerPipelineSteps() {
    m_renderer.pipeline().clear();

    // ============================
    // Phase 0: Shadow Pass（PrePass 最前面，生成 shadowMask）
    // ============================
    m_renderer.pipeline().addStep({
        .name = "Shadow",
        .phase = "PrePass",
        .record = [this](VkCommandBuffer cmd) {
            // 确保 depth 在 SHADER_READ_ONLY_OPTIMAL（shadow resolve 采样 GBuffer depth）
            transitionImage(cmd, m_renderer.rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            m_renderer.shadow().record(cmd, m_renderer.rt(),
                m_renderer.gbuffer().frameUboHandle(),
                m_sceneGpu, m_indirectBufSun.handle(), m_drawCount,
                m_renderer.frameIndex());
        }
    });

    // ============================
    // Phase 0: RSM 几何（sun-view MRT）
    // ============================
    m_renderer.pipeline().addStep({
        .name = "RSM-Geometry",
        .phase = "PrePass",
        .record = [this](VkCommandBuffer cmd) {
            m_renderer.rsmGeom().record(cmd, m_indirectBufSun.handle(), m_drawCount, m_sceneGpu);
        }
    });

    // ============================
    // Step 1 诊断：用 transfer clear 替代 vkCmdBeginRendering 写 hdrColor+depth
    // ============================
    m_renderer.pipeline().addStep({
        .name = "Forward",
        .phase = "PrePass",
        .enabled = false,
        .timestampSlot = m_renderer.kTsGBuffer,
        .record = [this](VkCommandBuffer cmd) {
            // hdrColor → COLOR_ATTACHMENT, depth → DEPTH_ATTACHMENT
            transitionImage(cmd, m_renderer.rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            transitionImage(cmd, m_renderer.rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            m_renderer.forward().record(cmd, m_renderer.rt(), m_indirectBuf.handle(), m_drawCount, m_sceneGpu);

            // hdrColor COLOR_ATTACHMENT → GENERAL（匹配延迟 Lighting 输出）
            transitionImage(cmd, m_renderer.rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            // depth DEPTH_ATTACHMENT → SR_O（匹配延迟 GBuffer 输出，供 Skybox 使用）
            transitionImage(cmd, m_renderer.rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

            writeTimestamp(cmd, m_renderer.kTsGBuffer);
            writeTimestamp(cmd, m_renderer.kTsAO);
            writeTimestamp(cmd, m_renderer.kTsVoxelGI);
            writeTimestamp(cmd, m_renderer.kTsLighting);
            writeTimestamp(cmd, m_renderer.kTsSkybox);
        }
    });

    // ============================
    // Phase 1: GBuffer prepass (graphics MRT with MSAA)
    // ============================
    m_renderer.pipeline().addStep({
        .name = "GBuffer",
        .phase = "PrePass",
        .timestampSlot = m_renderer.kTsGBuffer,
        .record = [this](VkCommandBuffer cmd) {
            // MSAA images → attachment layout
            auto toColorAttach = [&](VkImage img) {
                transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            };
            toColorAttach(m_renderer.rt().gAlbedoMetalMs.image());
            toColorAttach(m_renderer.rt().gNormalRoughMs.image());
            toColorAttach(m_renderer.rt().gEmissiveAOMs.image());
            transitionImage(cmd, m_renderer.rt().depthMs.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            // SS resolve targets → attachment layout
            toColorAttach(m_renderer.rt().gAlbedoMetal.image());
            toColorAttach(m_renderer.rt().gNormalRough.image());
            toColorAttach(m_renderer.rt().gEmissiveAO.image());
            transitionImage(cmd, m_renderer.rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            m_renderer.gbuffer().record(cmd, m_renderer.rt(), m_indirectBuf.handle(), m_drawCount, m_sceneGpu);

            // Resolved GBuffer → SHADER_READ_ONLY for downstream compute
            auto toSampled = [&](VkImage img) {
                transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            };
            toSampled(m_renderer.rt().gAlbedoMetal.image());
            toSampled(m_renderer.rt().gNormalRough.image());
            toSampled(m_renderer.rt().gEmissiveAO.image());
            transitionImage(cmd, m_renderer.rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

            writeTimestamp(cmd, m_renderer.kTsGBuffer);
        }
    });

    // ============================
    // Phase 1.5: AO (SSAO/GTAO/None, 互斥)
    // ============================
    m_renderer.pipeline().addStep({
        .name = "AO-SSAO",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_renderer.ssao().record(cmd, m_renderer.rt(),
                m_frameCtx.proj, glm::inverse(m_frameCtx.proj), m_frameCtx.view);
            transitionImage(cmd, m_renderer.rt().ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    m_renderer.pipeline().addStep({
        .name = "AO-GTAO",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_renderer.gtao().record(cmd, m_renderer.rt(),
                m_frameCtx.proj, m_frameCtx.view);
            transitionImage(cmd, m_renderer.rt().ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    m_renderer.pipeline().addStep({
        .name = "AO-Clear",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue white{};
            white.float32[0] = 1.0f; white.float32[1] = 1.0f;
            white.float32[2] = 1.0f; white.float32[3] = 1.0f;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_renderer.rt().ssao.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &range);
            transitionImage(cmd, m_renderer.rt().ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    // ============================
    // Phase 1.6: SSR
    // ============================
    m_renderer.pipeline().addStep({
        .name = "SSR",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().ssr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_renderer.ssr().record(cmd, m_renderer.rt());
            transitionImage(cmd, m_renderer.rt().ssr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    m_renderer.pipeline().addStep({
        .name = "SSR-Clear",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().ssr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_renderer.rt().ssr.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, m_renderer.rt().ssr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    // ============================
    // Phase 1.7: ScreenGI (SSGI/GTGI, 互斥)
    // ============================
    m_renderer.pipeline().addStep({
        .name = "ScreenGI",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            // Copy ssgi → ssgiPrev for temporal history
            transitionImage(cmd, m_renderer.rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            transitionImage(cmd, m_renderer.rt().ssgiPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.extent = {m_renderer.rt().extent.width, m_renderer.rt().extent.height, 1};
            vkCmdCopyImage(cmd,
                m_renderer.rt().ssgi.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_renderer.rt().ssgiPrev.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &region);

            // ssgiPrev → SHADER_READ_ONLY for sampling
            transitionImage(cmd, m_renderer.rt().ssgiPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            // ssgi → GENERAL for writing new value
            transitionImage(cmd, m_renderer.rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            if (m_renderer.ssgi().enabled) m_renderer.ssgi().record(cmd, m_renderer.rt());
            else                m_renderer.gtgi().record(cmd, m_renderer.rt());

            transitionImage(cmd, m_renderer.rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    m_renderer.pipeline().addStep({
        .name = "ScreenGI-Clear",
        .phase = "AO",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_renderer.rt().ssgi.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, m_renderer.rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    // === AO/SS 结束 timestamp ===
    m_renderer.pipeline().addStep({
        .name = "TS-AO",
        .phase = "AO",
        .record = [this](VkCommandBuffer cmd) {
            writeTimestamp(cmd, m_renderer.kTsAO);
        }
    });

    // ============================
    // Phase 1.83: VXGI voxelize → inject → mipmap → aniso → relight → 6axis
    // ============================
    m_renderer.pipeline().addStep({
        .name = "VXGI-Chain",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            // 1. Clear entire mip chain to 0
            auto barrierAllMips = [&](VkImageLayout oldL, VkImageLayout newL,
                                       VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                       VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
                b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
                b.oldLayout = oldL; b.newLayout = newL;
                b.image = m_renderer.vxgi().image().image();
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                      0, m_renderer.vxgi().mipLevels(), 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };
            barrierAllMips(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT,
                                       0, m_renderer.vxgi().mipLevels(), 0, 1};
            vkCmdClearColorImage(cmd, m_renderer.vxgi().image().image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &rg);
            barrierAllMips(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            // 2. Voxelize: scatter all primitives to mip 0
            m_renderer.vxgiVoxelize().record(cmd, m_scene, m_sceneGpu,
                m_renderer.vxgiGridMin(), m_renderer.vxgiCellSize(), m_renderer.kVxgiResolution);

            // 3. Inject: RSM flux → voxel mip 0 RGB
            {
                VkImageMemoryBarrier2 vbar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                vbar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                vbar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                vbar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                vbar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                vbar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                vbar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                vbar.image = m_renderer.vxgi().image().image();
                vbar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo vdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                vdi.imageMemoryBarrierCount = 1; vdi.pImageMemoryBarriers = &vbar;
                vkCmdPipelineBarrier2(cmd, &vdi);
            }
            m_renderer.vxgiInject().record(cmd, m_renderer.kVxgiResolution, m_renderer.vxgiGridMin(), m_renderer.vxgiCellSize());

            // 4. Mipmap: iterate src mip i → dst mip i+1
            m_renderer.vxgiMipmap().record(cmd, m_renderer.vxgi());

            // 5. Final mip → SHADER_READ_ONLY
            {
                VkImageMemoryBarrier2 fb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                fb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                fb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                fb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                fb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                fb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                fb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                fb.image = m_renderer.vxgi().image().image();
                fb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                       m_renderer.vxgi().mipLevels() - 1, 1, 0, 1};
                VkDependencyInfo fdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                fdi.imageMemoryBarrierCount = 1; fdi.pImageMemoryBarriers = &fb;
                vkCmdPipelineBarrier2(cmd, &fdi);
            }

            // 6. Aniso alpha mipchain: UNDEFINED → SHADER_READ_ONLY
            {
                VkImageMemoryBarrier2 ab{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                ab.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                ab.srcAccessMask = 0;
                ab.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                ab.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                ab.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                ab.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                ab.image = m_renderer.vxgi().aniso().image();
                ab.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                       0, m_renderer.vxgi().mipLevels(), 0, 1};
                VkDependencyInfo adi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                adi.imageMemoryBarrierCount = 1; adi.pImageMemoryBarriers = &ab;
                vkCmdPipelineBarrier2(cmd, &adi);
            }
            m_renderer.vxgiAniso().record(cmd, m_renderer.vxgi());
        }
    });

    // VXGI Relight (multi-bounce, within VXGI chain)
    m_renderer.pipeline().addStep({
        .name = "VXGI-Relight",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            int bounces = m_renderer.lumenEnabled() ? 3 : 1;

            auto transImg = [&](VkImage img, VkImageLayout oldL, VkImageLayout newL,
                                VkPipelineStageFlags2 srcS, VkAccessFlags2 srcA,
                                VkPipelineStageFlags2 dstS, VkAccessFlags2 dstA) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcS; b.srcAccessMask = srcA;
                b.dstStageMask = dstS; b.dstAccessMask = dstA;
                b.oldLayout = oldL; b.newLayout = newL;
                b.image = img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };

            auto blitScratchToVoxel = [&](VkImage srcImg) {
                transImg(m_renderer.vxgi().image().image(),
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT);
                transImg(srcImg,
                    VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_READ_BIT);
                VkImageCopy region{};
                region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.extent = {m_renderer.kVxgiResolution, m_renderer.kVxgiResolution, m_renderer.kVxgiResolution};
                vkCmdCopyImage(cmd,
                    srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    m_renderer.vxgi().image().image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &region);
                transImg(m_renderer.vxgi().image().image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            };

            // Bounce 1: read voxelGrid → write scratch
            transImg(m_renderer.vxgi().relightScratch().image(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_renderer.vxgiRelight().record(cmd, m_renderer.vxgiRelight().voxelSet(), m_renderer.kVxgiResolution,
                m_renderer.vxgi().mipLevels(), m_renderer.vxgiCellSize(), m_renderer.vxgiGridMin(),
                m_renderer.vxgiRelightStrength());

            if (bounces >= 2) {
                transImg(m_renderer.vxgi().relightScratch().image(),
                    VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                transImg(m_renderer.vxgi().relightScratch2().image(),
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                // Bounce 2: read scratch → write scratch2
                m_renderer.vxgiRelight().record(cmd, m_renderer.vxgiRelight().pingSet0(), m_renderer.kVxgiResolution,
                    m_renderer.vxgi().mipLevels(), m_renderer.vxgiCellSize(), m_renderer.vxgiGridMin(),
                    m_renderer.vxgiRelightStrength());

                if (bounces >= 3) {
                    transImg(m_renderer.vxgi().relightScratch2().image(),
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                    transImg(m_renderer.vxgi().relightScratch().image(),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                    // Bounce 3: read scratch2 → write scratch
                    m_renderer.vxgiRelight().record(cmd, m_renderer.vxgiRelight().pingSet1(), m_renderer.kVxgiResolution,
                        m_renderer.vxgi().mipLevels(), m_renderer.vxgiCellSize(), m_renderer.vxgiGridMin(),
                        m_renderer.vxgiRelightStrength());
                    blitScratchToVoxel(m_renderer.vxgi().relightScratch().image());
                } else {
                    blitScratchToVoxel(m_renderer.vxgi().relightScratch2().image());
                }
            } else {
                blitScratchToVoxel(m_renderer.vxgi().relightScratch().image());
            }
        }
    });

    // VXGI 6-axis resolve (Lumen mode only)
    m_renderer.pipeline().addStep({
        .name = "VXGI-6Axis",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            VkImageLayout axisOldL = m_renderer.lumenAtlasInited()
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            VkPipelineStageFlags2 axisSrcS = m_renderer.lumenAtlasInited()
                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            VkAccessFlags2 axisSrcA = m_renderer.lumenAtlasInited()
                ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0u;

            auto transAxisToGeneral = [&](VkImage img) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = axisSrcS; b.srcAccessMask = axisSrcA;
                b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                b.oldLayout = axisOldL;
                b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.image = img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };
            transAxisToGeneral(m_renderer.vxgi().sixAxisX().image());
            transAxisToGeneral(m_renderer.vxgi().sixAxisY().image());
            transAxisToGeneral(m_renderer.vxgi().sixAxisZ().image());

            m_renderer.vxgi6Axis().record(cmd, m_renderer.kVxgiResolution, m_renderer.vxgi().mipLevels(),
                m_renderer.vxgiCellSize(), m_renderer.vxgiGridMin(), m_renderer.vxgiRelightStrength());

            auto transAxisToSRO = [&](VkImage img) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b.image = img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };
            transAxisToSRO(m_renderer.vxgi().sixAxisX().image());
            transAxisToSRO(m_renderer.vxgi().sixAxisY().image());
            transAxisToSRO(m_renderer.vxgi().sixAxisZ().image());
        }
    });

    // VXGI bootstrap: when all consumers are off, transition voxel grid to SR_O
    m_renderer.pipeline().addStep({
        .name = "VXGI-Bootstrap",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b.srcAccessMask = 0;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.image = m_renderer.vxgi().image().image();
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                  0, m_renderer.vxgi().mipLevels(), 0, 1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &di);
            // aniso too
            b.image = m_renderer.vxgi().aniso().image();
            vkCmdPipelineBarrier2(cmd, &di);
        }
    });

    // ============================
    // Phase 1.835: SDFGI
    // ============================
    m_renderer.pipeline().addStep({
        .name = "SDFGI",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            if (!m_renderer.sdfgiBootstrapped()) {
                auto bootstrapToGeneral = [&](VkImage img) {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    b.srcAccessMask = 0;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.image = img;
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                };
                bootstrapToGeneral(m_renderer.sdfgi().seedA().image());
                bootstrapToGeneral(m_renderer.sdfgi().seedB().image());
                bootstrapToGeneral(m_renderer.sdfgi().udf().image());
                m_renderer.sdfgiBootstrapped() = true;
            }
            transitionImage(cmd, m_renderer.rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            m_renderer.sdfgiPass().record(cmd, m_renderer.sdfgi(), m_renderer.rt(), m_renderer.frameIndex(),
                m_renderer.sdfgiPass().seedThreshold, m_renderer.sdfgiPass().maxDistCells,
                (uint32_t)m_renderer.sdfgiPass().numRays,
                (uint32_t)m_renderer.sdfgiPass().maxSteps,
                m_renderer.sdfgiPass().rayMaxCells, m_renderer.sdfgiPass().hitEpsCells);

            transitionImage(cmd, m_renderer.rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    // ============================
    // Phase 1.836: RT GI
    // ============================
    m_renderer.pipeline().addStep({
        .name = "RTGI",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().rtGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_renderer.rtGi().record(cmd, m_renderer.rt());
            transitionImage(cmd, m_renderer.rt().rtGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    m_renderer.pipeline().addStep({
        .name = "RTGI-Clear",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().rtGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_renderer.rt().rtGI.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, m_renderer.rt().rtGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    // ============================
    // Phase 1.837: ReSTIR DI
    // ============================
    m_renderer.pipeline().addStep({
        .name = "ReSTIR",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            m_renderer.restir().updateLights(m_renderer.demoLights());
            if (!m_renderer.restirBootstrapped()) {
                auto bootstrapToGeneral = [&](VkImage img) {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    b.srcAccessMask = 0;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.image = img;
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                };
                bootstrapToGeneral(m_renderer.restir().reservoirA().image());
                bootstrapToGeneral(m_renderer.restir().reservoirB().image());
                m_renderer.restirBootstrapped() = true;
            }
            VkImageLayout restirOld = m_renderer.restirOutInited()
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            transitionImage(cmd, m_renderer.rt().restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                restirOld, VK_IMAGE_LAYOUT_GENERAL,
                m_renderer.restirOutInited() ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                   : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                m_renderer.restirOutInited() ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_renderer.restirOutInited() = true;

            uint32_t numLights = (uint32_t)m_renderer.demoLights().size();
            bool useRtVis = m_renderer.rtSupported() && m_renderer.rtGiBound();
            m_renderer.restirPass().record(cmd, m_renderer.restir(), m_renderer.rt(),
                numLights,
                (uint32_t)m_renderer.restirPass().numCandidates,
                (uint32_t)m_renderer.restirPass().numNeighbors,
                m_renderer.restirPass().spatialRadius,
                (uint32_t)m_renderer.restirPass().shadowSteps,
                m_renderer.restirPass().intensityScale,
                m_renderer.frameIndex(),
                useRtVis);

            transitionImage(cmd, m_renderer.rt().restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    m_renderer.pipeline().addStep({
        .name = "ReSTIR-Clear",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            VkImageLayout restirOld = m_renderer.restirOutInited()
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            transitionImage(cmd, m_renderer.rt().restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                restirOld, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                m_renderer.restirOutInited() ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                   : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                m_renderer.restirOutInited() ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_renderer.rt().restir.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, m_renderer.rt().restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            m_renderer.restirOutInited() = true;
        }
    });

    // ============================
    // Phase 1.84: DDGI
    // ============================
    m_renderer.pipeline().addStep({
        .name = "DDGI",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            auto barrierAtlas = [&](VkImage img,
                                    VkImageLayout oldL, VkImageLayout newL,
                                    VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                    VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
                b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
                b.oldLayout = oldL; b.newLayout = newL;
                b.image = img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };

            VkImageLayout oldAtlasL = m_renderer.ddgiAtlasInited()
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            VkAccessFlags2 srcAcc = m_renderer.ddgiAtlasInited()
                ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;
            VkPipelineStageFlags2 srcStg = m_renderer.ddgiAtlasInited()
                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;

            barrierAtlas(m_renderer.ddgi().irradiance().image(),
                oldAtlasL, VK_IMAGE_LAYOUT_GENERAL,
                srcStg, srcAcc,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            barrierAtlas(m_renderer.ddgi().distance().image(),
                oldAtlasL, VK_IMAGE_LAYOUT_GENERAL,
                srcStg, srcAcc,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            float jitterRot = float((m_renderer.frameIndex() % 360) * 0.0174532925);
            m_renderer.ddgiPass().record(cmd, m_renderer.ddgi(), m_renderer.ddgiOrigin(), m_renderer.ddgiSpacing(),
                m_renderer.vxgiGridMin(), m_renderer.vxgiCellSize(), m_renderer.kVxgiResolution,
                jitterRot, m_renderer.frameIndex());

            barrierAtlas(m_renderer.ddgi().irradiance().image(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            barrierAtlas(m_renderer.ddgi().distance().image(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            m_renderer.ddgiAtlasInited() = true;
        }
    });

    m_renderer.pipeline().addStep({
        .name = "DDGI-Bootstrap",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            auto barrierAtlas = [&](VkImage img,
                                    VkImageLayout oldL, VkImageLayout newL,
                                    VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                    VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
                b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
                b.oldLayout = oldL; b.newLayout = newL;
                b.image = img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };
            barrierAtlas(m_renderer.ddgi().irradiance().image(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            barrierAtlas(m_renderer.ddgi().distance().image(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            m_renderer.ddgiAtlasInited() = true;
        }
    });

    // ============================
    // Phase 1.845: Lumen-lite
    // ============================
    m_renderer.pipeline().addStep({
        .name = "Lumen-DebugClear",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            VkImageLayout oldL = m_renderer.lumenOutInited()
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            VkPipelineStageFlags2 srcS = m_renderer.lumenOutInited()
                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            VkAccessFlags2 srcA = m_renderer.lumenOutInited()
                ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0u;
            transitionImage(cmd, m_renderer.rt().lumenGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                oldL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                srcS, srcA,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue grey{};
            grey.float32[0] = 0.3f; grey.float32[1] = 0.3f;
            grey.float32[2] = 0.3f; grey.float32[3] = 1.0f;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_renderer.rt().lumenGI.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &grey, 1, &range);
            transitionImage(cmd, m_renderer.rt().lumenGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            m_renderer.lumenOutInited() = true;
        }
    });

    m_renderer.pipeline().addStep({
        .name = "Lumen-Probe",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            if (!m_renderer.lumenProbeInited()) {
                m_renderer.lumenProbe().init(*m_device);
                m_renderer.lumenProbe().bindResources(*m_device, m_renderer.lumen(), m_renderer.rtAS(), m_sceneGpu,
                                                m_renderer.vxgi(), m_renderer.rt(), m_renderer.gbuffer().frameUboHandle(),
                                                m_renderer.vxgiSixAxisInited());
                m_renderer.lumenProbeInited() = true;
            }
            // Transition probe + filtered atlas to GENERAL
            {
                VkImageLayout oldL = m_renderer.lumenAtlasInited()
                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    : VK_IMAGE_LAYOUT_UNDEFINED;
                VkPipelineStageFlags2 srcS = m_renderer.lumenAtlasInited()
                    ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                    : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                VkAccessFlags2 srcA = m_renderer.lumenAtlasInited()
                    ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0u;
                auto transToGeneral = [&](VkImage img) {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = srcS; b.srcAccessMask = srcA;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.oldLayout = oldL;
                    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.image = img;
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                };
                transToGeneral(m_renderer.lumen().probeAtlas().image());
                transToGeneral(m_renderer.lumen().filteredAtlas().image());
                m_renderer.lumenAtlasInited() = true;
            }
            m_renderer.lumenProbe().record(cmd, m_renderer.lumen(), m_renderer.frameIndex(),
                                     m_renderer.lumenDebugMode() >= 3 ? (uint32_t)m_renderer.lumenDebugMode() - 1u
                                                           : (m_renderer.vxgiSixAxisInited() ? 1u : 0u));

            // ProbeAtlas GENERAL → SR_O for filter
            {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b.image = m_renderer.lumen().probeAtlas().image();
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            }
        }
    });

    m_renderer.pipeline().addStep({
        .name = "Lumen-Filter",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            if (!m_renderer.lumenFilterInited()) {
                VkImageMemoryBarrier2 pb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                pb.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                pb.srcAccessMask = 0;
                pb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                pb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                pb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                pb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                pb.image = m_renderer.lumen().prevAtlas().image();
                pb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo pdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                pdi.imageMemoryBarrierCount = 1; pdi.pImageMemoryBarriers = &pb;
                vkCmdPipelineBarrier2(cmd, &pdi);

                m_renderer.lumenFilter().init(*m_device);
                m_renderer.lumenFilter().bindResources(*m_device, m_renderer.lumen(), m_renderer.rt(),
                                                 m_renderer.gbuffer().frameUboHandle());
                m_renderer.lumenFilterInited() = true;
            }
            m_renderer.lumenFilter().record(cmd, m_renderer.lumen(), m_renderer.rt());

            // Copy filteredAtlas → prevAtlas for next frame
            auto imgBarrier = [&](VkImage img, VkImageLayout oldL, VkImageLayout newL,
                                  VkPipelineStageFlags2 srcS, VkAccessFlags2 srcA,
                                  VkPipelineStageFlags2 dstS, VkAccessFlags2 dstA) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcS; b.srcAccessMask = srcA;
                b.dstStageMask = dstS; b.dstAccessMask = dstA;
                b.oldLayout = oldL; b.newLayout = newL;
                b.image = img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };

            imgBarrier(m_renderer.lumen().filteredAtlas().image(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            imgBarrier(m_renderer.lumen().prevAtlas().image(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.extent = {m_renderer.lumen().atlasWidth(), m_renderer.lumen().atlasHeight(), 1};
            vkCmdCopyImage(cmd,
                m_renderer.lumen().filteredAtlas().image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_renderer.lumen().prevAtlas().image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &region);

            imgBarrier(m_renderer.lumen().filteredAtlas().image(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            imgBarrier(m_renderer.lumen().prevAtlas().image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    m_renderer.pipeline().addStep({
        .name = "Lumen-Gather",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            if (!m_renderer.lumenGatherInited()) {
                m_renderer.lumenGather().init(*m_device);
                m_renderer.lumenGather().bindResources(*m_device, m_renderer.lumen(), m_renderer.rt(),
                                                 m_renderer.gbuffer().frameUboHandle(), true);
                m_renderer.lumenGatherInited() = true;
            }
            {
                VkImageLayout oldL = m_renderer.lumenOutInited()
                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    : VK_IMAGE_LAYOUT_UNDEFINED;
                VkAccessFlags2 srcA = m_renderer.lumenOutInited()
                    ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;
                VkPipelineStageFlags2 srcS = m_renderer.lumenOutInited()
                    ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                    : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcS; b.srcAccessMask = srcA;
                b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                b.oldLayout = oldL;
                b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.image = m_renderer.rt().lumenGI.image();
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            }
            m_renderer.lumenGather().record(cmd, m_renderer.lumen(), m_renderer.rt(),
                                     (uint32_t)m_renderer.lumenDebugMode());

            {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b.image = m_renderer.rt().lumenGI.image();
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            }
            m_renderer.lumenOutInited() = true;
        }
    });

    m_renderer.pipeline().addStep({
        .name = "Lumen-Clear",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            VkImageLayout oldL = m_renderer.lumenOutInited()
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            VkPipelineStageFlags2 srcS = m_renderer.lumenOutInited()
                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            VkAccessFlags2 srcA = m_renderer.lumenOutInited()
                ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;
            transitionImage(cmd, m_renderer.rt().lumenGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                oldL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                srcS, srcA,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_renderer.rt().lumenGI.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, m_renderer.rt().lumenGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            m_renderer.lumenOutInited() = true;
        }
    });

    // ============================
    // Phase 1.85: LPV inject + propagate
    // ============================
    m_renderer.pipeline().addStep({
        .name = "LPV",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            m_renderer.lpvInject().record(cmd, m_renderer.kLpvResolution, m_renderer.lpvGridMin(), m_renderer.lpvCellSize());

            transitionImage(cmd, m_renderer.lpv().gv().image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

            auto barrierLpv = [&](const LpvGrid& g,
                                  VkImageLayout oldL, VkImageLayout newL,
                                  VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                  VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                VkImage imgs[3] = {g.lpvR.image(), g.lpvG.image(), g.lpvB.image()};
                for (auto img : imgs) {
                    transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                        oldL, newL, srcStg, srcAcc, dstStg, dstAcc);
                }
            };

            int propIter = m_renderer.lpvProp().iterations & ~1;
            for (int it = 0; it < propIter; ++it) {
                LpvGrid& src = m_renderer.lpv().current();
                LpvGrid& dst = m_renderer.lpv().next();

                barrierLpv(src,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                barrierLpv(dst,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                m_renderer.lpvProp().record(cmd, m_renderer.lpv().curIdx(),
                                 m_renderer.kLpvResolution, m_renderer.lpvProp().occlusionAmplifier,
                                 m_renderer.lpvProp().gvOcclusionStrength);
                m_renderer.lpv().swap();
            }

            barrierLpv(m_renderer.lpv().current(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    m_renderer.pipeline().addStep({
        .name = "LPV-Bootstrap",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            VkImage imgs[3] = {m_renderer.lpv().current().lpvR.image(),
                               m_renderer.lpv().current().lpvG.image(),
                               m_renderer.lpv().current().lpvB.image()};
            for (auto img : imgs) {
                transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            }
        }
    });

    // ============================
    // Phase 1.8: RSM sample
    // ============================
    m_renderer.pipeline().addStep({
        .name = "RSM-Sample",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().rsmGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_renderer.rsmSample().record(cmd, m_renderer.rt());
            transitionImage(cmd, m_renderer.rt().rsmGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    m_renderer.pipeline().addStep({
        .name = "RSM-Clear",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().rsmGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_renderer.rt().rsmGI.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, m_renderer.rt().rsmGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    // ============================
    // NDGI (Neural Dynamic GI) —— 探针光线追踪收集训练样本
    // ============================
    m_renderer.pipeline().addStep({
        .name = "NDGI",
        .phase = "GI",
        .enabled = false,
        .record = [this](VkCommandBuffer cmd) {
            // 延迟到第一帧：此时 bindResources 已执行，descriptor 已就绪
            if (!m_renderer.ndgiInited()) {
                m_renderer.ndgiPass().initWeights(cmd);
                m_renderer.ndgiInited() = true;
            }
            m_renderer.ndgiPass().record(cmd, m_renderer.ndgi(), m_renderer.frameIndex(),
                m_renderer.ddgiOrigin(), m_renderer.ddgiSpacing());
            m_renderer.ndgiPass().recordTraining(cmd, m_renderer.ndgi(), m_renderer.frameIndex());
        }
    });

    // === GI 结束 timestamp ===
    m_renderer.pipeline().addStep({
        .name = "TS-GI",
        .phase = "GI",
        .record = [this](VkCommandBuffer cmd) {
            writeTimestamp(cmd, m_renderer.kTsVoxelGI);
        }
    });

    // ============================
    // Phase 2: Lighting (compute)
    // ============================
    m_renderer.pipeline().addStep({
        .name = "Lighting",
        .phase = "Shading",
        .timestampSlot = m_renderer.kTsLighting,
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_renderer.lighting().record(cmd, m_renderer.rt());
            writeTimestamp(cmd, m_renderer.kTsLighting);
        }
    });

    // ============================
    // Phase 3: Skybox (graphics)
    // ============================
    m_renderer.pipeline().addStep({
        .name = "Skybox",
        .phase = "Shading",
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                    VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
            transitionImage(cmd, m_renderer.rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
            m_renderer.skybox().record(cmd, m_renderer.rt());
        }
    });

    // ============================
    // Phase 3.5: Copy hdrColor → hdrPrev
    // ============================
    m_renderer.pipeline().addStep({
        .name = "Copy-hdrPrev",
        .phase = "Shading",
        .record = [this](VkCommandBuffer cmd) {
            transitionImage(cmd, m_renderer.rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            transitionImage(cmd, m_renderer.rt().hdrPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkImageCopy hdrCopy{};
            hdrCopy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            hdrCopy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            hdrCopy.extent = {m_renderer.rt().extent.width, m_renderer.rt().extent.height, 1};
            vkCmdCopyImage(cmd,
                m_renderer.rt().hdrColor.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_renderer.rt().hdrPrev.image(),  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &hdrCopy);
            transitionImage(cmd, m_renderer.rt().hdrPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

            writeTimestamp(cmd, m_renderer.kTsSkybox);
        }
    });
}

// ============================================================
// Frame loop helpers extracted from run()
// ============================================================

void App::buildFrameUBO(FrameUBO& ubo) {
    ubo.view = m_camera.view();
    ubo.proj = m_camera.proj((float)m_renderer.rt().extent.width / (float)m_renderer.rt().extent.height);
    // Apply jitter to projection
    ubo.proj[2][0] += m_jitter.x;
    ubo.proj[2][1] += m_jitter.y;
    ubo.viewProj = ubo.proj * ubo.view;
    ubo.invViewProj = glm::inverse(ubo.viewProj);
    ubo.prevViewProj = m_prevViewProj;
    ubo.cameraPos = glm::vec4(m_camera.position, 0);
    ubo.sunDir = glm::vec4(glm::normalize(m_sunDir), 0);
    ubo.sunColor_intensity = glm::vec4(1.0f, 0.95f, 0.85f, m_sunIntensity);
    ubo.ambient = glm::vec4(m_ambient, 0);

    int specMips = 0;
    specMips = (int)m_renderer.envIbl().specularMipCount;
    int indirectEnabled = (m_giIndexApplied >= 1) ? 1 : 0;
    int rsmEnabled = m_renderer.rsmSample().enabled ? 1 : 0;
    ubo.counts = glm::ivec4((int)m_scene.materials.size(), specMips, indirectEnabled, rsmEnabled);

    // LPV grid params
    ubo.lpvCounts = glm::ivec4((int)m_renderer.kLpvResolution, m_renderer.lpvEnabled() ? 1 : 0, 0, 0);
    ubo.lpvGridMinCell = glm::vec4(m_renderer.lpvGridMin(), m_renderer.lpvCellSize());

    // VXGI grid params
    ubo.vxgiCounts = glm::ivec4((int)m_renderer.kVxgiResolution, m_renderer.vxgiEnabled() ? 1 : 0,
                                 (int)m_renderer.vxgi().mipLevels(), 0);
    ubo.vxgiGridMinCell = glm::vec4(m_renderer.vxgiGridMin(), m_renderer.vxgiCellSize());

    // PRT params
    ubo.prtCounts = glm::ivec4((int)m_renderer.kPrtResolution,
                                (m_renderer.prtEnabled() && m_renderer.prtBaked()) ? 1 : 0,
                                m_renderer.prtShOrder(), 0);
    ubo.prtGridMinCell = glm::vec4(m_renderer.prtGridMin(), m_renderer.prtCellSize());

    // SH 投影：缓存结果，仅在 sunDir 或 intensity 变化时重算
    {
        glm::vec3 dToSun = -glm::normalize(m_sunDir);
        float I = m_sunIntensity;
        if (dToSun != m_cachedSH.lastSunDir || I != m_cachedSH.lastSunIntensity) {
            m_cachedSH.lastSunDir = dToSun;
            m_cachedSH.lastSunIntensity = I;

            float x = dToSun.x, y = dToSun.y, z = dToSun.z;
            glm::vec3 sunC{1.0f, 0.95f, 0.85f};

            // l=0,1
            float Y0   = 0.282094792f;
            float Y1n1 = 0.488602512f * y;
            float Y10  = 0.488602512f * z;
            float Y11  = 0.488602512f * x;
            // l=2
            float Y2n2 = 1.092548431f * x * y;
            float Y2n1 = 1.092548431f * y * z;
            float Y20  = 0.315391565f * (3.0f * z * z - 1.0f);
            float Y21  = 1.092548431f * z * x;
            float Y22  = 0.546274215f * (x * x - y * y);
            // l=3
            float Y3n3 = 0.590043589f * y * (3.0f * x * x - y * y);
            float Y3n2 = 2.890611442f * x * y * z;
            float Y3n1 = 0.457045799f * y * (5.0f * z * z - 1.0f);
            float Y30  = 0.373176333f * z * (5.0f * z * z - 3.0f);
            float Y31  = 0.457045799f * x * (5.0f * z * z - 1.0f);
            float Y32  = 1.445305721f * z * (x * x - y * y);
            float Y33  = 0.590043589f * x * (x * x - 3.0f * y * y);

            // SH4
            m_cachedSH.prtLightSH_R = glm::vec4(I*sunC.r*Y0, I*sunC.r*Y1n1, I*sunC.r*Y10, I*sunC.r*Y11);
            m_cachedSH.prtLightSH_G = glm::vec4(I*sunC.g*Y0, I*sunC.g*Y1n1, I*sunC.g*Y10, I*sunC.g*Y11);
            m_cachedSH.prtLightSH_B = glm::vec4(I*sunC.b*Y0, I*sunC.b*Y1n1, I*sunC.b*Y10, I*sunC.b*Y11);
            // SH9
            m_cachedSH.prtLightSH9_R0 = glm::vec4(I*sunC.r*Y2n2, I*sunC.r*Y2n1, I*sunC.r*Y20, I*sunC.r*Y21);
            m_cachedSH.prtLightSH9_R1 = glm::vec4(I*sunC.r*Y22,  0, 0, 0);
            m_cachedSH.prtLightSH9_G0 = glm::vec4(I*sunC.g*Y2n2, I*sunC.g*Y2n1, I*sunC.g*Y20, I*sunC.g*Y21);
            m_cachedSH.prtLightSH9_G1 = glm::vec4(I*sunC.g*Y22,  0, 0, 0);
            m_cachedSH.prtLightSH9_B0 = glm::vec4(I*sunC.b*Y2n2, I*sunC.b*Y2n1, I*sunC.b*Y20, I*sunC.b*Y21);
            m_cachedSH.prtLightSH9_B1 = glm::vec4(I*sunC.b*Y22,  0, 0, 0);
            // SH16
            m_cachedSH.prtLightSH16_R0 = glm::vec4(I*sunC.r*Y3n3, I*sunC.r*Y3n2, I*sunC.r*Y3n1, I*sunC.r*Y30);
            m_cachedSH.prtLightSH16_R1 = glm::vec4(I*sunC.r*Y31,  I*sunC.r*Y32,  I*sunC.r*Y33,  0);
            m_cachedSH.prtLightSH16_G0 = glm::vec4(I*sunC.g*Y3n3, I*sunC.g*Y3n2, I*sunC.g*Y3n1, I*sunC.g*Y30);
            m_cachedSH.prtLightSH16_G1 = glm::vec4(I*sunC.g*Y31,  I*sunC.g*Y32,  I*sunC.g*Y33,  0);
            m_cachedSH.prtLightSH16_B0 = glm::vec4(I*sunC.b*Y3n3, I*sunC.b*Y3n2, I*sunC.b*Y3n1, I*sunC.b*Y30);
            m_cachedSH.prtLightSH16_B1 = glm::vec4(I*sunC.b*Y31,  I*sunC.b*Y32,  I*sunC.b*Y33,  0);
        }
        // 从缓存写入 UBO（无论是否重算）
        ubo.prtLightSH_R   = m_cachedSH.prtLightSH_R;
        ubo.prtLightSH_G   = m_cachedSH.prtLightSH_G;
        ubo.prtLightSH_B   = m_cachedSH.prtLightSH_B;
        ubo.prtLightSH9_R0 = m_cachedSH.prtLightSH9_R0;
        ubo.prtLightSH9_R1 = m_cachedSH.prtLightSH9_R1;
        ubo.prtLightSH9_G0 = m_cachedSH.prtLightSH9_G0;
        ubo.prtLightSH9_G1 = m_cachedSH.prtLightSH9_G1;
        ubo.prtLightSH9_B0 = m_cachedSH.prtLightSH9_B0;
        ubo.prtLightSH9_B1 = m_cachedSH.prtLightSH9_B1;
        ubo.prtLightSH16_R0 = m_cachedSH.prtLightSH16_R0;
        ubo.prtLightSH16_R1 = m_cachedSH.prtLightSH16_R1;
        ubo.prtLightSH16_G0 = m_cachedSH.prtLightSH16_G0;
        ubo.prtLightSH16_G1 = m_cachedSH.prtLightSH16_G1;
        ubo.prtLightSH16_B0 = m_cachedSH.prtLightSH16_B0;
        ubo.prtLightSH16_B1 = m_cachedSH.prtLightSH16_B1;
    }

    // DDGI probe params
    ubo.ddgiCounts = glm::ivec4((int)DdgiResources::kProbesX,
                                 (int)DdgiResources::kProbesY,
                                 (int)DdgiResources::kProbesZ,
                                 m_renderer.ndgiEnabled() ? 2 : (m_renderer.ddgiEnabled() ? 1 : 0));
    ubo.ddgiOrigin = glm::vec4(m_renderer.ddgiOrigin(), 0);
    ubo.ddgiSpacing = glm::vec4(m_renderer.ddgiSpacing(), 0);
    ubo.ddgiOctaSizes = glm::ivec4((int)DdgiResources::kOctaIrr,
                                    (int)DdgiResources::kOctaDist, 0, 0);
    ubo.lumenCounts = glm::ivec4(m_renderer.lumenEnabled() ? 1 : 0, 0, 0, 0);

    // Push to GPU buffers
    m_renderer.gbuffer().updateFrame(ubo);
    m_renderer.forward().updateFrame(ubo);
    m_renderer.skybox().updateFrame(ubo.invViewProj, m_camera.position);
    m_renderer.rsmGeom().updateLight(m_scene.aabbMin, m_scene.aabbMax,
                          glm::normalize(m_sunDir),
                          glm::vec3(1.0f, 0.95f, 0.85f),
                          m_sunIntensity);
}

void App::recordIndirectDraws(VkCommandBuffer cmd, uint32_t frameInFlight, const glm::mat4& viewProj) {
    if (m_drawCount == 0) return;

    if (m_useGpuCulling) {
        // Build Hi-Z from previous frame's depth (only if occlusion enabled)
        if (m_useHiZOcclusion) m_renderer.hizPass().record(cmd, m_renderer.rt());

        // GPU frustum culling (+ Hi-Z occlusion if enabled)
        if (m_useHiZOcclusion) {
            m_renderer.cullPass().record(cmd, m_sceneGpu.drawDataBuffer.handle(),
                m_drawCount, m_indirectBuf.handle(), m_countBuf.handle(), viewProj,
                m_renderer.rt().extent, frameInFlight,
                m_renderer.hizPass().mip1View(), m_renderer.hizPass().mip2View(),
                m_renderer.hizPass().mip3View(), m_renderer.hizPass().mip4View());
        } else {
            m_renderer.cullPass().record(cmd, m_sceneGpu.drawDataBuffer.handle(),
                m_drawCount, m_indirectBuf.handle(), m_countBuf.handle(), viewProj,
                m_renderer.rt().extent, frameInFlight);
        }
        m_culledDrawCount = m_drawCount;  // conservative; GPU cull reduces this

        // Barrier: compute write → indirect draw
        VkBufferMemoryBarrier2 b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        b.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        b.buffer = m_indirectBuf.handle(); b.size = VK_WHOLE_SIZE;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.bufferMemoryBarrierCount = 1; di.pBufferMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);

        // Sun-view uses full unfiltered list
        auto* sunCmds = (VkDrawIndexedIndirectCommand*)m_indirectBufSun.mapped();
        for (uint32_t i = 0; i < m_drawCount; ++i) {
            const auto& e = m_drawEntries[i];
            sunCmds[i] = {e.indexCount, 1, e.firstIndex, e.vertexOffset, i};
        }
    } else {
        // CPU fill (no culling)
        m_culledDrawCount = m_drawCount;
        auto* icmds = (VkDrawIndexedIndirectCommand*)m_indirectBuf.mapped();
        for (uint32_t i = 0; i < m_drawCount; ++i) {
            const auto& e = m_drawEntries[i];
            icmds[i] = {e.indexCount, 1, e.firstIndex, e.vertexOffset, i};
        }
        std::memcpy(m_indirectBufSun.mapped(), icmds, m_drawCount * sizeof(VkDrawIndexedIndirectCommand));
    }
}

void App::recordPostProcessing(VkCommandBuffer cmd) {
    bool hdrActive = m_swap->hdrEnabled();
    bool aaActive = (m_aaMethod == AAMethod::TAA || m_aaMethod == AAMethod::SMAA);
    // FrameGraph 路径：HDR+AA 时 FG 统一处理 Tonemap + TAA/SMAA + Copy-aaHistory
    bool fgHandlesPostAA = m_useFrameGraph && hdrActive && aaActive;

    // hdrColor → SHADER_READ_ONLY：FrameGraph AA 模式下由 Tonemap pass auto-barrier 处理
    if (!fgHandlesPostAA) {
        transitionImage(cmd, m_renderer.rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    }

    if (hdrActive) {
        // === HDR path ===
        if (aaActive) {
            // FrameGraph 路径：Tonemap+TAA/SMAA+Copy-aaHistory 已由 FG 处理，
            // swapImage 已被 TAA/SMAA pass 写入为 GENERAL
            if (!fgHandlesPostAA) {
                m_renderer.rt().ensureAaResources(*m_device);
                transitionImage(cmd, m_renderer.rt().aaHdr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
                m_renderer.tonemap().bindOutput(*m_device, m_renderer.rt().aaHdr.view(), m_frameCtx.frameInFlight);
                m_renderer.tonemap().record(cmd, m_renderer.rt(), m_frameCtx.frameInFlight, true, 1.0f);
                writeTimestamp(cmd, m_renderer.kTsTonemap);

                // aaHdr: Tonemap 写入后布局为 GENERAL → SR_O 供 TAA/SMAA 读取
                transitionImage(cmd, m_renderer.rt().aaHdr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

                transitionImage(cmd, m_frameCtx.swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);

                if (m_aaMethod == AAMethod::TAA) {
                    m_renderer.taa().bindResources(*m_device, m_renderer.rt(), m_frameCtx.frameInFlight);
                    m_renderer.taa().bindOutput(*m_device, m_frameCtx.swapView, m_frameCtx.frameInFlight);
                    m_renderer.taa().record(cmd, m_renderer.rt(), m_jitter, m_prevJitter,
                                m_frameCtx.invViewProj, m_prevViewProj, m_frameCtx.frameInFlight, m_taaBlendAlpha);
                    // Copy aaHdr → aaHistory for next frame
                    transitionImage(cmd, m_renderer.rt().aaHdr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                    transitionImage(cmd, m_renderer.rt().aaHistory.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                    VkImageCopy histCopy{};
                    histCopy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    histCopy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    histCopy.extent = {m_renderer.rt().extent.width, m_renderer.rt().extent.height, 1};
                    vkCmdCopyImage(cmd, m_renderer.rt().aaHdr.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   m_renderer.rt().aaHistory.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &histCopy);
                    transitionImage(cmd, m_renderer.rt().aaHdr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                    transitionImage(cmd, m_renderer.rt().aaHistory.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                } else {
                    m_renderer.smaa().bindResources(*m_device, m_renderer.rt());
                    m_renderer.smaa().bindOutput(*m_device, m_frameCtx.swapView);
                    m_renderer.smaa().record(cmd, m_renderer.rt());
                }
                writeTimestamp(cmd, m_renderer.kTsAA);
            }
        } else {
            // No AA: tonemap writes directly to swapchain
            transitionImage(cmd, m_frameCtx.swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
            m_renderer.tonemap().bindOutput(*m_device, m_frameCtx.swapView, m_frameCtx.frameInFlight);
            m_renderer.tonemap().record(cmd, m_renderer.rt(), m_frameCtx.frameInFlight, true, 1.0f);
            writeTimestamp(cmd, m_renderer.kTsTonemap);
            writeTimestamp(cmd, m_renderer.kTsAA);
        }

        // Transition swapchain to COLOR_ATTACHMENT for ImGui
        transitionImage(cmd, m_frameCtx.swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    } else {
        // === SDR path ===
        if (aaActive) {
            m_renderer.rt().ensureAaResources(*m_device);

            // Tonemap writes to aaHdr
            transitionImage(cmd, m_renderer.rt().aaHdr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
            m_renderer.tonemap().bindOutput(*m_device, m_renderer.rt().aaHdr.view(), m_frameCtx.frameInFlight);
            m_renderer.tonemap().record(cmd, m_renderer.rt(), m_frameCtx.frameInFlight);
            writeTimestamp(cmd, m_renderer.kTsTonemap);

            transitionImage(cmd, m_renderer.rt().aaHdr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

            transitionImage(cmd, m_renderer.rt().ldrTonemap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);

            if (m_aaMethod == AAMethod::TAA) {
                if (m_renderer.aaHistoryNeedsInit()) {
                    transitionImage(cmd, m_renderer.rt().aaHistory.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                    m_renderer.aaHistoryNeedsInit() = false;
                }
                transitionImage(cmd, m_renderer.rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                m_renderer.taa().bindResources(*m_device, m_renderer.rt(), m_frameCtx.frameInFlight);
                m_renderer.taa().record(cmd, m_renderer.rt(), m_jitter, m_prevJitter,
                            m_frameCtx.invViewProj, m_prevViewProj, m_frameCtx.frameInFlight, m_taaBlendAlpha);

                // Copy aaHdr → aaHistory for next frame
                transitionImage(cmd, m_renderer.rt().aaHdr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                transitionImage(cmd, m_renderer.rt().aaHistory.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                VkImageCopy histCopy{};
                histCopy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                histCopy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                histCopy.extent = {m_renderer.rt().extent.width, m_renderer.rt().extent.height, 1};
                vkCmdCopyImage(cmd,
                    m_renderer.rt().aaHdr.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    m_renderer.rt().aaHistory.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &histCopy);
                transitionImage(cmd, m_renderer.rt().aaHdr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                transitionImage(cmd, m_renderer.rt().aaHistory.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            } else {
                m_renderer.smaa().bindResources(*m_device, m_renderer.rt());
                m_renderer.smaa().record(cmd, m_renderer.rt());
            }
            writeTimestamp(cmd, m_renderer.kTsAA);

            // Barrier: ldrTonemap GENERAL → TRANSFER_SRC for blit
            transitionImage(cmd, m_renderer.rt().ldrTonemap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        } else {
            // No AA: tonemap writes directly to ldrTonemap
            transitionImage(cmd, m_renderer.rt().ldrTonemap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
            m_renderer.tonemap().bindOutput(*m_device, m_renderer.rt().ldrTonemap.view(), m_frameCtx.frameInFlight);
            m_renderer.tonemap().record(cmd, m_renderer.rt(), m_frameCtx.frameInFlight);
            writeTimestamp(cmd, m_renderer.kTsTonemap);
            writeTimestamp(cmd, m_renderer.kTsAA);

            // Barrier: ldrTonemap GENERAL → TRANSFER_SRC for blit
            transitionImage(cmd, m_renderer.rt().ldrTonemap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        }

        // SDR blit: ldrTonemap → swapchain
        transitionImage(cmd, m_frameCtx.swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {(int32_t)m_renderer.rt().extent.width, (int32_t)m_renderer.rt().extent.height, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[1] = {(int32_t)m_frameCtx.swapExtent.width, (int32_t)m_frameCtx.swapExtent.height, 1};
        vkCmdBlitImage(cmd,
            m_renderer.rt().ldrTonemap.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            m_frameCtx.swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        // SDR: transition swapchain to COLOR_ATTACHMENT for ImGui
        transitionImage(cmd, m_frameCtx.swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    }

    // 恢复 hdrColor 到 TRANSFER_SRC，匹配 Copy-hdrPrev 的 descriptor layout
    transitionImage(cmd, m_renderer.rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);

    // Final timestamp + transition to present
    writeTimestamp(cmd, m_renderer.kTsEnd);
    transitionImage(cmd, m_frameCtx.swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
}

void App::renderDebugWindow() {
    if (m_imguiWin->shouldClose()) {
        m_device->waitIdle();
        m_renderer.imgui().destroy();
        m_imguiSwap.reset();
        m_imguiWin.reset();
        WindowDesc iwd; iwd.title = "SomeGI Debug"; iwd.width = 600; iwd.height = 900;
        m_imguiWin = std::make_unique<Window>(iwd);
        m_imguiSwap = std::make_unique<Swapchain>(*m_device, *m_imguiWin);
        m_renderer.imgui().init(*m_device, m_imguiWin->handle(), m_imguiSwap->format(), kFramesInFlight);
        return;
    }
    auto f = m_imguiSwap->acquireNextFrame();
    if (f.needsResize) { m_imguiSwap->recreate(); return; }

    VkCommandBuffer c = m_imguiCmds[f.frameInFlight];
    vkResetCommandBuffer(c, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(c, &bi));
    transitionImage(c, f.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    m_renderer.imgui().render(c, f.view, f.extent);
    transitionImage(c, f.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    VK_CHECK(vkEndCommandBuffer(c));
    VkCommandBufferSubmitInfo cs{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, c};
    VkSemaphoreSubmitInfo ws{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, f.sync->imageAvailable, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphoreSubmitInfo ss{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, f.renderFinished, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT};
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.waitSemaphoreInfoCount = 1; si.pWaitSemaphoreInfos = &ws;
    si.commandBufferInfoCount = 1; si.pCommandBufferInfos = &cs;
    si.signalSemaphoreInfoCount = 1; si.pSignalSemaphoreInfos = &ss;
    VK_CHECK(vkQueueSubmit2(m_device->graphicsQueue(), 1, &si, f.sync->inFlight));
    m_imguiSwap->present(f);
}
void App::run() {
    auto last = std::chrono::high_resolution_clock::now();
    float fpsTimer = 0;
    int fpsFrames = 0;
    while (!m_window->shouldClose()) {
        // ---- Timing & input ----
        m_window->pollEvents();
        m_imguiWin->pollEvents();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        m_dtMs = dt * 1000.0f;
        fpsTimer += dt; fpsFrames++;
        if (fpsTimer >= 0.5f) {
            m_fpsAvg = fpsFrames / fpsTimer;
            fpsTimer = 0; fpsFrames = 0;
            std::printf("[profile] fps=%.0f cpu=%.2fms gpu=%.2fms gi=%d\n",
                        m_fpsAvg, m_dtMs, m_renderer.gpuMs(), m_giIndexApplied);
        }

        m_renderer.imgui().newFrame();
        bool wantMouse = ImGui::GetIO().WantCaptureMouse;
        bool wantKbd   = ImGui::GetIO().WantCaptureKeyboard;
        if (!wantMouse && !wantKbd) {
            m_flyer.update(m_camera, dt, m_window->handle());
        }
        buildUI();
        applySceneSelection();

        if (!wantKbd && ImGui::IsKeyPressed(ImGuiKey_F2)) {
            startBenchmark();
        }
        if (m_benchmark.running()) {
            tickBenchmark(dt);
        } else {
            applyGiSelection();
            applyShadowSelection();
        }
        if (!m_renderer.prtBaked()) {
            bakePrt();
            m_renderer.prtBaked() = true;
        }

        // ---- Swapchain acquire ----
        auto frame = m_swap->acquireNextFrame();
        if (frame.needsResize) { m_swap->recreate(); onSwapchainResized(); continue; }
        if (frame.extent.width != m_renderer.rt().extent.width || frame.extent.height != m_renderer.rt().extent.height) {
            onSwapchainResized();
        }

        // ---- TAA jitter ----
        m_prevJitter = m_jitter;
        if (m_aaMethod == AAMethod::TAA) {
            auto halton = [](int idx, int base) -> float {
                float f = 1.0f, r = 0.0f;
                int i = idx + 1;
                while (i > 0) { f /= (float)base; r += f * (float)(i % base); i /= base; }
                return r;
            };
            float jx = (halton((int)m_renderer.frameIndex(), 2) - 0.5f) * 2.0f;
            float jy = (halton((int)m_renderer.frameIndex(), 3) - 0.5f) * 2.0f;
            m_jitter = glm::vec2(jx / m_renderer.rt().extent.width, jy / m_renderer.rt().extent.height);
        } else {
            m_jitter = glm::vec2(0.0f);
        }

        // ---- Build FrameUBO ----
        FrameUBO ubo{};
        buildFrameUBO(ubo);

        // ---- Begin command buffer ----
        VkCommandBuffer cmd = m_cmds[frame.frameInFlight];
        vkResetCommandBuffer(cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        // ---- Timestamp readback from previous frame ----
        // 使用 WITH_AVAILABILITY 而非 WAIT：FrameGraph 路径不会写入旧管线的
        // AO/VoxelGI/Skybox 等 slot，WAIT 会导致在这些未写入 slot 上永久阻塞。
        uint32_t qBase = frame.frameInFlight * m_renderer.kTimestampSlots;
        if (m_renderer.timestampValid(frame.frameInFlight)) {
            uint64_t tsArr[m_renderer.kTimestampSlots * 2] = {};
            VkResult r = vkGetQueryPoolResults(m_device->device(), m_renderer.timestampPool(),
                qBase, m_renderer.kTimestampSlots, sizeof(tsArr), tsArr, 2 * sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            if (r == VK_SUCCESS || r == VK_NOT_READY) {
                float period = m_device->timestampPeriod() * 1e-6f;
                float total = 0;
                float* dst = m_renderer.passTimes(frame.frameInFlight);
                for (uint32_t i = 1; i < m_renderer.kTimestampSlots; ++i) {
                    uint64_t curVal   = tsArr[i * 2];
                    uint64_t curAvail = tsArr[i * 2 + 1];
                    uint64_t prevVal   = tsArr[(i - 1) * 2];
                    uint64_t prevAvail = tsArr[(i - 1) * 2 + 1];
                    if (curAvail && prevAvail && curVal > prevVal) {
                        float ms = float(curVal - prevVal) * period;
                        dst[i] = dst[i] * 0.9f + ms * 0.1f;
                        total += ms;
                    }
                }
                if (total > 0) m_renderer.gpuMs() = m_renderer.gpuMs() * 0.9f + total * 0.1f;
            }
        }
        if (m_useGpuCulling && m_countBuf.handle() != VK_NULL_HANDLE && m_drawCount > 0) {
            uint32_t culled = *(uint32_t*)m_countBuf.mapped();
            if (culled > 0 && culled <= m_drawCount) m_culledDrawCount = culled;
        }

        // ---- Set frame context for pipeline lambda captures ----
        m_frameCtx = {frame.frameInFlight, frame.view, frame.image,
                      frame.extent, ubo.proj, ubo.view, ubo.invViewProj};

        // ---- Reset timestamp pool + write start ----
        vkCmdResetQueryPool(cmd, m_renderer.timestampPool(), qBase, m_renderer.kTimestampSlots);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             m_renderer.timestampPool(), qBase + m_renderer.kTsStart);

        // ---- GPU-driven indirect draws ----
        recordIndirectDraws(cmd, frame.frameInFlight, ubo.viewProj);

        // ---- Shadow: 更新 sun 方向 + 绑定每帧资源 ----
        m_renderer.shadow().setSunDir(m_sunDir);
        m_renderer.shadow().bindFrameResources(*m_device,
            m_renderer.gbuffer().frameUboHandle(), m_renderer.rt().depth.view(),
            m_renderer.rt().gNormalRough.view());

        // ---- Execute render pipeline ----
        if (m_useFrameGraph) {
            m_fg.reset();
            m_fg.setAutoBarriers(true);
            setupFrameGraph();
            m_fg.compile();
            buildPipelineTable();
            m_fg.execute(cmd);
            m_fg.applyTimestampsToDebug();
        } else {
            // 现有 RenderPipeline 路径
            buildPipelineTable();
            m_renderer.pipeline().execute(cmd);
        }
        ++m_renderer.frameIndex();

        // ---- Post-processing (tonemap + AA + blit) ----
        recordPostProcessing(cmd);

        // ---- Finalize + submit main window ----
        m_renderer.timestampValid(m_frameCtx.frameInFlight) = true;
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        csi.commandBuffer = cmd;
        VkSemaphoreSubmitInfo waitS{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        waitS.semaphore = frame.sync->imageAvailable;
        waitS.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSemaphoreSubmitInfo signalS{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signalS.semaphore = frame.renderFinished;
        signalS.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.waitSemaphoreInfoCount = 1;     si.pWaitSemaphoreInfos = &waitS;
        si.signalSemaphoreInfoCount = 1;   si.pSignalSemaphoreInfos = &signalS;
        si.commandBufferInfoCount = 1;     si.pCommandBufferInfos = &csi;
        VK_CHECK(vkQueueSubmit2(m_device->graphicsQueue(), 1, &si, frame.sync->inFlight));
        m_swap->present(frame);

        // ---- ImGui debug window ----
        renderDebugWindow();

        // ---- Save viewProj for next frame's reprojection ----
        m_prevViewProj = ubo.viewProj;
    }
}

void App::setupFrameGraph() {
    using namespace somegi::fg;
    auto ext2d = m_swap->extent();
    VkExtent3D ext{ext2d.width, ext2d.height, 1};
    auto& rt = m_renderer.rt();

    // ================================================================
    // 导入持久纹理资源（跨帧存活，FrameGraph 不管理其生命周期）
    // ================================================================

    // GBuffer (resolved, single-sample) —— 初始布局为 UNDEFINED（首帧由 GBuffer pass 写入）
    m_fgh.gAlbedoMetal = m_fg.importTexture("gAlbedoMetal", rt.gAlbedoMetal.image(),
        {ext, rt.gAlbedoMetal.format(), 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.gNormalRough = m_fg.importTexture("gNormalRough", rt.gNormalRough.image(),
        {ext, rt.gNormalRough.format(), 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.gEmissiveAO = m_fg.importTexture("gEmissiveAO", rt.gEmissiveAO.image(),
        {ext, rt.gEmissiveAO.format(), 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.depth = m_fg.importTexture("depth", rt.depth.image(),
        {ext, VK_FORMAT_D32_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // GBuffer MSAA
    m_fgh.gAlbedoMetalMs = m_fg.importTexture("gAlbedoMetalMs", rt.gAlbedoMetalMs.image(),
        {ext, rt.gAlbedoMetalMs.format(), 1, 1, m_msaaSamples,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT}, VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.gNormalRoughMs = m_fg.importTexture("gNormalRoughMs", rt.gNormalRoughMs.image(),
        {ext, rt.gNormalRoughMs.format(), 1, 1, m_msaaSamples,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT}, VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.gEmissiveAOMs = m_fg.importTexture("gEmissiveAOMs", rt.gEmissiveAOMs.image(),
        {ext, rt.gEmissiveAOMs.format(), 1, 1, m_msaaSamples,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT}, VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.depthMs = m_fg.importTexture("depthMs", rt.depthMs.image(),
        {ext, VK_FORMAT_D32_SFLOAT, 1, 1, m_msaaSamples,
         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT}, VK_IMAGE_LAYOUT_UNDEFINED);

    // AO
    m_fgh.ssao = m_fg.importTexture("ssao", rt.ssao.image(),
        {ext, VK_FORMAT_R8_UNORM, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // SSR
    m_fgh.ssr = m_fg.importTexture("ssr", rt.ssr.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // SSGI
    m_fgh.ssgi = m_fg.importTexture("ssgi", rt.ssgi.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.ssgiPrev = m_fg.importTexture("ssgiPrev", rt.ssgiPrev.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // HDR
    m_fgh.hdrColor = m_fg.importTexture("hdrColor", rt.hdrColor.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);
    m_fgh.hdrPrev = m_fg.importTexture("hdrPrev", rt.hdrPrev.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // LDR
    m_fgh.ldrTonemap = m_fg.importTexture("ldrTonemap", rt.ldrTonemap.image(),
        {ext, VK_FORMAT_B8G8R8A8_UNORM, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // RT GI
    m_fgh.rtGI = m_fg.importTexture("rtGI", rt.rtGI.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // ReSTIR
    m_fgh.restir = m_fg.importTexture("restir", rt.restir.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // RSM GI
    m_fgh.rsmGI = m_fg.importTexture("rsmGI", rt.rsmGI.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // Lumen GI
    m_fgh.lumenGI = m_fg.importTexture("lumenGI", rt.lumenGI.image(),
        {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // Shadow mask
    m_fgh.shadowMask = m_fg.importTexture("shadowMask",
        m_renderer.shadow().shadowMask().image(),
        {ext, VK_FORMAT_R8_UNORM, 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // AA
    bool aaEnabled = (m_aaMethod == AAMethod::TAA || m_aaMethod == AAMethod::SMAA);
    if (aaEnabled) {
        m_fgh.aaHdr = m_fg.importTexture("aaHdr", rt.aaHdr.image(),
            {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
            VK_IMAGE_LAYOUT_UNDEFINED);
        m_fgh.aaHistory = m_fg.importTexture("aaHistory", rt.aaHistory.image(),
            {ext, VK_FORMAT_R16G16B16A16_SFLOAT, 1, 1, VK_SAMPLE_COUNT_1_BIT,
             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT},
            VK_IMAGE_LAYOUT_UNDEFINED);
    }

    // Swapchain 图像（每帧导入，acquire 后布局为 UNDEFINED）
    // 注意：不使用 debugName，避免 persistentState 跨帧恢复错误的布局
    //       （vkAcquireNextImageKHR 每帧将 swapImage 重置为 UNDEFINED）
    m_fgh.swapImage = m_fg.importTexture(nullptr, m_frameCtx.swapImage,
        {ext, m_swap->format(), 1, 1, VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT},
        VK_IMAGE_LAYOUT_UNDEFINED);

    // ================================================================
    // 注册所有 Pass（按执行顺序）
    // 注意：FrameGraph 会自动插入 barrier，不需要手动 transitionImage
    // ================================================================

    bool fwd = (m_renderingMode == RenderingMode::Forward);
    bool aoEnabled = (!fwd && m_aoMethod != AOMethod::None);
    bool needVoxelGrid = !fwd && (m_renderer.vxgiEnabled() || m_renderer.ddgiEnabled()
        || m_renderer.sdfgiPass().enabled || m_renderer.lumenEnabled()
        || m_renderer.restirPass().enabled);

    // ════════════════════════════════════════════════════════════════
    // 注意：必须先声明 GBuffer/Forward（写 depth），再声明 Shadow（读 depth），
    // 否则 FrameGraph 的 buildEdges 无法建立 RAW 依赖边。
    // ════════════════════════════════════════════════════════════════

    // --- Forward (替代 GBuffer，前向渲染模式) ---
    if (fwd) {
        // FrameGraph auto-barrier: hdrColor → COLOR_ATTACHMENT（entry），depth → DEPTH_ATTACHMENT（entry）
        //                           hdrColor → COPY_SRC（exit，由 Copy-hdrPrev 触发），depth → SR_O（exit，由 Shadow 触发）
        m_fg.addPass("Forward", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Graphics);
            b.write(m_fgh.hdrColor);
            b.write(m_fgh.depth);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                m_renderer.forward().record(cmd, m_renderer.rt(), m_indirectBuf.handle(), m_drawCount, m_sceneGpu);
            });
        });
    }

    // --- GBuffer ---
    // FrameGraph auto-barrier:
    //   MSAA targets → COLOR_ATTACHMENT / DEPTH_ATTACHMENT（entry）
    //   Resolve targets → COLOR_ATTACHMENT / DEPTH_ATTACHMENT（entry）
    //   Resolved GBuffer → SR_O（exit，由 SSAO/SSR/Lighting 的 read 触发）
    if (!fwd) {
        m_fg.addPass("GBuffer", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Graphics);
            b.write(m_fgh.gAlbedoMetalMs);
            b.write(m_fgh.gNormalRoughMs);
            b.write(m_fgh.gEmissiveAOMs);
            b.write(m_fgh.depthMs);
            b.write(m_fgh.gAlbedoMetal);   // resolve targets
            b.write(m_fgh.gNormalRough);
            b.write(m_fgh.gEmissiveAO);
            b.write(m_fgh.depth);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                m_renderer.gbuffer().record(cmd, m_renderer.rt(), m_indirectBuf.handle(), m_drawCount, m_sceneGpu);
            });
        });
    }

    // --- RSM Geometry ---
    m_fg.addPass("RSM-Geometry", [&](FGBuilder& b) {
        b.setPassType(FGPassType::Graphics);
        b.setManualBarriers();  // 复杂 Pass：内部自行管理 layout
        b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
            m_renderer.rsmGeom().record(cmd, m_indirectBufSun.handle(), m_drawCount, m_sceneGpu);
        });
    });

    // --- Shadow Pass ---
    // Auto-barrier：depth read → SHADER_READ_ONLY（compute shader 采样）
    //               shadowMask write → GENERAL（storage image write）
    // ShadowPass 内部跳过 UNDEFINED→GENERAL 和 GENERAL→SR_O 过渡
    m_renderer.shadow().setFgAutoBarrier(true);
    m_fg.addPass("Shadow", [&](FGBuilder& b) {
        b.setPassType(FGPassType::Compute);
        b.read(m_fgh.depth);
        b.write(m_fgh.shadowMask);
        b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
            m_renderer.shadow().record(cmd, m_renderer.rt(),
                m_renderer.gbuffer().frameUboHandle(),
                m_sceneGpu, m_indirectBufSun.handle(), m_drawCount,
                m_renderer.frameIndex());
        });
    });

    // --- AO ---
    if (aoEnabled) {
        const char* aoName = (m_aoMethod == AOMethod::SSAO) ? "SSAO" : "GTAO";
        // FrameGraph auto-barrier: depth/normal → SR_O（entry），ssao → GENERAL（entry）
        //                           ssao → SR_O（exit，由 Lighting 的 read 触发）
        m_fg.addPass(aoName, [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.read(m_fgh.depth);
            b.read(m_fgh.gNormalRough);
            b.write(m_fgh.ssao);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                if (m_aoMethod == AOMethod::SSAO) {
                    m_renderer.ssao().record(cmd, m_renderer.rt(),
                        m_frameCtx.proj, glm::inverse(m_frameCtx.proj), m_frameCtx.view);
                } else {
                    m_renderer.gtao().record(cmd, m_renderer.rt(),
                        m_frameCtx.proj, m_frameCtx.view);
                }
            });
        });
    } else if (!fwd) {
        // AO-Clear: 清除 ssao 为 1.0（无遮挡）
        // FrameGraph auto-barrier 负责 UNDEFINED→TRANSFER_DST（entry）和 TRANSFER_DST→SR_O（exit）
        m_fg.addPass("AO-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.ssao, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue white{};
                white.float32[0] = 1.0f; white.float32[1] = 1.0f;
                white.float32[2] = 1.0f; white.float32[3] = 1.0f;
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().ssao.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &range);
            });
        });
    } else {
        // Forward mode: AO not applicable
    }

    // --- SSR ---
    if (!fwd && m_renderer.ssr().enabled) {
        // FrameGraph auto-barrier: GBuffer → SR_O（entry），ssr → GENERAL（entry）
        //                           ssr → SR_O（exit，由 Lighting 的 read 触发）
        m_fg.addPass("SSR", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.read(m_fgh.gAlbedoMetal);
            b.read(m_fgh.gNormalRough);
            b.read(m_fgh.gEmissiveAO);
            b.read(m_fgh.depth);
            b.read(m_fgh.hdrPrev);
            b.write(m_fgh.ssr);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                m_renderer.ssr().record(cmd, m_renderer.rt());
            });
        });
    } else if (!fwd && !m_renderer.ssr().enabled) {
        m_fg.addPass("SSR-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.ssr, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().ssr.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            });
        });
    }

    // --- ScreenGI (SSGI/GTGI) ---
    bool screenGiOn = !fwd && (m_renderer.ssgi().enabled || m_renderer.gtgi().enabled);
    if (screenGiOn) {
        m_fg.addPass("ScreenGI", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：内部 copy+transition+dispatch
            b.read(m_fgh.gAlbedoMetal);
            b.read(m_fgh.gNormalRough);
            b.read(m_fgh.gEmissiveAO);
            b.read(m_fgh.depth);
            b.read(m_fgh.ssgiPrev);
            b.write(m_fgh.ssgi);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                // Copy ssgi → ssgiPrev for temporal history
                transitionImage(cmd, m_renderer.rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                transitionImage(cmd, m_renderer.rt().ssgiPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

                VkImageCopy region{};
                region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.extent = {m_renderer.rt().extent.width, m_renderer.rt().extent.height, 1};
                vkCmdCopyImage(cmd,
                    m_renderer.rt().ssgi.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    m_renderer.rt().ssgiPrev.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &region);

                // ssgiPrev → SHADER_READ_ONLY for sampling
                transitionImage(cmd, m_renderer.rt().ssgiPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                // ssgi → GENERAL for writing new value
                transitionImage(cmd, m_renderer.rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                if (m_renderer.ssgi().enabled) m_renderer.ssgi().record(cmd, m_renderer.rt());
                else m_renderer.gtgi().record(cmd, m_renderer.rt());

                transitionImage(cmd, m_renderer.rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            });
        });
    } else if (!fwd) {
        m_fg.addPass("ScreenGI-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.ssgi, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().ssgi.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            });
        });
    }

    // --- VXGI Chain ---
    if (needVoxelGrid) {
        m_fg.addPass("VXGI-Chain", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：clear+voxelize+inject+mipmap+aniso 多步操作
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                // 1. Clear entire mip chain to 0
                auto barrierAllMips = [&](VkImageLayout oldL, VkImageLayout newL,
                                           VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                           VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
                    b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
                    b.oldLayout = oldL; b.newLayout = newL;
                    b.image = m_renderer.vxgi().image().image();
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                          0, m_renderer.vxgi().mipLevels(), 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                };
                barrierAllMips(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                VkClearColorValue zeroV{};
                VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT,
                                           0, m_renderer.vxgi().mipLevels(), 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.vxgi().image().image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zeroV, 1, &rg);
                barrierAllMips(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                // 2. Voxelize: scatter all primitives to mip 0
                m_renderer.vxgiVoxelize().record(cmd, m_scene, m_sceneGpu,
                    m_renderer.vxgiGridMin(), m_renderer.vxgiCellSize(), m_renderer.kVxgiResolution);

                // 3. Inject: RSM flux → voxel mip 0 RGB
                {
                    VkImageMemoryBarrier2 vbar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    vbar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    vbar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    vbar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    vbar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    vbar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    vbar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    vbar.image = m_renderer.vxgi().image().image();
                    vbar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo vdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    vdi.imageMemoryBarrierCount = 1; vdi.pImageMemoryBarriers = &vbar;
                    vkCmdPipelineBarrier2(cmd, &vdi);
                }
                m_renderer.vxgiInject().record(cmd, m_renderer.kVxgiResolution,
                    m_renderer.vxgiGridMin(), m_renderer.vxgiCellSize());

                // 4. Mipmap: iterate src mip i → dst mip i+1
                m_renderer.vxgiMipmap().record(cmd, m_renderer.vxgi());

                // 5. Final mip → SHADER_READ_ONLY
                {
                    VkImageMemoryBarrier2 fb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    fb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    fb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    fb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    fb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    fb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    fb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    fb.image = m_renderer.vxgi().image().image();
                    fb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                           m_renderer.vxgi().mipLevels() - 1, 1, 0, 1};
                    VkDependencyInfo fdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    fdi.imageMemoryBarrierCount = 1; fdi.pImageMemoryBarriers = &fb;
                    vkCmdPipelineBarrier2(cmd, &fdi);
                }

                // 6. Aniso alpha mipchain: UNDEFINED → SHADER_READ_ONLY
                {
                    VkImageMemoryBarrier2 ab{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    ab.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    ab.srcAccessMask = 0;
                    ab.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    ab.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    ab.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    ab.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    ab.image = m_renderer.vxgi().aniso().image();
                    ab.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                           0, m_renderer.vxgi().mipLevels(), 0, 1};
                    VkDependencyInfo adi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    adi.imageMemoryBarrierCount = 1; adi.pImageMemoryBarriers = &ab;
                    vkCmdPipelineBarrier2(cmd, &adi);
                }
                m_renderer.vxgiAniso().record(cmd, m_renderer.vxgi());
            });
        });

        if (m_renderer.vxgiRelightEnabled()) {
            m_fg.addPass("VXGI-Relight", [&](FGBuilder& b) {
                b.setPassType(FGPassType::Compute);
                b.setManualBarriers();  // 复杂 Pass：multi-bounce ping-pong
                b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                    int bounces = m_renderer.lumenEnabled() ? 3 : 1;

                    auto transImg = [&](VkImage img, VkImageLayout oldL, VkImageLayout newL,
                                        VkPipelineStageFlags2 srcS, VkAccessFlags2 srcA,
                                        VkPipelineStageFlags2 dstS, VkAccessFlags2 dstA) {
                        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                        b.srcStageMask = srcS; b.srcAccessMask = srcA;
                        b.dstStageMask = dstS; b.dstAccessMask = dstA;
                        b.oldLayout = oldL; b.newLayout = newL;
                        b.image = img;
                        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                        vkCmdPipelineBarrier2(cmd, &di);
                    };

                    auto blitScratchToVoxel = [&](VkImage srcImg) {
                        transImg(m_renderer.vxgi().image().image(),
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                            VK_PIPELINE_STAGE_2_COPY_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT);
                        transImg(srcImg,
                            VK_IMAGE_LAYOUT_GENERAL,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_COPY_BIT,
                            VK_ACCESS_2_TRANSFER_READ_BIT);
                        VkImageCopy region{};
                        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                        region.extent = {m_renderer.kVxgiResolution, m_renderer.kVxgiResolution, m_renderer.kVxgiResolution};
                        vkCmdCopyImage(cmd,
                            srcImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            m_renderer.vxgi().image().image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            1, &region);
                        transImg(m_renderer.vxgi().image().image(),
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COPY_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                    };

                    // Bounce 1: read voxelGrid → write scratch
                    transImg(m_renderer.vxgi().relightScratch().image(),
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                    m_renderer.vxgiRelight().record(cmd, m_renderer.vxgiRelight().voxelSet(),
                        m_renderer.kVxgiResolution, m_renderer.vxgi().mipLevels(),
                        m_renderer.vxgiCellSize(), m_renderer.vxgiGridMin(),
                        m_renderer.vxgiRelightStrength());

                    if (bounces >= 2) {
                        transImg(m_renderer.vxgi().relightScratch().image(),
                            VK_IMAGE_LAYOUT_GENERAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                        transImg(m_renderer.vxgi().relightScratch2().image(),
                            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                        // Bounce 2: read scratch → write scratch2
                        m_renderer.vxgiRelight().record(cmd, m_renderer.vxgiRelight().pingSet0(),
                            m_renderer.kVxgiResolution, m_renderer.vxgi().mipLevels(),
                            m_renderer.vxgiCellSize(), m_renderer.vxgiGridMin(),
                            m_renderer.vxgiRelightStrength());

                        if (bounces >= 3) {
                            transImg(m_renderer.vxgi().relightScratch2().image(),
                                VK_IMAGE_LAYOUT_GENERAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                            transImg(m_renderer.vxgi().relightScratch().image(),
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_LAYOUT_GENERAL,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                            // Bounce 3: read scratch2 → write scratch
                            m_renderer.vxgiRelight().record(cmd, m_renderer.vxgiRelight().pingSet1(),
                                m_renderer.kVxgiResolution, m_renderer.vxgi().mipLevels(),
                                m_renderer.vxgiCellSize(), m_renderer.vxgiGridMin(),
                                m_renderer.vxgiRelightStrength());
                            blitScratchToVoxel(m_renderer.vxgi().relightScratch().image());
                        } else {
                            blitScratchToVoxel(m_renderer.vxgi().relightScratch2().image());
                        }
                    } else {
                        blitScratchToVoxel(m_renderer.vxgi().relightScratch().image());
                    }
                });
            });
        }

        if (m_renderer.lumenEnabled() && m_renderer.vxgiSixAxisInited()) {
            m_fg.addPass("VXGI-6Axis", [&](FGBuilder& b) {
                b.setPassType(FGPassType::Compute);
                b.setManualBarriers();  // 复杂 Pass：多 image layout 过渡
                b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                    VkImageLayout axisOldL = m_renderer.lumenAtlasInited()
                        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                        : VK_IMAGE_LAYOUT_UNDEFINED;
                    VkPipelineStageFlags2 axisSrcS = m_renderer.lumenAtlasInited()
                        ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                        : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    VkAccessFlags2 axisSrcA = m_renderer.lumenAtlasInited()
                        ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0u;

                    auto transAxisToGeneral = [&](VkImage img) {
                        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                        b.srcStageMask = axisSrcS; b.srcAccessMask = axisSrcA;
                        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                        b.oldLayout = axisOldL;
                        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                        b.image = img;
                        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                        vkCmdPipelineBarrier2(cmd, &di);
                    };
                    transAxisToGeneral(m_renderer.vxgi().sixAxisX().image());
                    transAxisToGeneral(m_renderer.vxgi().sixAxisY().image());
                    transAxisToGeneral(m_renderer.vxgi().sixAxisZ().image());

                    m_renderer.vxgi6Axis().record(cmd, m_renderer.kVxgiResolution,
                        m_renderer.vxgi().mipLevels(), m_renderer.vxgiCellSize(),
                        m_renderer.vxgiGridMin(), m_renderer.vxgiRelightStrength());

                    auto transAxisToSRO = [&](VkImage img) {
                        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        b.image = img;
                        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                        vkCmdPipelineBarrier2(cmd, &di);
                    };
                    transAxisToSRO(m_renderer.vxgi().sixAxisX().image());
                    transAxisToSRO(m_renderer.vxgi().sixAxisY().image());
                    transAxisToSRO(m_renderer.vxgi().sixAxisZ().image());
                });
            });
        }
    }

    // --- SDFGI ---
    if (!fwd && m_renderer.sdfgiPass().enabled) {
        m_fg.addPass("SDFGI", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：bootstrap + compute
            b.read(m_fgh.gNormalRough);
            b.read(m_fgh.depth);
            b.read(m_fgh.ssgi);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                if (!m_renderer.sdfgiBootstrapped()) {
                    auto bootstrapToGeneral = [&](VkImage img) {
                        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                        b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                        b.srcAccessMask = 0;
                        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                        b.image = img;
                        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                        vkCmdPipelineBarrier2(cmd, &di);
                    };
                    bootstrapToGeneral(m_renderer.sdfgi().seedA().image());
                    bootstrapToGeneral(m_renderer.sdfgi().seedB().image());
                    bootstrapToGeneral(m_renderer.sdfgi().udf().image());
                    m_renderer.sdfgiBootstrapped() = true;
                }
                transitionImage(cmd, m_renderer.rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                m_renderer.sdfgiPass().record(cmd, m_renderer.sdfgi(), m_renderer.rt(), m_renderer.frameIndex(),
                    m_renderer.sdfgiPass().seedThreshold, m_renderer.sdfgiPass().maxDistCells,
                    (uint32_t)m_renderer.sdfgiPass().numRays,
                    (uint32_t)m_renderer.sdfgiPass().maxSteps,
                    m_renderer.sdfgiPass().rayMaxCells, m_renderer.sdfgiPass().hitEpsCells);

                transitionImage(cmd, m_renderer.rt().ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            });
        });
    }

    // --- RT GI ---
    bool rtGiActive = m_renderer.rtGiBound() && m_giIndexApplied == 10;
    if (!fwd && rtGiActive) {
        // FrameGraph auto-barrier: rtGI → GENERAL（entry），rtGI → SR_O（exit 由 Lighting read 触发）
        m_fg.addPass("RTGI", [&](FGBuilder& b) {
            b.setPassType(FGPassType::RayTracing);
            b.write(m_fgh.rtGI);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                m_renderer.rtGi().record(cmd, m_renderer.rt());
            });
        });
    } else if (!fwd && m_renderer.rtGiInited() && !rtGiActive) {
        m_fg.addPass("RTGI-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.rtGI, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().rtGI.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            });
        });
    }

    // --- ReSTIR ---
    if (!fwd && m_renderer.restirPass().enabled) {
        m_fg.addPass("ReSTIR", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：bootstrap + 条件 oldLayout 过渡
            b.write(m_fgh.restir);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                m_renderer.restir().updateLights(m_renderer.demoLights());
                if (!m_renderer.restirBootstrapped()) {
                    auto bootstrapToGeneral = [&](VkImage img) {
                        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                        b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                        b.srcAccessMask = 0;
                        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                        b.image = img;
                        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                        vkCmdPipelineBarrier2(cmd, &di);
                    };
                    bootstrapToGeneral(m_renderer.restir().reservoirA().image());
                    bootstrapToGeneral(m_renderer.restir().reservoirB().image());
                    m_renderer.restirBootstrapped() = true;
                }
                VkImageLayout restirOld = m_renderer.restirOutInited()
                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    : VK_IMAGE_LAYOUT_UNDEFINED;
                transitionImage(cmd, m_renderer.rt().restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    restirOld, VK_IMAGE_LAYOUT_GENERAL,
                    m_renderer.restirOutInited() ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                       : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                    m_renderer.restirOutInited() ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                m_renderer.restirOutInited() = true;

                uint32_t numLights = (uint32_t)m_renderer.demoLights().size();
                bool useRtVis = m_renderer.rtSupported() && m_renderer.rtGiBound();
                m_renderer.restirPass().record(cmd, m_renderer.restir(), m_renderer.rt(),
                    numLights,
                    (uint32_t)m_renderer.restirPass().numCandidates,
                    (uint32_t)m_renderer.restirPass().numNeighbors,
                    m_renderer.restirPass().spatialRadius,
                    (uint32_t)m_renderer.restirPass().shadowSteps,
                    m_renderer.restirPass().intensityScale,
                    m_renderer.frameIndex(),
                    useRtVis);

                transitionImage(cmd, m_renderer.rt().restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            });
        });
    } else if (!fwd && !m_renderer.restirPass().enabled) {
        m_fg.addPass("ReSTIR-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.restir, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().restir.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            });
        });
    }

    // --- DDGI ---
    if (!fwd && m_renderer.ddgiEnabled()) {
        m_fg.addPass("DDGI", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：多 atlas barrier 管理
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                auto barrierAtlas = [&](VkImage img,
                                        VkImageLayout oldL, VkImageLayout newL,
                                        VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                        VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
                    b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
                    b.oldLayout = oldL; b.newLayout = newL;
                    b.image = img;
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                };

                VkImageLayout oldAtlasL = m_renderer.ddgiAtlasInited()
                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    : VK_IMAGE_LAYOUT_UNDEFINED;
                VkAccessFlags2 srcAcc = m_renderer.ddgiAtlasInited()
                    ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;
                VkPipelineStageFlags2 srcStg = m_renderer.ddgiAtlasInited()
                    ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                    : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;

                barrierAtlas(m_renderer.ddgi().irradiance().image(),
                    oldAtlasL, VK_IMAGE_LAYOUT_GENERAL,
                    srcStg, srcAcc,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                barrierAtlas(m_renderer.ddgi().distance().image(),
                    oldAtlasL, VK_IMAGE_LAYOUT_GENERAL,
                    srcStg, srcAcc,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                float jitterRot = float((m_renderer.frameIndex() % 360) * 0.0174532925);
                m_renderer.ddgiPass().record(cmd, m_renderer.ddgi(), m_renderer.ddgiOrigin(), m_renderer.ddgiSpacing(),
                    m_renderer.vxgiGridMin(), m_renderer.vxgiCellSize(), m_renderer.kVxgiResolution,
                    jitterRot, m_renderer.frameIndex());

                barrierAtlas(m_renderer.ddgi().irradiance().image(),
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                barrierAtlas(m_renderer.ddgi().distance().image(),
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                m_renderer.ddgiAtlasInited() = true;
            });
        });
    }

    // --- Lumen ---
    bool lumenActive = !fwd && m_renderer.lumenEnabled();
    if (lumenActive) {
        m_fg.addPass("Lumen-Probe", [&](FGBuilder& b) {
            b.setPassType(FGPassType::RayTracing);
            b.setManualBarriers();  // 复杂 Pass：bootstrap + 多 image 过渡 + dispatch
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                if (!m_renderer.lumenProbeInited()) {
                    m_renderer.lumenProbe().init(*m_device);
                    m_renderer.lumenProbe().bindResources(*m_device, m_renderer.lumen(), m_renderer.rtAS(), m_sceneGpu,
                                                    m_renderer.vxgi(), m_renderer.rt(), m_renderer.gbuffer().frameUboHandle(),
                                                    m_renderer.vxgiSixAxisInited());
                    m_renderer.lumenProbeInited() = true;
                }
                // Transition probe + filtered atlas to GENERAL
                {
                    VkImageLayout oldL = m_renderer.lumenAtlasInited()
                        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                        : VK_IMAGE_LAYOUT_UNDEFINED;
                    VkPipelineStageFlags2 srcS = m_renderer.lumenAtlasInited()
                        ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                        : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    VkAccessFlags2 srcA = m_renderer.lumenAtlasInited()
                        ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0u;
                    auto transToGeneral = [&](VkImage img) {
                        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                        b.srcStageMask = srcS; b.srcAccessMask = srcA;
                        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                        b.oldLayout = oldL;
                        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                        b.image = img;
                        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                        vkCmdPipelineBarrier2(cmd, &di);
                    };
                    transToGeneral(m_renderer.lumen().probeAtlas().image());
                    transToGeneral(m_renderer.lumen().filteredAtlas().image());
                    m_renderer.lumenAtlasInited() = true;
                }
                m_renderer.lumenProbe().record(cmd, m_renderer.lumen(), m_renderer.frameIndex(),
                    m_renderer.vxgiSixAxisInited() ? 1u : 0u);

                // ProbeAtlas GENERAL → SR_O for filter
                {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    b.image = m_renderer.lumen().probeAtlas().image();
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                }
            });
        });
        m_fg.addPass("Lumen-Filter", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：init + dispatch + copy + 多 barrier
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                if (!m_renderer.lumenFilterInited()) {
                    VkImageMemoryBarrier2 pb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    pb.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    pb.srcAccessMask = 0;
                    pb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    pb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    pb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    pb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    pb.image = m_renderer.lumen().prevAtlas().image();
                    pb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo pdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    pdi.imageMemoryBarrierCount = 1; pdi.pImageMemoryBarriers = &pb;
                    vkCmdPipelineBarrier2(cmd, &pdi);

                    m_renderer.lumenFilter().init(*m_device);
                    m_renderer.lumenFilter().bindResources(*m_device, m_renderer.lumen(), m_renderer.rt(),
                                                     m_renderer.gbuffer().frameUboHandle());
                    m_renderer.lumenFilterInited() = true;
                }
                m_renderer.lumenFilter().record(cmd, m_renderer.lumen(), m_renderer.rt());

                // Copy filteredAtlas → prevAtlas for next frame
                auto imgBarrier = [&](VkImage img, VkImageLayout oldL, VkImageLayout newL,
                                      VkPipelineStageFlags2 srcS, VkAccessFlags2 srcA,
                                      VkPipelineStageFlags2 dstS, VkAccessFlags2 dstA) {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = srcS; b.srcAccessMask = srcA;
                    b.dstStageMask = dstS; b.dstAccessMask = dstA;
                    b.oldLayout = oldL; b.newLayout = newL;
                    b.image = img;
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                };

                imgBarrier(m_renderer.lumen().filteredAtlas().image(),
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                imgBarrier(m_renderer.lumen().prevAtlas().image(),
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

                VkImageCopy region{};
                region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.extent = {m_renderer.lumen().atlasWidth(), m_renderer.lumen().atlasHeight(), 1};
                vkCmdCopyImage(cmd,
                    m_renderer.lumen().filteredAtlas().image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    m_renderer.lumen().prevAtlas().image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &region);

                imgBarrier(m_renderer.lumen().filteredAtlas().image(),
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                imgBarrier(m_renderer.lumen().prevAtlas().image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            });
        });
        m_fg.addPass("Lumen-Gather", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：bootstrap + 多 layout 过渡 + dispatch
            b.write(m_fgh.lumenGI);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                if (!m_renderer.lumenGatherInited()) {
                    m_renderer.lumenGather().init(*m_device);
                    m_renderer.lumenGather().bindResources(*m_device, m_renderer.lumen(), m_renderer.rt(),
                                                     m_renderer.gbuffer().frameUboHandle(), true);
                    m_renderer.lumenGatherInited() = true;
                }
                {
                    VkImageLayout oldL = m_renderer.lumenOutInited()
                        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                        : VK_IMAGE_LAYOUT_UNDEFINED;
                    VkAccessFlags2 srcA = m_renderer.lumenOutInited()
                        ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;
                    VkPipelineStageFlags2 srcS = m_renderer.lumenOutInited()
                        ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                        : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = srcS; b.srcAccessMask = srcA;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.oldLayout = oldL;
                    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.image = m_renderer.rt().lumenGI.image();
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                }
                m_renderer.lumenGather().record(cmd, m_renderer.lumen(), m_renderer.rt(),
                    (uint32_t)m_renderer.lumenDebugMode());

                {
                    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                    b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    b.image = m_renderer.rt().lumenGI.image();
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                    vkCmdPipelineBarrier2(cmd, &di);
                }
                m_renderer.lumenOutInited() = true;
            });
        });
    } else if (!fwd) {
        m_fg.addPass("Lumen-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.lumenGI, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().lumenGI.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            });
        });
    }

    // --- LPV ---
    if (!fwd && m_renderer.lpvEnabled()) {
        m_fg.addPass("LPV-Inject", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：compute + exit transition
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                m_renderer.lpvInject().record(cmd, m_renderer.kLpvResolution,
                    m_renderer.lpvGridMin(), m_renderer.lpvCellSize());

                transitionImage(cmd, m_renderer.lpv().gv().image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            });
        });
        m_fg.addPass("LPV-Propagate", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.setManualBarriers();  // 复杂 Pass：multi-iteration ping-pong barrier
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                auto barrierLpv = [&](const LpvGrid& g,
                                      VkImageLayout oldL, VkImageLayout newL,
                                      VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                      VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                    VkImage imgs[3] = {g.lpvR.image(), g.lpvG.image(), g.lpvB.image()};
                    for (auto img : imgs) {
                        transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                            oldL, newL, srcStg, srcAcc, dstStg, dstAcc);
                    }
                };

                int propIter = m_renderer.lpvProp().iterations & ~1;
                for (int it = 0; it < propIter; ++it) {
                    LpvGrid& src = m_renderer.lpv().current();
                    LpvGrid& dst = m_renderer.lpv().next();

                    // 首次迭代 src 布局未知，使用 UNDEFINED；后续迭代 src 来自上轮 dst 写入的 GENERAL
                    VkImageLayout srcOldL = (it == 0) ? VK_IMAGE_LAYOUT_UNDEFINED
                                                      : VK_IMAGE_LAYOUT_GENERAL;
                    VkPipelineStageFlags2 srcOldS = (it == 0) ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
                                                              : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    VkAccessFlags2 srcOldA = (it == 0) ? 0
                                                       : VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    barrierLpv(src,
                        srcOldL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        srcOldS, srcOldA,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                    barrierLpv(dst,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                    m_renderer.lpvProp().record(cmd, m_renderer.lpv().curIdx(),
                        m_renderer.kLpvResolution, m_renderer.lpvProp().occlusionAmplifier,
                        m_renderer.lpvProp().gvOcclusionStrength);
                    m_renderer.lpv().swap();
                }

                barrierLpv(m_renderer.lpv().current(),
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            });
        });
    }

    // --- RSM Sample ---
    if (!fwd && m_renderer.rsmSample().enabled) {
        // FrameGraph auto-barrier: rsmGI → GENERAL（entry），rsmGI → SR_O（exit 由 Lighting read 触发）
        m_fg.addPass("RSM-Sample", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.rsmGI);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                m_renderer.rsmSample().record(cmd, m_renderer.rt());
            });
        });
    } else if (!fwd && !m_renderer.rsmSample().enabled) {
        m_fg.addPass("RSM-Clear", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.write(m_fgh.rsmGI, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkClearColorValue zero{};
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cmd, m_renderer.rt().rsmGI.image(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            });
        });
    }

    // --- NDGI ---
    if (m_renderer.ndgiEnabled() && m_renderer.rtSupported()) {
        m_fg.addPass("NDGI", [&](FGBuilder& b) {
            b.setPassType(FGPassType::RayTracing);
            b.setManualBarriers();  // RT dispatch，无 read/write 声明，内部管理 barrier
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                m_renderer.ndgiPass().record(cmd, m_renderer.ndgi(), m_renderer.frameIndex(),
                    m_renderer.ddgiOrigin(), m_renderer.ddgiSpacing());
                m_renderer.ndgiPass().recordTraining(cmd, m_renderer.ndgi(), m_renderer.frameIndex());
            });
        });
    }

    // --- Lighting ---
    if (!fwd) {
        // FrameGraph auto-barrier: 所有 GBuffer+GI → SR_O（entry），hdrColor → GENERAL（entry+exit）
        m_fg.addPass("Lighting", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.read(m_fgh.gAlbedoMetal);
            b.read(m_fgh.gNormalRough);
            b.read(m_fgh.gEmissiveAO);
            b.read(m_fgh.depth);
            b.read(m_fgh.ssao);
            b.read(m_fgh.ssr);
            b.read(m_fgh.ssgi);
            b.read(m_fgh.shadowMask);
            // Optional GI reads (if GI is active, but framegraph can handle extra deps)
            b.read(m_fgh.rsmGI);
            b.read(m_fgh.rtGI);
            b.read(m_fgh.restir);
            b.read(m_fgh.lumenGI);
            b.write(m_fgh.hdrColor);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                m_renderer.lighting().record(cmd, m_renderer.rt());
            });
        });
    }

    // --- Skybox ---
    // Manual barrier：depth 在 render pass 内作为 DEPTH_ATTACHMENT，与 Lighting
    // 等 compute pass 的 SHADER_READ_ONLY descriptor 冲突，submit-time 无法同时满足
    if (!fwd) {
        m_fg.addPass("Skybox", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Graphics);
            b.setManualBarriers();
            b.read(m_fgh.depth);
            b.write(m_fgh.hdrColor);
            b.setExitLayout(m_fgh.hdrColor, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            b.setExitLayout(m_fgh.depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                VkImageLayout depthOld = (m_renderer.frameIndex() == 0)
                    ? VK_IMAGE_LAYOUT_UNDEFINED
                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                transitionImage(cmd, m_renderer.rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                    depthOld, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
                VkImageLayout hdrOld = (m_renderer.frameIndex() == 0)
                    ? VK_IMAGE_LAYOUT_UNDEFINED
                    : VK_IMAGE_LAYOUT_GENERAL;
                transitionImage(cmd, m_renderer.rt().hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                    hdrOld, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
                m_renderer.skybox().record(cmd, m_renderer.rt());
                transitionImage(cmd, m_renderer.rt().depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            });
        });
    }

    // --- Copy hdrPrev ---
    // FrameGraph auto-barrier: hdrColor → TRANSFER_SRC, hdrPrev → TRANSFER_DST（entry）
    //                          hdrPrev → SHADER_READ_ONLY（exit, 由下一帧读触发）
    m_fg.addPass("Copy-hdrPrev", [&](FGBuilder& b) {
        b.setPassType(FGPassType::Compute);
        b.read(m_fgh.hdrColor, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        b.write(m_fgh.hdrPrev, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.extent = {m_renderer.rt().extent.width, m_renderer.rt().extent.height, 1};
            vkCmdCopyImage(cmd,
                m_renderer.rt().hdrColor.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_renderer.rt().hdrPrev.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &region);
        });
    });

    // --- Tonemap（AA 模式下纳入 FrameGraph，非 AA 模式仍由 recordPostProcessing 处理）---
    if (aaEnabled && m_swap->hdrEnabled()) {
        m_fg.addPass("Tonemap", [&](FGBuilder& b) {
            b.setPassType(FGPassType::Compute);
            b.read(m_fgh.hdrColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            b.write(m_fgh.aaHdr);
            b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                m_renderer.rt().ensureAaResources(*m_device);
                m_renderer.tonemap().bindOutput(*m_device, m_renderer.rt().aaHdr.view(), m_frameCtx.frameInFlight);
                m_renderer.tonemap().record(cmd, m_renderer.rt(), m_frameCtx.frameInFlight, true, 1.0f);
                writeTimestamp(cmd, m_renderer.kTsTonemap);
            });
        });

        // --- TAA / SMAA ---
        // 读取 aaHdr + depth(+aaHistory in TAA)，写入 swapImage
        if (m_aaMethod == AAMethod::TAA) {
            m_fg.addPass("TAA", [&](FGBuilder& b) {
                b.setPassType(FGPassType::Compute);
                b.read(m_fgh.aaHdr, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                b.read(m_fgh.aaHistory, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                b.read(m_fgh.depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                b.write(m_fgh.swapImage, VK_IMAGE_LAYOUT_GENERAL);
                b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                    m_renderer.taa().bindResources(*m_device, m_renderer.rt(), m_frameCtx.frameInFlight);
                    m_renderer.taa().bindOutput(*m_device, m_frameCtx.swapView, m_frameCtx.frameInFlight);
                    m_renderer.taa().record(cmd, m_renderer.rt(), m_jitter, m_prevJitter,
                        m_frameCtx.invViewProj, m_prevViewProj, m_frameCtx.frameInFlight, m_taaBlendAlpha);
                    writeTimestamp(cmd, m_renderer.kTsAA);
                });
            });

            // --- Copy aaHdr → aaHistory（下一帧 TAA 的历史输入）---
            m_fg.addPass("Copy-aaHistory", [&](FGBuilder& b) {
                b.setPassType(FGPassType::Compute);
                b.read(m_fgh.aaHdr, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                b.write(m_fgh.aaHistory, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                    VkImageCopy region{};
                    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    region.extent = {m_renderer.rt().extent.width, m_renderer.rt().extent.height, 1};
                    vkCmdCopyImage(cmd,
                        m_renderer.rt().aaHdr.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        m_renderer.rt().aaHistory.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &region);
                });
            });
        } else {
            // SMAA：只需 aaHdr 输入，内部 m_edgeTex 自行管理 barrier
            m_fg.addPass("SMAA", [&](FGBuilder& b) {
                b.setPassType(FGPassType::Compute);
                b.read(m_fgh.aaHdr, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                b.write(m_fgh.swapImage, VK_IMAGE_LAYOUT_GENERAL);
                b.setExecute([this](VkCommandBuffer cmd, const FGResources&) {
                    m_renderer.smaa().bindResources(*m_device, m_renderer.rt());
                    m_renderer.smaa().bindOutput(*m_device, m_frameCtx.swapView);
                    m_renderer.smaa().record(cmd, m_renderer.rt());
                    writeTimestamp(cmd, m_renderer.kTsAA);
                });
            });
        }
    }

    // 注：TAA/SMAA/Copy-aaHistory 的 barrier 由 FrameGraph 自动管理。
    //      ImGui 和最终 swapImage→PRESENT 过渡仍由 recordPostProcessing 处理。
}

}
