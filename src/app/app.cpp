#include "app.h"
#include "tests/regression_test.h"
#include "rhi/base/device.h"
#include "rhi/base/swapchain.h"
#include "rhi/base/command_buffer.h"
#include "rhi/base/fence.h"
#include "rhi/base/shader.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/buffer.h"
#include "renderer/core/render_targets.h"
#include "renderer/core/tonemap_pass.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include "scene/draw_list.h"
#include "core/window.h"
#include "core/device.h"
#include "core/swapchain.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_command.h"
#include <GLFW/glfw3.h>
#include "scene/gltf_loader.h"
#include "scene/scene_gpu.h"
#include "scene/env_loader.h"
#include "scene/upload.h"
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "app_common.h"


namespace somegi {


App::App() {
    WindowDesc wd; wd.title = "SomeGI"; wd.width = 800; wd.height = 450;
    m_window = std::make_unique<Window>(wd);
    m_device = std::make_unique<Device>(*m_window, /*validation=*/true);

    // 从文件恢复全局渲染设置（基础字段在 renderer.init() 前应用，
    // shadow/mesh-shader 等依赖 renderer 的字段在 init() 之后应用）
    AppSettings loadedCfg = loadAppSettings();
    {
        m_currentGiIndex     = loadedCfg.giIndex;
        m_currentShadowIndex = loadedCfg.shadowIndex;
        m_aoMethod           = (AOMethod)loadedCfg.aoMethod;
        m_aaMethod           = (AAMethod)loadedCfg.aaMethod;
        m_renderingMode      = (RenderingMode)loadedCfg.renderingMode;
        m_msaaSamples        = (VkSampleCountFlagBits)loadedCfg.msaaSamples;
        m_useFrameGraph      = loadedCfg.useFrameGraph;
        m_useGpuCulling      = loadedCfg.useGpuCulling;
        m_useHiZOcclusion    = loadedCfg.useHiZOcclusion;
        m_useMipmaps         = loadedCfg.useMipmaps;
        m_taaBlendAlpha      = loadedCfg.taaBlendAlpha;
        std::printf("[init] loaded app settings from %s\n", kAppSettingsPath);
    }

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

    // 截图 staging buffer 初始化（先于 ImGui 窗口避免无用的初始化）
    m_screenshot.init(*m_device, m_swap->extent());

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

    // 应用依赖 renderer 的持久化设置
    if (loadedCfg.useMeshShader && m_renderer.meshShaderSupported())
        m_renderer.setUseMeshShader(true);
    if (loadedCfg.shadowRtRays >= 4 && loadedCfg.shadowRtRays <= 32)
        m_renderer.shadow().rtRayCount() = loadedCfg.shadowRtRays;
    if (loadedCfg.shadowRtRadius >= 0.01f && loadedCfg.shadowRtRadius <= 0.10f)
        m_renderer.shadow().rtSunRadius() = loadedCfg.shadowRtRadius;

    // 初始化 FrameGraph（实验性）
    m_fg.init(*m_device);
    m_fg.initTimestamps(*m_device, 32);  // GPU timestamp profiling: 最多 32 个 active pass

    // Init ImGui on the separate debug window
    m_renderer.imgui().init(*m_device, *m_renderer.rhiDevice(), m_imguiWin->handle(), m_imguiSwap->format(), kFramesInFlight);

    // 将管线步骤注册委托给 FrameRenderer
    m_renderer.registerPipelineSteps();

    // Load all scenes' persisted view + lighting before the first scene apply.
    {
        PersistedAll p = loadAllSceneStates();
        m_sceneStates = std::move(p.scenes);
        if (!p.lastScene.empty()) {
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
    m_renderer.skybox().bindEnv(m_renderer.envIbl().envCube.view(),
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
    m_renderer.tonemap().init(*m_renderer.rhiDevice(), m_sceneGpu.linearSampler);
    m_renderer.tonemap().bindTargets(m_renderer.rt());

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
            m_renderer.rtGi().bindFrame(m_renderer.rt(), m_renderer.gbuffer().frameUboHandle(), m_renderer.rtAS(), m_sceneGpu);
        }
        // 绑定 TLAS 到 ShadowPass（RT shadow 用）
        m_renderer.shadow().bindTLAS(*m_device, m_renderer.rtAS().tlas());
        // M10：TLAS 就绪，绑定到 ReSTIR RT shade pipeline
        if (m_renderer.rtAS().instanceCount() > 0) {
            m_renderer.restirPass().bindResourcesRt(m_renderer.restir(), m_renderer.rt(),
                m_renderer.gbuffer().frameUboHandle(), m_renderer.rtAS().tlas());
        }
        m_renderer.ndgiPass().bindResources(m_renderer.ndgi(), m_renderer.rtAS(), m_sceneGpu, m_renderer.rt(),
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

        // 将场景绑定数据传递给 FrameRenderer（registerPipelineSteps 中的 lambda 已捕获引用）
        m_renderer.setBoundScene({
            &m_scene, &m_sceneGpu,
            m_sceneGpu.drawDataBuffer.handle(),
            m_indirectBuf.handle(), m_indirectBufSun.handle(), m_countBuf.handle(),
            m_drawCount, 0, m_useHiZOcclusion
        });

        m_renderer.gbuffer().bindDrawData(*m_device, m_sceneGpu.drawDataBuffer.handle());
        m_renderer.forward().bindDrawData(*m_device, m_sceneGpu.drawDataBuffer.handle());
        m_renderer.rsmGeom().bindDrawData(m_sceneGpu.drawDataBuffer.handle());
    }

    m_renderer.gbuffer().bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size());
    m_renderer.forward().bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size());
    m_renderer.rsmGeom().bindScene(m_sceneGpu, (uint32_t)m_sceneGpu.images.size());
    m_renderer.vxgiVoxelize().bindScene(m_sceneGpu, (uint32_t)m_sceneGpu.images.size(), m_renderer.vxgi());

    // 绑定场景数据到阴影 pass
    m_renderer.shadow().bindScene(*m_device, m_sceneGpu);
    m_renderer.shadow().setSceneAabb(m_scene.aabbMin, m_scene.aabbMax);
    m_renderer.shadow().setSunDir(m_sunDir);
    m_renderer.lighting().bindShadowMask(*m_device, m_renderer.shadow().shadowMask().view());

    if (m_sceneIndexApplied >= 0) {
        // Tonemap pass cached the old sampler; old one was destroyed above.
        m_renderer.tonemap().destroy();
        m_renderer.tonemap().init(*m_renderer.rhiDevice(), m_sceneGpu.linearSampler);
        m_renderer.tonemap().bindTargets(m_renderer.rt());
        if (m_aaMethod == AAMethod::TAA || m_aaMethod == AAMethod::SMAA) {
            m_renderer.rt().ensureAaResources(*m_device);
            m_renderer.taa().bindResources(m_renderer.rt(), 0);
            m_renderer.smaa().bindResources(m_renderer.rt());
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
    if (m_device) m_device->waitIdle();

    // FrameRenderer 统一清理所有 pass（含 envIbl / rt / timestamp）
    m_renderer.destroy();
    // FrameGraph 资源池清理
    m_fg.destroy();

    if (m_device) destroySceneSamplers(*m_device, m_sceneGpu);
    // 显式释放 SceneGpu 中的 GPU 资源（必须在 Device 销毁前完成）
    m_sceneGpu.vertexBuffer.reset();
    m_sceneGpu.indexBuffer.reset();
    m_sceneGpu.materialBuffer.reset();
    m_sceneGpu.drawDataBuffer.reset();
    for (auto& img : m_sceneGpu.images) img.reset();
    m_sceneGpu.images.clear();
    m_sceneGpu.whiteTex.reset();
    m_sceneGpu.normalTex.reset();
    if (m_sceneGpu.linearSampler) {
        vkDestroySampler(m_device->device(), m_sceneGpu.linearSampler, nullptr);
        m_sceneGpu.linearSampler = VK_NULL_HANDLE;
    }
    // 显式释放 indirect buffer
    m_indirectBuf.reset();
    m_indirectBufSun.reset();
    m_countBuf.reset();
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
                m_renderer.taa().bindResources(m_renderer.rt(), 0);
                m_renderer.smaa().bindResources(m_renderer.rt());
                m_renderer.aaHistoryNeedsInit() = true;
            } else {
                m_renderer.rt().destroyAaResources();
                m_renderer.tonemap().bindOutput(m_renderer.rt().ldrTonemap.view(), 0);
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
                m_renderer.taa().bindResources(m_renderer.rt(), 0);
                m_renderer.smaa().bindResources(m_renderer.rt());
                m_renderer.aaHistoryNeedsInit() = true;
            } else {
                m_renderer.rt().destroyAaResources();
                m_renderer.tonemap().bindOutput(m_renderer.rt().ldrTonemap.view(), 0);
            }
        }
        m_aoMethod = (AOMethod)ao;
    });
}

void App::onSwapchainResized() {
    // 截图 staging buffer 随 extent 变化重建
    m_screenshot.init(*m_device, m_swap->extent());

    m_renderer.rt().destroy();
    m_renderer.rt().create(*m_device, m_swap->extent(), m_msaaSamples);
    m_renderer.lighting().bindFrame(*m_device, m_renderer.rt(), m_renderer.gbuffer().frameUboHandle(),
                         m_renderer.lpv().current(), m_renderer.vxgi(), m_renderer.prt(), m_renderer.ddgi(),
                         m_renderer.ddgi().probeStates().handle());
    m_renderer.ssao().bindFrame(m_renderer.rt());
    m_renderer.gtao().bindFrame(m_renderer.rt());
    m_renderer.ssr().bindFrame(m_renderer.rt(), m_renderer.gbuffer().frameUboHandle());
    m_renderer.ssgi().bindFrame(m_renderer.rt(), m_renderer.gbuffer().frameUboHandle());
    m_renderer.gtgi().bindFrame(m_renderer.rt(), m_renderer.gbuffer().frameUboHandle());
    // SDFGI trace 也读 rt.gNormalRough/depth/ssgi → resize 后重绑。
    m_renderer.sdfgiPass().bindResources(m_renderer.sdfgi(), m_renderer.vxgi(), m_renderer.rt(), m_renderer.gbuffer().frameUboHandle());
    // ReSTIR：reservoir image 跟 swapchain，需重建；lightBuffer 跨帧持久。
    m_renderer.restir().resize(*m_device, m_swap->extent());
    m_renderer.restirPass().bindResources(m_renderer.restir(), m_renderer.vxgi(), m_renderer.rt(), m_renderer.gbuffer().frameUboHandle());
    m_renderer.restirOutInited() = false;
    m_renderer.restirBootstrapped() = false;
    m_renderer.rsmSample().bindFrame(m_renderer.rt(),
        m_renderer.gbuffer().frameUboHandle(),
        m_renderer.rsmGeom().frameUboHandle(),
        m_renderer.rsmGeom().position(), m_renderer.rsmGeom().normal(), m_renderer.rsmGeom().flux());
    // M9：swapchain resize 后 rtGI 换新 view → 重绑 RT pass。
    if (m_renderer.rtSupported() && m_renderer.rtGiBound()) {
        m_renderer.rtGi().bindFrame(m_renderer.rt(), m_renderer.gbuffer().frameUboHandle(), m_renderer.rtAS(), m_sceneGpu);
        // M10：resize 后重绑 ReSTIR RT shade 的 TLAS
        m_renderer.restirPass().bindResourcesRt(m_renderer.restir(), m_renderer.rt(),
            m_renderer.gbuffer().frameUboHandle(), m_renderer.rtAS().tlas());
    }
    // L.2：swapchain resize 后 probe atlas 重建。
    if (m_renderer.rtSupported()) {
        m_renderer.lumen().destroy();
        m_renderer.lumen().create(*m_device, m_swap->extent());
        m_renderer.lumenAtlasInited() = false;
        m_renderer.lumenOutInited()  = false;
        if (m_renderer.lumenProbeInited()) {
            m_renderer.lumenProbe().bindResources(m_renderer.lumen(), m_renderer.rtAS(), m_sceneGpu,
                                            m_renderer.vxgi(), m_renderer.rt(), m_renderer.gbuffer().frameUboHandle(),
                                            m_renderer.vxgiSixAxisInited());
        }
        if (m_renderer.lumenFilterInited()) {
            m_renderer.lumenFilter().bindResources( m_renderer.lumen(), m_renderer.rt(),
                                             m_renderer.gbuffer().frameUboHandle());
        }
        if (m_renderer.lumenGatherInited()) {
            m_renderer.lumenGather().bindResources(m_renderer.lumen(), m_renderer.rt(),
                                             m_renderer.gbuffer().frameUboHandle(), true);
        }
    }
    m_renderer.tonemap().bindTargets(m_renderer.rt());
    // AA resources were destroyed by m_renderer.rt().destroy(), recreate if needed
    if (m_aaMethod == AAMethod::TAA || m_aaMethod == AAMethod::SMAA) {
        m_renderer.rt().ensureAaResources(*m_device);
        m_renderer.taa().bindResources(m_renderer.rt(), 0);
        m_renderer.smaa().destroy();
        m_renderer.smaa().init(*m_device, *m_renderer.rhiDevice(), m_swap->extent());
        m_renderer.smaa().bindResources(m_renderer.rt());
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


// ============================================================
// RenderPipeline 辅助方法
// ============================================================

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
                m_renderer.tonemap().bindOutput(m_renderer.rt().aaHdr.view(), m_frameCtx.frameInFlight);
                m_renderer.tonemap().record(cmd, m_renderer.rt(), m_frameCtx.frameInFlight, true, 1.0f);
                m_renderer.writeTimestamp(cmd, m_renderer.kTsTonemap);

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
                    m_renderer.taa().bindResources(m_renderer.rt(), m_frameCtx.frameInFlight);
                    m_renderer.taa().bindOutput(m_frameCtx.swapView, m_frameCtx.frameInFlight);
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
                    m_renderer.smaa().bindResources(m_renderer.rt());
                    m_renderer.smaa().bindOutput(m_frameCtx.swapView);
                    m_renderer.smaa().record(cmd, m_renderer.rt());
                }
                m_renderer.writeTimestamp(cmd, m_renderer.kTsAA);
            }
        } else {
            // No AA: tonemap writes directly to swapchain
            transitionImage(cmd, m_frameCtx.swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
            m_renderer.tonemap().bindOutput(m_frameCtx.swapView, m_frameCtx.frameInFlight);
            m_renderer.tonemap().record(cmd, m_renderer.rt(), m_frameCtx.frameInFlight, true, 1.0f);
            m_renderer.writeTimestamp(cmd, m_renderer.kTsTonemap);
            m_renderer.writeTimestamp(cmd, m_renderer.kTsAA);
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
            m_renderer.tonemap().bindOutput(m_renderer.rt().aaHdr.view(), m_frameCtx.frameInFlight);
            m_renderer.tonemap().record(cmd, m_renderer.rt(), m_frameCtx.frameInFlight);
            m_renderer.writeTimestamp(cmd, m_renderer.kTsTonemap);

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
                m_renderer.taa().bindResources(m_renderer.rt(), m_frameCtx.frameInFlight);
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
                m_renderer.smaa().bindResources(m_renderer.rt());
                m_renderer.smaa().record(cmd, m_renderer.rt());
            }
            m_renderer.writeTimestamp(cmd, m_renderer.kTsAA);

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
            m_renderer.tonemap().bindOutput(m_renderer.rt().ldrTonemap.view(), m_frameCtx.frameInFlight);
            m_renderer.tonemap().record(cmd, m_renderer.rt(), m_frameCtx.frameInFlight);
            m_renderer.writeTimestamp(cmd, m_renderer.kTsTonemap);
            m_renderer.writeTimestamp(cmd, m_renderer.kTsAA);

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
    m_renderer.writeTimestamp(cmd, m_renderer.kTsEnd);
    transitionImage(cmd, m_frameCtx.swapImage, VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
}

void App::setScreenshotConfig(int interval, int oneFrame, const char* dir) {
    m_screenshot.captureInterval = interval;
    m_screenshot.captureOneFrame = oneFrame;
    if (dir && dir[0]) m_screenshot.outputDir = dir;
}

void App::setBackend(const char* name) {
    if (name && name[0]) m_backendName = name;
    std::printf("[init] backend: %s\n", m_backendName.c_str());
}

void App::setInitialShadowMethod(int method) {
    if (method >= 0 && method < kShadowCount && kShadows[method].implemented) {
        m_currentShadowIndex = method;
        std::printf("[init] shadow method set to %d (%s) via CLI\n",
                    method, kShadows[method].name);
    }
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
        m_renderer.imgui().init(*m_device, *m_renderer.rhiDevice(), m_imguiWin->handle(), m_imguiSwap->format(), kFramesInFlight);
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
    if (m_backendName == "d3d12") {
        runD3D12();
        return;
    }
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
        m_renderer.imgui().saveSettings();  // 窗口/控件变更有修改就保存
        applySceneSelection();

        if (!wantKbd && ImGui::IsKeyPressed(ImGuiKey_F2)) {
            startBenchmark();
        }
        if (!wantKbd && ImGui::IsKeyPressed(ImGuiKey_F12)) {
            m_screenshot.manualRequest = true;
            std::printf("[screenshot] F12 pressed, will capture next frame\n");
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

        // ---- Screenshot frame counter ----
        m_screenshot.frameCount++;

        // ---- Swapchain acquire ----
        auto frame = m_swap->acquireNextFrame();
        if (frame.needsResize) { m_swap->recreate(); onSwapchainResized(); continue; }
        m_renderer.setFrameInFlight(frame.frameInFlight);
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
        // RHI 包装器：供 RenderPipeline/FG 等通过 executeRHI 消费
        auto& vkDev = static_cast<rhi::VkRHIDevice&>(*m_renderer.rhiDevice());
        rhi::VkRHICommandBuffer rhiCmd(vkDev, cmd);

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

        // 同步每帧渲染状态到 FrameRenderer（registerPipelineSteps 的 lambda 已捕获引用）
        m_renderer.setFrameState({
            frame.frameInFlight,
            frame.image, frame.view, frame.extent,
            ubo.proj, ubo.view, ubo.proj * ubo.view, ubo.invViewProj,
            m_prevViewProj, m_jitter, m_prevJitter, m_taaBlendAlpha
        });

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
            m_fg.executeRHI(rhiCmd);
            m_fg.applyTimestampsToDebug();

            // 将 FG per-pass GPU 耗时映射到 FrameRenderer 的 profiler 槽位
            {
                float* dst = m_renderer.passTimes(m_frameCtx.frameInFlight);
                float total = 0.0f;
                for (auto& p : m_fg.debug().passes) {
                    float t = p.gpuMs;
                    if (p.name == "GBuffer")       dst[m_renderer.kTsGBuffer]  = t;
                    else if (p.name == "SSAO" || p.name == "GTAO")
                                                    dst[m_renderer.kTsAO]       = t;
                    else if (p.name == "VXGI-Chain") dst[m_renderer.kTsVoxelGI] = t;
                    else if (p.name == "Lighting")  dst[m_renderer.kTsLighting] = t;
                    else if (p.name == "Skybox")    dst[m_renderer.kTsSkybox]   = t;
                    else if (p.name == "Tonemap")   dst[m_renderer.kTsTonemap]  = t;
                    else if (p.name == "TAA" || p.name == "SMAA")
                                                    dst[m_renderer.kTsAA]       = t;
                    total += t;
                }
                dst[m_renderer.kTsEnd] = total;
                if (total > 0.0f) m_renderer.gpuMs() = m_renderer.gpuMs() * 0.9f + total * 0.1f;
            }
        } else {
            // 现有 RenderPipeline 路径
            buildPipelineTable();
            m_renderer.pipeline().executeRHI(rhiCmd);
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

        // ---- Screenshot capture (post-submit, pre-present) ----
        if (m_screenshot.shouldCapture() && !m_swap->hdrEnabled()) {
            // 等待 GPU 完成，再单独提交一次 copy 命令
            vkWaitForFences(m_device->device(), 1, &frame.sync->inFlight, VK_TRUE, UINT64_MAX);

            // ldrTonemap 在 SDR 路径末尾处于 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
            // 直接用 oneShotSubmit 拷到 staging buffer
            oneShotSubmit(*m_device, m_pool, [&](VkCommandBuffer scmd) {
                m_screenshot.recordCopy(scmd, m_renderer.rt().ldrTonemap.image(),
                                        m_renderer.rt().extent);
            });

            // 写 PNG 文件
            char filename[256];
            std::snprintf(filename, sizeof(filename), "%s_%04d.png",
                          m_screenshot.outputPrefix.c_str(),
                          m_screenshot.frameCount);
            m_screenshot.savePng(filename, m_renderer.rt().extent);

            // ---- 回归测试：复制参考图或对比 ----
            std::string baseName = std::string(m_screenshot.outputPrefix) + "_" +
                                   std::to_string(m_screenshot.frameCount);
            if (m_captureRef || m_captureCompare) {
                // 复制截图到 tests/ref/（参考模式）或留 screenshots/ 后对比
                if (m_captureRef) {
                    std::error_code ec;
                    std::filesystem::create_directories("tests/ref", ec);
                    std::string srcPath = m_screenshot.outputDir + "/" + filename;
                    std::string dstPath = "tests/ref/" + baseName + ".png";
                    std::filesystem::copy_file(srcPath, dstPath,
                        std::filesystem::copy_options::overwrite_existing, ec);
                    if (!ec)
                        std::printf("[regress] ref saved: %s\n", dstPath.c_str());
                    else
                        std::fprintf(stderr, "[regress] ref copy failed: %s\n", ec.message().c_str());
                }
                if (m_captureCompare) {
                    RegressionTest regress;
                    regress.setThreshold(m_refThreshold);
                    RegressResult r = regress.compare(baseName, baseName);
                    if (!r.passed) {
                        std::printf("[regress] FAILED — exiting\n");
                        glfwSetWindowShouldClose(m_window->handle(), GLFW_TRUE);
                    }
                }
            }

            // 清除触发标记
            m_screenshot.manualRequest = false;
            if (m_screenshot.captureOneFrame >= 0 &&
                m_screenshot.frameCount >= m_screenshot.captureOneFrame) {
                m_screenshot.captureOneFrame = -1;  // 一次性截图完成
                if (m_exitAfterCapture || m_captureCompare) {
                    glfwSetWindowShouldClose(m_window->handle(), GLFW_TRUE);
                }
            }
        } else if (m_screenshot.shouldCapture() && m_swap->hdrEnabled()) {
            std::printf("[screenshot] HDR mode — screenshot not yet supported (use SDR)\n");
            m_screenshot.manualRequest = false;
        }

        m_swap->present(frame);

        // ---- ImGui debug window ----
        renderDebugWindow();

        // ---- Save viewProj for next frame's reprojection ----
        m_prevViewProj = ubo.viewProj;
    }
}

App::App(ForD3D12) {
    m_backendName = "d3d12";
    // 创建简单 GLFW 窗口（无 Vulkan surface）
    WindowDesc wd; wd.title = "SomeGI D3D12"; wd.width = 800; wd.height = 450;
    m_window = std::make_unique<Window>(wd);
    std::printf("[d3d12] App(ForD3D12) — Vulkan init skipped\n");
}

// D3D12 独立渲染循环（--backend d3d12 时调用）
void App::runD3D12() {
    std::printf("[d3d12] App::runD3D12() — starting\n");
    HWND hwnd = glfwGetWin32Window(m_window->handle());
    auto d3dDevice = rhi::RHIDevice::create(rhi::Backend::D3D12, hwnd, false);
    m_d3d12Device = d3dDevice.get();
    auto swapchain = d3dDevice->createSwapchain(hwnd, 800, 450);
    auto cmdPool = d3dDevice->createCommandPool();
    std::unique_ptr<rhi::RHICommandBuffer> cmdBuf(cmdPool->allocateRaw());
    auto submitFence = d3dDevice->createFence(false);

    // 加载简单三角形 shader（验证图形管线）
    auto loadShader = [&](const char* path, rhi::ShaderStage stage, const char* entry) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "[d3d12] shader not found: %s\n", path); return std::unique_ptr<rhi::RHIShader>(); }
        std::vector<uint8_t> code(std::istreambuf_iterator<char>(f), {});
        rhi::ShaderDesc sd; sd.stage = stage; sd.entryPoint = entry;
        return d3dDevice->createShader(sd, code.data(), code.size());
    };
    auto vs = loadShader("build/shaders_dxil/tri_vs.dxil", rhi::ShaderStage::Vertex, "main");
    auto ps = loadShader("build/shaders_dxil/tri_ps.dxil", rhi::ShaderStage::Fragment, "main");
    if (!vs || !ps) { std::fprintf(stderr, "[d3d12] triangle shaders missing\n"); return; }

    // Graphics PSO（无 descriptor set，仅光栅化三角形）
    rhi::VertexInputState vis;
    vis.bindings = {{0, 20, false}}; // pos(8) + color(12) = 20 bytes
    vis.attributes = {{0, rhi::VertexFormat::Float2, 0, 0}, {1, rhi::VertexFormat::Float3, 8, 0}};
    rhi::GraphicsPSODesc gpsd;
    gpsd.vertexShader = vs.get(); gpsd.fragmentShader = ps.get();
    gpsd.vertexInput = vis; gpsd.topology = rhi::PrimitiveTopology::TriangleList;
    gpsd.renderTargets.colorFormats = {rhi::Format::B8G8R8A8_UNORM};
    gpsd.renderTargets.depthFormat = rhi::Format::Unknown;
    gpsd.renderTargets.sampleCount = 1;
    auto pso = d3dDevice->createGraphicsPSO(gpsd);
    std::printf("[d3d12] graphics PSO created — triangle pipeline\n");

    // 创建 RHI 线性采样器
    rhi::SamplerDesc sampDesc;
    sampDesc.magFilter = rhi::Filter::Linear;
    sampDesc.minFilter = rhi::Filter::Linear;
    sampDesc.mipmapMode = rhi::SamplerMipmapMode::Linear;
    auto linearSampler = d3dDevice->createSampler(sampDesc);
    std::printf("[d3d12] linear sampler created\n");

    // 初始化 TonemapPass（D3D12 路径）
    TonemapPass tonemap;
    tonemap.init(*d3dDevice, *linearSampler);
    std::printf("[d3d12] TonemapPass initialized\n");

    // 创建三角形顶点缓冲
    struct Vertex { float x, y; float r, g, b; };
    Vertex triVerts[] = {
        { 0.0f, -0.5f, 1,0,0 },
        { 0.5f,  0.5f, 0,1,0 },
        {-0.5f,  0.5f, 0,0,1 },
    };
    rhi::BufferDesc vbDesc{};
    vbDesc.size = sizeof(triVerts);
    vbDesc.usage = rhi::BufferUsage::Vertex;
    vbDesc.memory = rhi::MemoryType::HostVisible;
    auto vb = d3dDevice->createBuffer(vbDesc);
    std::memcpy(vb->map(), triVerts, sizeof(triVerts));
    vb->unmap();
    std::printf("[d3d12] vertex buffer: %zu bytes\n", sizeof(triVerts));

    // SSAO compute PSO（多 compute pass 同帧验证）
    std::unique_ptr<rhi::RHIPipelineState> ssaoPSO;
    {
        std::ifstream f("build/shaders_dxil/ssao/ssao.dxil", std::ios::binary);
        if (f) {
            std::vector<uint8_t> bc(std::istreambuf_iterator<char>(f), {});
            rhi::ShaderDesc sd; sd.stage = rhi::ShaderStage::Compute; sd.entryPoint = "comp_main";
            auto cs = d3dDevice->createShader(sd, bc.data(), bc.size());
            rhi::DescSetLayoutDesc dslD;
            auto add = [&](uint32_t vk, rhi::DescriptorType t, uint32_t hlsl) {
                rhi::DescriptorBinding b; b.binding = vk; b.type = t; b.hlslRegister = hlsl;
                dslD.bindings.push_back(b);
            };
            add(0, rhi::DescriptorType::UniformBuffer, 0);
            add(0, rhi::DescriptorType::SampledImage, 0);
            add(1, rhi::DescriptorType::SampledImage, 1);
            add(2, rhi::DescriptorType::StorageImage, 2);
            auto sdsl = d3dDevice->createDescriptorSetLayout(dslD);
            rhi::ComputePSODesc psd; psd.computeShader = cs.get();
            psd.descriptorSetLayouts.push_back(sdsl.get());
            ssaoPSO = d3dDevice->createComputePSO(psd);
            std::printf("[d3d12] SSAO PSO created\n");
        }
    }

    // 创建渲染目标 + push constant buffer
    rhi::TextureDesc hdrDesc; hdrDesc.format = rhi::Format::R16G16B16A16_SFLOAT;
    hdrDesc.width = 800; hdrDesc.height = 450;
    hdrDesc.usage = (rhi::TextureUsage)((uint32_t)rhi::TextureUsage::Sampled | (uint32_t)rhi::TextureUsage::Storage);
    auto hdrTex = d3dDevice->createTexture(hdrDesc);
    auto hdrView = hdrTex->createView({});
    rhi::TextureDesc ldrDesc = hdrDesc; ldrDesc.format = rhi::Format::B8G8R8A8_UNORM;
    auto ldrTex = d3dDevice->createTexture(ldrDesc);
    auto ldrView = ldrTex->createView({});

    rhi::BufferDesc pcDesc{}; pcDesc.size = 16;
    pcDesc.usage = rhi::BufferUsage::Uniform; pcDesc.memory = rhi::MemoryType::HostVisible;
    auto pcBuf = d3dDevice->createBuffer(pcDesc);

    // 获取 tonemap 的 descriptor set (slot 0, frame 0)
    auto& tonemapSets = tonemap.sets();
    if (tonemapSets[0]) {
        tonemapSets[0]->write({
            {0, rhi::DescriptorType::UniformBuffer, nullptr, pcBuf.get()},
            {1, rhi::DescriptorType::SampledImage, hdrView.get()},
            {2, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, linearSampler.get()},
            {3, rhi::DescriptorType::StorageImage, ldrView.get()},
        });
        std::printf("[d3d12] tonemap descriptor set bound\n");
    }
    while (!m_window->shouldClose()) {
        m_window->pollEvents();
        if (glfwGetKey(m_window->handle(), GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(m_window->handle(), GLFW_TRUE);

        auto frame = swapchain->acquireNextFrame();
        if (frame.needsResize) continue;

        // 三角形顶点数据（CPU 端）
        struct Vertex { float x, y; float r, g, b; };
        Vertex tri[] = {
            { 0.0f, -0.5f, 1,0,0 },
            { 0.5f,  0.5f, 0,1,0 },
            {-0.5f,  0.5f, 0,0,1 },
        };

        cmdBuf->begin();

        // 清除 + 渲染三角形到 swapchain
        rhi::RenderingAttachmentInfo colorAttach{};
        colorAttach.view = frame.view.get();
        colorAttach.loadOp = rhi::AttachmentLoadOp::Clear;
        colorAttach.storeOp = rhi::AttachmentStoreOp::Store;
        colorAttach.clearColor[0] = 0.1f; colorAttach.clearColor[1] = 0.2f;
        colorAttach.clearColor[2] = 0.4f; colorAttach.clearColor[3] = 1.0f;
        cmdBuf->beginRendering(&colorAttach, 1, nullptr, frame.width, frame.height);

        // Graphics pipeline: draw triangle
        cmdBuf->bindPipelineState(*pso);
        cmdBuf->setViewport(0, 0, (float)frame.width, (float)frame.height);
        cmdBuf->setScissor(0, 0, frame.width, frame.height);
        cmdBuf->bindVertexBuffer(0, *vb);
        cmdBuf->draw(3);
        cmdBuf->endRendering();

        // Compute pipeline: tonemap dispatch (HDR→LDR)
        if (auto* tPSO = tonemap.pipeline()) {
            cmdBuf->bindPipelineState(*tPSO);
            cmdBuf->bindDescriptorSet(0, *tonemap.sets()[0]);
            uint32_t pc[4] = {0}; // hdrMode=0, exposure=1.0
            std::memcpy(pcBuf->map(), pc, sizeof(pc));
            pcBuf->unmap();
            cmdBuf->pushConstants(rhi::ShaderStage::Compute, pc, sizeof(pc));
            cmdBuf->dispatch((800 + 7) / 8, (450 + 7) / 8);
        }
        // SSAO compute dispatch（第二个 compute pass 同帧运行）
        if (ssaoPSO) {
            cmdBuf->bindPipelineState(*ssaoPSO);
            cmdBuf->dispatch((800 + 7) / 8, (450 + 7) / 8);
        }

        cmdBuf->end();

        rhi::SubmitDesc sub{}; sub.commandBuffer = cmdBuf.get();
        sub.signalFence = submitFence.get();
        d3dDevice->submit(sub);
        d3dDevice->waitForFence(*submitFence, UINT64_MAX);
        submitFence->reset();
        swapchain->present(frame);
    }
    d3dDevice->waitIdle();
    std::printf("[d3d12] loop ended\n");
}

} // namespace somegi
