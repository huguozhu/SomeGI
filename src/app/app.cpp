#include "app.h"
#include "core/window.h"
#include "core/device.h"
#include "core/swapchain.h"
#include "scene/gltf_loader.h"
#include "scene/scene_gpu.h"
#include "scene/env_loader.h"
#include "scene/upload.h"
#include "gi/ibl_technique.h"
#include "gi/gi_technique.h"
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
    { "cube",   "gltf/cube/cube.gltf" },
    { "Sponza", "gltf/Sponza/Sponza.gltf" },
};

// 用 implemented=false 占位的下拉项当 roadmap 用：选中后 applyGiSelection
// 把 effective 落到 IBL 兜底，但 dropdown 标签仍显示原 name + " (未实现)"。
struct GiEntry { const char* name; bool implemented; };
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
    {"RayTracing",         false},  // M9 deferred (no HW RT on Intel UHD 770)
    {"ReSTIR DI",          true},   // C.4 软件版（reservoir resampling on point lights）
};
constexpr int kGiCount = (int)(sizeof(kGis)/sizeof(kGis[0]));

inline const char* giLabel(int i, char* buf, size_t bufSize) {
    if (i < 0 || i >= kGiCount) return "?";
    if (kGis[i].implemented) return kGis[i].name;
    std::snprintf(buf, bufSize, "%s (未实现)", kGis[i].name);
    return buf;
}

constexpr const char* kStatePath = "scene_state.ini";

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
        f << "\n";
    }
}
}

static void transitionImage(VkCommandBuffer cmd, VkImage image,
                            VkImageAspectFlags aspect,
                            VkImageLayout oldL, VkImageLayout newL,
                            VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                            VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask  = srcStage;  b.srcAccessMask = srcAccess;
    b.dstStageMask  = dstStage;  b.dstAccessMask = dstAccess;
    b.oldLayout = oldL;          b.newLayout = newL;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}

App::App() {
    WindowDesc wd; wd.title = "SomeGI [M1 forward]";
    m_window = std::make_unique<Window>(wd);
    m_device = std::make_unique<Device>(*m_window, /*validation=*/true);
    m_swap   = std::make_unique<Swapchain>(*m_device, *m_window);

    // M9：检测 HW RT 支持并更新 dropdown 实现状态。
    m_rtSupported = m_device->features().rayQuery && m_device->features().accelStruct;
    if (m_rtSupported) {
        kGis[10].implemented = true;
        std::printf("[init] HW RT supported — RayTracing GI enabled\n");
    } else {
        std::printf("[init] HW RT not supported — RayTracing GI unavailable\n");
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

    // A.2：GPU timestamp query pool。每帧 2 个 (start + end)，per-in-flight
    // 各占 2 个 query slot。读回时机：下一轮同 frameInFlight 时，上次结果
    // 已经完成（vkWaitForFences 已保证），可安全 vkGetQueryPoolResults。
    {
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = kFramesInFlight * 2;
        VK_CHECK(vkCreateQueryPool(m_device->device(), &qpci, nullptr, &m_timestampPool));
    }

    std::printf("[init] render targets...\n");
    m_rt.create(*m_device, m_swap->extent());
    std::printf("[init] gbuffer pass...\n");
    m_gbuffer.init(*m_device,
                   VK_FORMAT_R8G8B8A8_UNORM,
                   VK_FORMAT_R16G16B16A16_SFLOAT,
                   VK_FORMAT_R8G8B8A8_UNORM,
                   VK_FORMAT_D32_SFLOAT,
                   kMaxTextures);
    // M5.0：RSM 几何 pass（sun-view 4-RT）。本里程碑只跑 record，下游
    // 暂不消费（RsmSamplePass 在 M5.1 接入）。
    std::printf("[init] rsm geometry pass...\n");
    m_rsmGeom.init(*m_device, kMaxTextures);
    // M5.1：RSM gather compute。bindFrame 在 GBuffer 就绪后调一次（init
    // 末尾），swapchain resize 时再调一次（onSwapchainResized）。
    std::printf("[init] rsm sample pass...\n");
    m_rsmSample.init(*m_device);
    // M6 LPV / M7 VXGI 资源要在 lighting bindFrame 之前 ready —— lighting
    // 的 set=0 bindings 10/11/12 绑 LPV grid[0]，14 绑 VXGI 全 mip view。
    std::printf("[init] lpv resources...\n");
    m_lpv.create(*m_device, kLpvResolution);
    std::printf("[init] vxgi resources...\n");
    m_vxgi.create(*m_device, kVxgiResolution);
    // M8 PRT 资源也要在 lighting bindFrame 之前 ready；lighting 的 binding 15
    // 绑 PRT transfer 全 view。
    std::printf("[init] prt resources...\n");
    m_prt.create(*m_device, kPrtResolution);
    std::printf("[init] ddgi resources + pass...\n");
    m_ddgi.create(*m_device);
    // B.5：把 probeStates 全初始化成 1（active）；classify 第一次跑前
    // lighting 已经会读，不能给它垃圾值。
    oneShotSubmit(*m_device, m_pool, [&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, m_ddgi.probeStates().handle(),
                        0, VK_WHOLE_SIZE, 1u);
    });
    m_ddgiPass.init(*m_device);
    m_ddgiPass.bindResources(*m_device, m_ddgi, m_vxgi);
    std::printf("[init] lighting pass...\n");
    m_lighting.init(*m_device);
    m_lighting.bindFrame(*m_device, m_rt, m_gbuffer.frameUboHandle(),
                         m_lpv.current(), m_vxgi, m_prt, m_ddgi,
                         m_ddgi.probeStates().handle());

    std::printf("[init] ssao pass...\n");
    m_ssao.init(*m_device);
    m_ssao.bindFrame(*m_device, m_rt);
    std::printf("[init] gtao pass...\n");
    m_gtao.init(*m_device);
    m_gtao.bindFrame(*m_device, m_rt, m_gbuffer.frameUboHandle());

    std::printf("[init] ssr pass...\n");
    m_ssr.init(*m_device);
    m_ssr.bindFrame(*m_device, m_rt, m_gbuffer.frameUboHandle());

    std::printf("[init] ssgi pass...\n");
    m_ssgi.init(*m_device);
    m_ssgi.bindFrame(*m_device, m_rt, m_gbuffer.frameUboHandle());
    std::printf("[init] gtgi pass...\n");
    m_gtgi.init(*m_device);
    m_gtgi.bindFrame(*m_device, m_rt, m_gbuffer.frameUboHandle());

    // C.3 SDFGI：分配 128³ 三张 3D image，初始化 4 个 sub-pass。
    std::printf("[init] sdfgi resources/pass...\n");
    m_sdfgi.create(*m_device, kSdfgiResolution);
    m_sdfgiPass.init(*m_device);
    m_sdfgiPass.bindResources(*m_device, m_sdfgi, m_vxgi, m_rt, m_gbuffer.frameUboHandle());

    // C.4 ReSTIR DI：reservoir A/B（screen-res RGBA32_UINT）+ light SSBO。
    std::printf("[init] restir resources/pass...\n");
    m_restir.create(*m_device, m_swap->extent(), kRestirMaxLights);
    m_restirPass.init(*m_device, m_rtSupported);

    // M9 RT GI：仅在 HW 支持时初始化。
    if (m_rtSupported) {
        std::printf("[init] rt gi pass...\n");
        m_rtGiPass.init(*m_device);
        m_rtGiInited = true;
    }
    m_restirPass.bindResources(*m_device, m_restir, m_vxgi, m_rt, m_gbuffer.frameUboHandle());

    // M5.1：RsmSamplePass.bindFrame 需要 GBuffer 的 frameUbo + RsmGeom 的
    // rsmFrameUbo + 3 张 RSM RT。三者在 init 此刻已就绪，绑一次。
    m_rsmSample.bindFrame(*m_device, m_rt,
        m_gbuffer.frameUboHandle(),
        m_rsmGeom.frameUboHandle(),
        m_rsmGeom.position(), m_rsmGeom.normal(), m_rsmGeom.flux());

    // M6.0：LPV inject pass。grid 的几何参数（gridMin / cellSize）在
    // applySceneSelection 里随 AABB 重算。bindResources 把 RSM 3 张 +
    // LPV current() 的 3 张 3D image 写到 inject 的 set=0；inject 总是
    // 写 grid[0]，不参 propagate 的 ping-pong。
    std::printf("[init] lpv inject pass...\n");
    m_lpvInject.init(*m_device, RsmGeometryPass::kRsmSize);

    // M7 VXGI：voxel grid 已在 lighting init 前创建（lighting bindFrame
    // 引用 vxgi.fullView）。这里只 init 体素化 / inject / mipmap 子 pass。
    std::printf("[init] vxgi voxelize/inject/mipmap pass...\n");
    m_vxgiVoxelize.init(*m_device, kMaxTextures);
    m_vxgiInject.init(*m_device, RsmGeometryPass::kRsmSize);
    m_vxgiInject.bindResources(*m_device,
        m_rsmGeom.position(), m_rsmGeom.flux(), m_vxgi);
    m_vxgiMipmap.init(*m_device, m_vxgi.mipLevels());
    m_vxgiMipmap.bindResources(*m_device, m_vxgi);
    m_vxgiAniso.init(*m_device, m_vxgi.mipLevels());
    m_vxgiAniso.bindResources(*m_device, m_vxgi);
    m_vxgiRelight.init(*m_device);
    m_vxgiRelight.bindResources(*m_device, m_vxgi, m_vxgi.relightScratch().view());

    std::printf("[init] prt bake pass...\n");
    m_prtBake.init(*m_device);
    m_prtBake.bindResources(*m_device, m_vxgi, m_prt);
    // inject 总把 grid 索引 0 当 dst —— 跟"current() 永远是 grid[curIdx=0]
    // 在 inject 之后"对齐。propagate 之后会做 ping-pong swap，但 inject
    // 只写 grid[0]，bindResources 一次就够。
    m_lpvInject.bindResources(*m_device,
        m_rsmGeom.position(), m_rsmGeom.normal(), m_rsmGeom.flux(),
        m_lpv.current(), m_lpv.gv());
    std::printf("[init] lpv propagate pass...\n");
    m_lpvProp.init(*m_device);
    // bindResources 把两组 ping-pong descriptor set 都填好。第 0 组 src=
    // grid[0]→dst=grid[1]；第 1 组反过来。record 时按 srcIdx 选哪组。
    m_lpvProp.bindResources(*m_device, m_lpv.current(), m_lpv.next(), m_lpv.gv());

    // Bootstrap hdrPrev: SSR's first read happens before any copy has run, so
    // the image must be in SHADER_READ_ONLY with deterministic (cleared)
    // content. One-shot clear to zero — SSR will produce fade-modulated zero
    // reflections for one frame, then per-frame copy keeps it fresh.
    bootstrapHdrPrev();
    bootstrapSsgiTemporal();

    // Bake skybox.hdr once — used by SkyboxPass (always) and IBLTechnique.
    // Kept alive for the lifetime of the App.
    std::printf("[init] env (skybox.hdr) load + bake...\n");
    bakeEnvIbl();
    std::printf("[init] env bake done.\n");

    std::printf("[init] skybox pass...\n");
    m_skybox.init(*m_device, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_D32_SFLOAT);
    m_skybox.bindEnv(*m_device, m_envIbl.envCube.view(), m_envIbl.linear);

    // Load all scenes' persisted view + lighting before the first scene apply.
    {
        PersistedAll p = loadAllSceneStates();
        m_sceneStates = std::move(p.scenes);
        // Resume the scene that was active at last shutdown, if known.
        if (!p.lastScene.empty()) {
            constexpr int kSceneCount = (int)(sizeof(kScenes)/sizeof(kScenes[0]));
            for (int i = 0; i < kSceneCount; ++i) {
                if (p.lastScene == kScenes[i].name) { m_currentSceneIndex = i; break; }
            }
        }
    }

    // Initial scene load (camera framed inside applySceneSelection).
    applySceneSelection();   // 内部包含 M9 AS build + bindFrame（RT 支持时）

    std::printf("[init] GI technique attach...\n");
    applyGiSelection();   // honor m_currentGiIndex (default 1 = IBL)

    std::printf("[init] tonemap pass...\n");
    m_tonemap.init(*m_device, m_sceneGpu.linearSampler);
    m_tonemap.bindTargets(*m_device, m_rt);
    std::printf("[init] imgui pass...\n");
    m_imgui.init(*m_device, m_window->handle(), m_swap->format(), kFramesInFlight);
    std::printf("[init] all set up, entering main loop.\n");
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
    baker.bake(*m_device, m_pool, env, m_envIbl);
}

void App::loadAndUploadScene(const std::filesystem::path& gltfPath) {
    std::string err;
    if (!loadGltf(gltfPath, m_scene, err)) {
        throw std::runtime_error("loadGltf: " + err);
    }
    std::printf("[scene] %s\n  vertices=%zu  indices=%zu  meshes=%zu  materials=%zu  textures=%zu\n  AABB min=(%.2f,%.2f,%.2f) max=(%.2f,%.2f,%.2f)\n",
        gltfPath.string().c_str(),
        m_scene.vertices.size(), m_scene.indices.size(),
        m_scene.meshes.size(), m_scene.materials.size(), m_scene.textures.size(),
        m_scene.aabbMin.x, m_scene.aabbMin.y, m_scene.aabbMin.z,
        m_scene.aabbMax.x, m_scene.aabbMax.y, m_scene.aabbMax.z);

    std::printf("[scene] uploading to GPU...\n");
    uploadScene(*m_device, m_pool, m_scene, m_sceneGpu);
    std::printf("[scene] GPU upload done.\n");
}

void App::applySceneSelection() {
    if (m_currentSceneIndex == m_sceneIndexApplied) return;
    constexpr int kSceneCount = (int)(sizeof(kScenes)/sizeof(kScenes[0]));
    if (m_currentSceneIndex < 0 || m_currentSceneIndex >= kSceneCount) {
        m_currentSceneIndex = m_sceneIndexApplied >= 0 ? m_sceneIndexApplied : 1;
        return;
    }

    if (m_sceneIndexApplied >= 0) {
        // Mid-run switch: snapshot the outgoing scene's state into the in-memory
        // map (we'll write the whole file once the switch completes, so
        // `last_scene` matches the new active scene).
        m_sceneStates[kScenes[m_sceneIndexApplied].name] = captureSceneState();
        m_device->waitIdle();
        destroySceneSamplers(*m_device, m_sceneGpu);
        m_sceneGpu = SceneGpu{};
        m_scene = SceneCpu{};
    }

    auto path = std::filesystem::path(SOMEGI_ASSET_DIR) / kScenes[m_currentSceneIndex].relPath;
    loadAndUploadScene(path);

    // M9：场景已加载，构建加速结构 + bindFrame（仅 HW 支持时）。
    if (m_rtSupported) {
        m_rtAS.build(*m_device, m_pool, m_scene, m_sceneGpu);
        m_rtGiBound = (m_rtAS.instanceCount() > 0);
        if (m_rtGiBound) {
            m_rtGiPass.bindFrame(*m_device, m_rt, m_gbuffer.frameUboHandle(), m_rtAS, m_sceneGpu);
        }
        // M10：TLAS 就绪，绑定到 ReSTIR RT shade pipeline
        if (m_rtAS.instanceCount() > 0) {
            m_restirPass.bindResourcesRt(*m_device, m_restir, m_rt,
                m_gbuffer.frameUboHandle(), m_rtAS.tlas());
        }
    }

    m_gbuffer.bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size());
    m_rsmGeom.bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size());
    m_vxgiVoxelize.bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size(), m_vxgi);
    if (m_sceneIndexApplied >= 0) {
        // Tonemap pass cached the old sampler; old one was destroyed above.
        m_tonemap.destroy();
        m_tonemap.init(*m_device, m_sceneGpu.linearSampler);
        m_tonemap.bindTargets(*m_device, m_rt);
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

    // M6 LPV：32³ 网格按 scene AABB 摆。每边 padding 5% 防边界漏；cellSize
    // 选 max axis 平均出来，让 grid 是均匀立方体（容易 trilinear 插值）。
    {
        glm::vec3 padded = d * 1.10f;
        float maxExtent = std::max({padded.x, padded.y, padded.z});
        m_lpvCellSize = maxExtent / float(kLpvResolution);
        glm::vec3 gridHalf = glm::vec3(m_lpvCellSize * float(kLpvResolution) * 0.5f);
        m_lpvGridMin = c - gridHalf;
    }

    // M7 VXGI：128³ 网格同样按 scene AABB 摆，但 cell 更细 → 4× LPV 的精度。
    {
        glm::vec3 padded = d * 1.10f;
        float maxExtent = std::max({padded.x, padded.y, padded.z});
        m_vxgiCellSize = maxExtent / float(kVxgiResolution);
        glm::vec3 gridHalf = glm::vec3(m_vxgiCellSize * float(kVxgiResolution) * 0.5f);
        m_vxgiGridMin = c - gridHalf;
    }

    // C.4 ReSTIR DI demo lights：按 scene AABB 摆 8 个 point light。
    rebuildDemoLights();

    // M8 PRT：与 LPV 同 32³，但语义不同 —— 存的是 visibility transfer SH。
    {
        glm::vec3 padded = d * 1.10f;
        float maxExtent = std::max({padded.x, padded.y, padded.z});
        m_prtCellSize = maxExtent / float(kPrtResolution);
        glm::vec3 gridHalf = glm::vec3(m_prtCellSize * float(kPrtResolution) * 0.5f);
        m_prtGridMin = c - gridHalf;
    }
    m_prtBaked = false;   // scene 切换 → bake 失效，下一帧 main loop 入口重 bake

    // M11 DDGI：probe grid 按 scene AABB 摆，padding 5%；间距 = padded / probes。
    {
        glm::vec3 padded = d * 1.05f;
        m_ddgiSpacing = glm::vec3(
            padded.x / float(DdgiResources::kProbesX - 1),
            padded.y / float(DdgiResources::kProbesY - 1),
            padded.z / float(DdgiResources::kProbesZ - 1));
        glm::vec3 half = padded * 0.5f;
        m_ddgiOrigin = c - half;
    }
    m_ddgiAtlasInited = false;   // 重新初始化 atlas（清零等价于 first-time）

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
    m_ssao.radius = std::max(0.1f, glm::length(d) * 0.005f);

    // SSR maxDist also scales with scene: about half the diagonal lets a
    // reflection ray cross the frame's worth of geometry. Cube ~1200,
    // Sponza ~20.
    m_ssr.maxDist = std::max(2.0f, glm::length(d) * 0.5f);

    // SSGI maxDist: shorter — diffuse contributions decay quickly with
    // distance, so a quarter of the diagonal is plenty. Reduces wasted
    // march steps. Cube ~600, Sponza ~10.
    m_ssgi.maxDist = std::max(1.0f, glm::length(d) * 0.25f);

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
    if (m_device) m_device->waitIdle();
    m_imgui.destroy();
    m_tonemap.destroy();
    m_rtGiPass.destroy();
    m_rtAS.destroy();
    m_skybox.destroy();
    m_ssgi.destroy();
    m_gtgi.destroy();
    m_ssr.destroy();
    m_ssao.destroy();
    m_gtao.destroy();
    m_lighting.destroy();   // pipeline references IBL DSL — must die before GI tech
    m_ddgiPass.destroy();
    m_ddgi.destroy();
    m_prtBake.destroy();
    m_prt.destroy();
    m_vxgiMipmap.destroy();
    m_vxgiAniso.destroy();
    m_vxgiRelight.destroy();
    m_sdfgiPass.destroy();
    m_sdfgi.destroy();
    m_restirPass.destroy();
    m_restir.destroy();
    m_vxgiInject.destroy();
    m_vxgiVoxelize.destroy();
    m_vxgi.destroy();
    m_lpvProp.destroy();
    m_lpvInject.destroy();
    m_lpv.destroy();
    m_rsmSample.destroy();
    m_rsmGeom.destroy();
    m_gbuffer.destroy();
    m_giTech.reset();       // GI onDetach (releases borrow of m_envIbl)
    if (m_device) m_envIbl.destroy(*m_device);
    m_rt.destroy();
    if (m_device) destroySceneSamplers(*m_device, m_sceneGpu);
    if (m_timestampPool) vkDestroyQueryPool(m_device->device(), m_timestampPool, nullptr);
    if (m_pool) vkDestroyCommandPool(m_device->device(), m_pool, nullptr);
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

    if (!m_giTech) {
        if (m_device) m_device->waitIdle();
        m_giTech = std::make_unique<IBLTechnique>();
        GIContext gctx{};
        gctx.device = m_device.get();
        gctx.oneShotPool = m_pool;
        gctx.iblBaked = &m_envIbl;
        m_giTech->onAttach(gctx);
        m_lighting.setTechnique(m_giTech.get());
        std::printf("[GI] IBLTechnique attached (set=1 bound)\n");
    }

    // Preset：下拉里 SSGI / RSM 两个屏幕空间技术互斥。选 SSGI 时打开
    // m_ssgi、关 m_rsmSample；选 RSM 反过来。其他模式（None/IBL）两者
    // 都关，rsmGI / ssgi 都被 clear path 抹成 0，lighting.slang 的 lerp
    // 退化成纯 IBL diffuse。
    m_ssgi.enabled      = (effective == 2);
    m_rsmSample.enabled = (effective == 3);
    m_lpvEnabled        = (effective == 4);
    m_vxgiEnabled       = (effective == 5);
    m_prtEnabled        = (effective == 6);
    m_ddgiEnabled       = (effective == 7);
    m_gtgi.enabled      = (effective == 8);
    m_sdfgiPass.enabled = (effective == 9);
    // M9 RT GI (index 10) — 不用额外 enabled flag，render loop 检查 m_rtGiBound && m_giIndexApplied == 10
    m_restirPass.enabled = (effective == 11);   // C.4

    m_giIndexApplied = effective;
    std::printf("[GI] applied technique index=%d (UI=%d SSGI=%d RSM=%d LPV=%d VXGI=%d PRT=%d DDGI=%d GTGI=%d SDFGI=%d RT=%d ReSTIR=%d)\n",
                m_giIndexApplied, m_currentGiIndex,
                m_ssgi.enabled ? 1 : 0, m_rsmSample.enabled ? 1 : 0,
                m_lpvEnabled ? 1 : 0, m_vxgiEnabled ? 1 : 0,
                m_prtEnabled ? 1 : 0, m_ddgiEnabled ? 1 : 0,
                m_gtgi.enabled ? 1 : 0,
                m_sdfgiPass.enabled ? 1 : 0,
                effective == 10 ? 1 : 0,
                m_restirPass.enabled ? 1 : 0);
}

void App::onSwapchainResized() {
    m_device->waitIdle();
    m_rt.destroy();
    m_rt.create(*m_device, m_swap->extent());
    m_lighting.bindFrame(*m_device, m_rt, m_gbuffer.frameUboHandle(),
                         m_lpv.current(), m_vxgi, m_prt, m_ddgi,
                         m_ddgi.probeStates().handle());
    m_ssao.bindFrame(*m_device, m_rt);
    m_gtao.bindFrame(*m_device, m_rt, m_gbuffer.frameUboHandle());
    m_ssr.bindFrame(*m_device, m_rt, m_gbuffer.frameUboHandle());
    m_ssgi.bindFrame(*m_device, m_rt, m_gbuffer.frameUboHandle());
    m_gtgi.bindFrame(*m_device, m_rt, m_gbuffer.frameUboHandle());
    // SDFGI trace 也读 rt.gNormalRough/depth/ssgi → resize 后重绑。
    m_sdfgiPass.bindResources(*m_device, m_sdfgi, m_vxgi, m_rt, m_gbuffer.frameUboHandle());
    // ReSTIR：reservoir image 跟 swapchain，需重建；lightBuffer 跨帧持久。
    m_restir.resize(*m_device, m_swap->extent());
    m_restirPass.bindResources(*m_device, m_restir, m_vxgi, m_rt, m_gbuffer.frameUboHandle());
    m_restirOutInited = false;
    m_restirBootstrapped = false;
    m_rsmSample.bindFrame(*m_device, m_rt,
        m_gbuffer.frameUboHandle(),
        m_rsmGeom.frameUboHandle(),
        m_rsmGeom.position(), m_rsmGeom.normal(), m_rsmGeom.flux());
    // M9：swapchain resize 后 rtGI 换新 view → 重绑 RT pass。
    if (m_rtSupported && m_rtGiBound) {
        m_rtGiPass.bindFrame(*m_device, m_rt, m_gbuffer.frameUboHandle(), m_rtAS, m_sceneGpu);
        // M10：resize 后重绑 ReSTIR RT shade 的 TLAS
        m_restirPass.bindResourcesRt(*m_device, m_restir, m_rt,
            m_gbuffer.frameUboHandle(), m_rtAS.tlas());
    }
    m_tonemap.bindTargets(*m_device, m_rt);
    bootstrapHdrPrev();   // fresh hdrPrev image — clear it before SSR can read
    bootstrapSsgiTemporal();
}

void App::rebuildDemoLights() {
    // demo lights：按 scene AABB 摆若干 point light。
    // 位置：4 个 ceiling 角 + 4 个 floor 角（默认 8 盏）。当 m_demoLightCount
    // 改了，只取前 N 盏。颜色：HSV 环绕；intensity：m_demoLightIntensity。
    m_demoLights.clear();
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
    int n = std::clamp(m_demoLightCount, 0, 8);
    for (int i = 0; i < n; ++i) {
        PointLightCpu L{};
        L.pos = corners[i];
        L.radius = 0.0f;
        L.color = colors[i];
        L.intensity = m_demoLightIntensity;
        m_demoLights.push_back(L);
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
        b.image = m_vxgi.image().image();
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, m_vxgi.mipLevels(), 0, 1};
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
        VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT, 0, m_vxgi.mipLevels(), 0, 1};
        vkCmdClearColorImage(cmd, m_vxgi.image().image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &rg);
        barrierAllVxgiMips(cmd,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        // 2. voxelize：写 mip 0
        m_vxgiVoxelize.record(cmd, m_scene, m_sceneGpu,
            m_vxgiGridMin, m_vxgiCellSize, kVxgiResolution);

        // 3. mipmap：内部把 src mip 转 SHADER_READ_ONLY，dst 保持 GENERAL。
        m_vxgiMipmap.record(cmd, m_vxgi);

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
            mb.image = m_vxgi.image().image();
            mb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                   m_vxgi.mipLevels() - 1, 1, 0, 1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cmd, &di);
        }

        // 5. prtTransfer A/B/C/D/E: UNDEFINED → GENERAL（bake 写 SH16 五张 atlas）
        VkImage prtImgs[5] = {m_prt.image().image(),
                              m_prt.imageB().image(),
                              m_prt.imageC().image(),
                              m_prt.imageD().image(),
                              m_prt.imageE().image()};
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
        m_prtBake.record(cmd,
            m_prtGridMin, m_prtCellSize, kPrtResolution,
            m_vxgiGridMin, m_vxgiCellSize, kVxgiResolution,
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
    std::printf("[PRT] bake complete (%u^3 cells, 64 rays/cell)\n", kPrtResolution);
}

void App::bootstrapSsgiTemporal() {
    // 把 ssgi 与 ssgiPrev 都清成 0 + 转 SHADER_READ_ONLY，让第一帧 SSGI on
    // 时 copy / sample 都有合法 layout。两张 image 同形 + 同初始内容。
    oneShotSubmit(*m_device, m_pool, [&](VkCommandBuffer cmd) {
        VkImage imgs[2] = {m_rt.ssgi.image(), m_rt.ssgiPrev.image()};
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
        transitionImage(cmd, m_rt.hdrPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkClearColorValue zero{};
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, m_rt.hdrPrev.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &r);
        transitionImage(cmd, m_rt.hdrPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
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
                    m_dtMs, m_fpsAvg, m_gpuMs);
        ImGui::Separator();
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
                    if (ImGui::Selectable(itemLabel, sel)) m_currentGiIndex = i;
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }
        if (m_giTech) {
            ImGui::Text("Active: %s", m_giTech->name());
            m_giTech->drawUI();
        }
        ImGui::Separator();
        ImGui::Text("SSAO");
        // AO method radio
        const char* aoLabels[] = {"None", "SSAO", "GTAO"};
        int aoIdx = (int)m_aoMethod;
        if (ImGui::BeginCombo("AO method", aoLabels[aoIdx])) {
            for (int i = 0; i < 3; ++i) {
                if (ImGui::Selectable(aoLabels[i], aoIdx == i)) m_aoMethod = (AOMethod)i;
            }
            ImGui::EndCombo();
        }
        if (m_aoMethod == AOMethod::SSAO) {
            ImGui::DragFloat("SSAO radius",  &m_ssao.radius, 0.05f, 0.05f, 100.0f, "%.3f");
            ImGui::DragFloat("SSAO bias",    &m_ssao.bias,   0.005f, 0.0f, 0.5f);
            ImGui::SliderInt("SSAO samples", &m_ssao.sampleCount, 4, 64);
        } else if (m_aoMethod == AOMethod::GTAO) {
            ImGui::SliderInt("GTAO slices",   &m_gtao.sliceCount, 1, 8);
            ImGui::SliderInt("GTAO samples/slice", &m_gtao.samplesPerSlice, 2, 16);
            ImGui::DragFloat("GTAO radius (px)", &m_gtao.radiusPixels, 1.0f, 4.0f, 256.0f);
            ImGui::DragFloat("GTAO falloff", &m_gtao.falloff, 0.1f, 0.5f, 50.0f);
        }
        ImGui::Separator();
        ImGui::Text("SSR");
        ImGui::Checkbox("SSR enabled",     &m_ssr.enabled);
        ImGui::SliderInt("SSR steps",      &m_ssr.maxSteps, 8, 128);
        ImGui::DragFloat("SSR max dist",   &m_ssr.maxDist,  0.5f, 0.1f, 1000.0f);
        ImGui::DragFloat("SSR thickness",  &m_ssr.thickness, 0.005f, 0.001f, 0.5f);
        ImGui::DragFloat("SSR rough threshold", &m_ssr.roughThreshold, 0.01f, 0.0f, 1.0f);
        ImGui::Separator();
        ImGui::Text("SSGI");
        ImGui::Checkbox("SSGI enabled",    &m_ssgi.enabled);
        ImGui::SliderInt("SSGI samples",   &m_ssgi.sampleCount, 2, 32);
        ImGui::SliderInt("SSGI steps",     &m_ssgi.maxSteps, 8, 64);
        ImGui::DragFloat("SSGI max dist",  &m_ssgi.maxDist,  0.2f, 0.1f, 500.0f);
        ImGui::DragFloat("SSGI thickness", &m_ssgi.thickness, 0.005f, 0.001f, 0.5f);
        ImGui::Separator();
        ImGui::Text("GTGI (Sucker Punch 2024 horizon-based GI)");
        ImGui::Checkbox("GTGI enabled",   &m_gtgi.enabled);
        ImGui::SliderInt("GTGI slices",   &m_gtgi.sliceCount, 1, 8);
        ImGui::SliderInt("GTGI samples/slice", &m_gtgi.samplesPerSlice, 2, 16);
        ImGui::DragFloat("GTGI radius (px)", &m_gtgi.radiusPixels, 1.0f, 4.0f, 256.0f);
        ImGui::DragFloat("GTGI falloff", &m_gtgi.falloff, 0.1f, 0.5f, 50.0f);
        ImGui::Separator();
        // M9 RT GI：在 HW 支持时显示状态信息
        if (m_rtSupported) {
            ImGui::Text("RT GI (M9 硬件光线追踪)");
            ImGui::Text("RT GI %s（GI 下拉切到 'RayTracing'）",
                        m_giIndexApplied == 10 ? "active" : "off");
            if (m_rtAS.instanceCount() > 0) {
                ImGui::Text("TLAS instances: %u", m_rtAS.instanceCount());
            }
            ImGui::Separator();
        }

        ImGui::Text("SDFGI (C.3 Godot4 风格 - JFA UDF + sphere-trace)");
        ImGui::Text("SDFGI %s（GI 下拉切到 'SDFGI'）",
                    m_sdfgiPass.enabled ? "active" : "off");
        ImGui::SliderInt("SDFGI rays", &m_sdfgiPass.numRays, 1, 16);
        ImGui::SliderInt("SDFGI maxSteps", &m_sdfgiPass.maxSteps, 8, 96);
        ImGui::DragFloat("SDFGI rayMax (cells)", &m_sdfgiPass.rayMaxCells, 1.0f, 8.0f, 256.0f);
        ImGui::DragFloat("SDFGI hitEps (cells)", &m_sdfgiPass.hitEpsCells, 0.05f, 0.1f, 2.0f);
        ImGui::DragFloat("SDFGI seedThr", &m_sdfgiPass.seedThreshold, 0.005f, 0.0f, 0.5f);
        ImGui::Separator();
        bool restirUsingRt = m_rtSupported && m_rtGiBound && m_restirPass.enabled;
        ImGui::Text("ReSTIR DI (%s - reservoir resampling on point lights)",
                    restirUsingRt ? "M10 HW RT visibility" : "C.4 voxel visibility");
        ImGui::Text("ReSTIR %s（GI 下拉切到 'ReSTIR DI'）",
                    m_restirPass.enabled ? "active" : "off");
        if (ImGui::SliderInt("ReSTIR demo lights", &m_demoLightCount, 0, 8)) {
            rebuildDemoLights();
        }
        if (ImGui::DragFloat("ReSTIR light intensity", &m_demoLightIntensity, 0.1f, 0.0f, 50.0f)) {
            rebuildDemoLights();
        }
        ImGui::SliderInt("ReSTIR M (candidates)", &m_restirPass.numCandidates, 1, 32);
        ImGui::SliderInt("ReSTIR K (neighbors)", &m_restirPass.numNeighbors, 0, 8);
        ImGui::DragFloat("ReSTIR spatial radius (px)", &m_restirPass.spatialRadius, 1.0f, 4.0f, 96.0f);
        ImGui::SliderInt("ReSTIR shadow steps", &m_restirPass.shadowSteps, 0, 16);
        ImGui::DragFloat("ReSTIR intensity scale", &m_restirPass.intensityScale, 0.05f, 0.0f, 8.0f);
        ImGui::Separator();
        ImGui::Text("RSM");
        ImGui::Checkbox("RSM enabled",     &m_rsmSample.enabled);
        ImGui::SliderInt("RSM samples",    &m_rsmSample.sampleCount, 4, 128);
        ImGui::DragFloat("RSM radius",     &m_rsmSample.radius,    0.005f, 0.001f, 0.5f);
        ImGui::DragFloat("RSM intensity",  &m_rsmSample.intensity, 0.05f,  0.0f, 20.0f);
        ImGui::Separator();
        ImGui::Text("LPV (32^3 grid)");
        ImGui::Checkbox("LPV enabled",     &m_lpvEnabled);
        ImGui::SliderInt("LPV iterations", &m_lpvProp.iterations, 0, 16);
        ImGui::DragFloat("LPV amplifier",  &m_lpvProp.occlusionAmplifier, 0.05f, 0.0f, 5.0f);
        ImGui::DragFloat("LPV GV occlusion", &m_lpvProp.gvOcclusionStrength, 0.05f, 0.0f, 5.0f);
        ImGui::Text("LPV cell=%.2f gridMin=(%.1f %.1f %.1f)",
                    m_lpvCellSize, m_lpvGridMin.x, m_lpvGridMin.y, m_lpvGridMin.z);
        ImGui::Separator();
        ImGui::Text("VXGI (128^3 voxel + cone trace)");
        ImGui::Checkbox("VXGI enabled", &m_vxgiEnabled);
        ImGui::Text("VXGI cell=%.3f mipLevels=%u",
                    m_vxgiCellSize, m_vxgi.mipLevels());
        // C.2 multi-bounce relight：每帧把 indirect cone-gather 加回 voxel mip 0
        // → 下一帧 lighting 的 cone trace 看到 2-bounce 间接光。
        ImGui::Checkbox("VXGI multi-bounce relight (C.2)", &m_vxgiRelightEnabled);
        ImGui::SliderFloat("Relight bounce strength", &m_vxgiRelightStrength,
                           0.0f, 4.0f, "%.2f");
        ImGui::Separator();
        ImGui::Text("PRT (32^3 visibility transfer SH)");
        ImGui::Checkbox("PRT enabled", &m_prtEnabled);
        // SH order toggle：bake 永远是 SH16（16 系数），lighting 按 toggle 选 4/9/16。
        // SH16 的 l=3 项 A_3=0（Ramamoorthi-Hanrahan）→ diffuse 与 SH9 数值
        // 等价；SH16 选项主要为 toggle 验证 + 后续 specular PRT 预留基础。
        const char* shLabels[] = {
            "SH4  (order-1, 4 coefs)",
            "SH9  (order-2, 9 coefs)",
            "SH16 (order-3, 16 coefs; A_3=0 → diffuse = SH9)"
        };
        ImGui::Combo("PRT SH order", &m_prtShOrder, shLabels, 3);
        if (m_prtShOrder < 0) m_prtShOrder = 0;
        if (m_prtShOrder > 2) m_prtShOrder = 2;
        ImGui::Text("PRT cell=%.2f baked=%s", m_prtCellSize, m_prtBaked ? "yes" : "no");
        if (ImGui::Button("Re-bake PRT")) m_prtBaked = false;
        ImGui::Separator();
        ImGui::Text("DDGI (8x4x8 probes, octahedral atlas)");
        ImGui::Checkbox("DDGI enabled", &m_ddgiEnabled);
        ImGui::Text("DDGI spacing=(%.1f %.1f %.1f) origin=(%.1f %.1f %.1f)",
                    m_ddgiSpacing.x, m_ddgiSpacing.y, m_ddgiSpacing.z,
                    m_ddgiOrigin.x, m_ddgiOrigin.y, m_ddgiOrigin.z);
        ImGui::Separator();
        ImGui::Text("Mouse RMB: rotate  WASD/QE: move  Shift: fast");
    }
    ImGui::End();
    (void)io;
}

void App::run() {
    auto last = std::chrono::high_resolution_clock::now();
    float fpsTimer = 0;
    int fpsFrames = 0;
    while (!m_window->shouldClose()) {
        m_window->pollEvents();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        m_dtMs = dt * 1000.0f;
        fpsTimer += dt; fpsFrames++;
        if (fpsTimer >= 0.5f) {
            m_fpsAvg = fpsFrames / fpsTimer;
            fpsTimer = 0; fpsFrames = 0;
            // A.2 console echo —— sweep 时拿来汇总各模式 ms。
            std::printf("[profile] fps=%.0f cpu=%.2fms gpu=%.2fms gi=%d\n",
                        m_fpsAvg, m_dtMs, m_gpuMs, m_giIndexApplied);
        }

        // 不要在 ImGui 想要键盘/鼠标时给相机
        m_imgui.newFrame();
        bool wantMouse = ImGui::GetIO().WantCaptureMouse;
        bool wantKbd   = ImGui::GetIO().WantCaptureKeyboard;
        if (!wantMouse && !wantKbd) {
            m_flyer.update(m_camera, dt, m_window->handle());
        }
        buildUI();
        applySceneSelection();  // user-driven scene switch from UI dropdown
        applyGiSelection();     // user-driven GI switch from UI dropdown

        // M8 PRT 一次性 bake：scene 切换 / 第一帧时执行；oneShot 内部
        // waitIdle 不会让 main loop 卡顿，但这一帧会有明显停顿（visible
        // 给用户的预期：PRT 模式或场景切换时短暂等待）。
        if (!m_prtBaked) {
            bakePrt();
            m_prtBaked = true;
        }

        auto frame = m_swap->acquireNextFrame();
        if (frame.needsResize) { m_swap->recreate(); onSwapchainResized(); continue; }
        if (frame.extent.width != m_rt.extent.width || frame.extent.height != m_rt.extent.height) {
            onSwapchainResized();
        }

        // Update FrameUBO
        FrameUBO ubo{};
        ubo.view = m_camera.view();
        ubo.proj = m_camera.proj((float)m_rt.extent.width / (float)m_rt.extent.height);
        ubo.viewProj = ubo.proj * ubo.view;
        ubo.invViewProj = glm::inverse(ubo.viewProj);
        ubo.prevViewProj = m_prevViewProj;   // B.4 SSGI 时序 reproject 用
        ubo.cameraPos = glm::vec4(m_camera.position, 0);
        ubo.sunDir = glm::vec4(glm::normalize(m_sunDir), 0);
        ubo.sunColor_intensity = glm::vec4(1.0f, 0.95f, 0.85f, m_sunIntensity);
        ubo.ambient = glm::vec4(m_ambient, 0);
        int specMips = 0;
        if (auto* ibl = dynamic_cast<IBLTechnique*>(m_giTech.get())) {
            specMips = (int)ibl->specularMipCount();
        }
        // counts.z = "indirect lighting enabled" (IBL/SSGI/RSM/LPV 都算).
        // 0 = None → lighting.slang takes hemispheric-ambient fallback.
        // 1 = IBL（含 SSGI/RSM/LPV 的 lerp 叠加） → 走 evalIBLDiffuse + 各路混合。
        //     SSGI 通过 ssgi.a 自门控；RSM 通过 counts.w；LPV 通过 lpvCounts.y。
        // counts.w = RSM 启用闸门（0 / 1）。
        int indirectEnabled = (m_giIndexApplied >= 1) ? 1 : 0;
        int rsmEnabled = m_rsmSample.enabled ? 1 : 0;
        ubo.counts = glm::ivec4((int)m_scene.materials.size(), specMips, indirectEnabled, rsmEnabled);
        // M6 LPV：lpvCounts.x=gridResolution, .y=lpvEnabled。
        // lpvGridMinCell.xyz=gridMin, .w=cellSize（在 applySceneSelection
        // 里按 AABB 重算）。lighting.slang 用这两个把 worldPos → grid UV。
        ubo.lpvCounts = glm::ivec4((int)kLpvResolution, m_lpvEnabled ? 1 : 0, 0, 0);
        ubo.lpvGridMinCell = glm::vec4(m_lpvGridMin, m_lpvCellSize);
        // M7 VXGI：vxgiCounts.x=gridResolution, .y=enabled, .z=mipLevels。
        ubo.vxgiCounts = glm::ivec4((int)kVxgiResolution, m_vxgiEnabled ? 1 : 0,
                                     (int)m_vxgi.mipLevels(), 0);
        ubo.vxgiGridMinCell = glm::vec4(m_vxgiGridMin, m_vxgiCellSize);
        // M8 PRT：把 sun 投到 SH order-1。lightSH[k] = I·color · Y_k(d_sun)。
        // d_sun 取"从 surface 到 sun"的方向 = -sunDirNormalized（与
        // lighting.slang 一致）。
        ubo.prtCounts = glm::ivec4((int)kPrtResolution,
                                    (m_prtEnabled && m_prtBaked) ? 1 : 0,
                                    m_prtShOrder, 0);
        ubo.prtGridMinCell = glm::vec4(m_prtGridMin, m_prtCellSize);
        {
            glm::vec3 dToSun = -glm::normalize(m_sunDir);
            float x = dToSun.x, y = dToSun.y, z = dToSun.z;
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
            // l=3 (B.10)
            float Y3n3 = 0.590043589f * y * (3.0f * x * x - y * y);
            float Y3n2 = 2.890611442f * x * y * z;
            float Y3n1 = 0.457045799f * y * (5.0f * z * z - 1.0f);
            float Y30  = 0.373176333f * z * (5.0f * z * z - 3.0f);
            float Y31  = 0.457045799f * x * (5.0f * z * z - 1.0f);
            float Y32  = 1.445305721f * z * (x * x - y * y);
            float Y33  = 0.590043589f * x * (x * x - 3.0f * y * y);
            glm::vec3 sunC{1.0f, 0.95f, 0.85f};
            float I = m_sunIntensity;
            // SH4
            ubo.prtLightSH_R = glm::vec4(I*sunC.r*Y0, I*sunC.r*Y1n1, I*sunC.r*Y10, I*sunC.r*Y11);
            ubo.prtLightSH_G = glm::vec4(I*sunC.g*Y0, I*sunC.g*Y1n1, I*sunC.g*Y10, I*sunC.g*Y11);
            ubo.prtLightSH_B = glm::vec4(I*sunC.b*Y0, I*sunC.b*Y1n1, I*sunC.b*Y10, I*sunC.b*Y11);
            // SH9
            ubo.prtLightSH9_R0 = glm::vec4(I*sunC.r*Y2n2, I*sunC.r*Y2n1, I*sunC.r*Y20, I*sunC.r*Y21);
            ubo.prtLightSH9_R1 = glm::vec4(I*sunC.r*Y22,  0, 0, 0);
            ubo.prtLightSH9_G0 = glm::vec4(I*sunC.g*Y2n2, I*sunC.g*Y2n1, I*sunC.g*Y20, I*sunC.g*Y21);
            ubo.prtLightSH9_G1 = glm::vec4(I*sunC.g*Y22,  0, 0, 0);
            ubo.prtLightSH9_B0 = glm::vec4(I*sunC.b*Y2n2, I*sunC.b*Y2n1, I*sunC.b*Y20, I*sunC.b*Y21);
            ubo.prtLightSH9_B1 = glm::vec4(I*sunC.b*Y22,  0, 0, 0);
            // SH16 (B.10) —— 投影完整，但 lighting 端 A_3=0 → diffuse 不贡献
            ubo.prtLightSH16_R0 = glm::vec4(I*sunC.r*Y3n3, I*sunC.r*Y3n2, I*sunC.r*Y3n1, I*sunC.r*Y30);
            ubo.prtLightSH16_R1 = glm::vec4(I*sunC.r*Y31,  I*sunC.r*Y32,  I*sunC.r*Y33,  0);
            ubo.prtLightSH16_G0 = glm::vec4(I*sunC.g*Y3n3, I*sunC.g*Y3n2, I*sunC.g*Y3n1, I*sunC.g*Y30);
            ubo.prtLightSH16_G1 = glm::vec4(I*sunC.g*Y31,  I*sunC.g*Y32,  I*sunC.g*Y33,  0);
            ubo.prtLightSH16_B0 = glm::vec4(I*sunC.b*Y3n3, I*sunC.b*Y3n2, I*sunC.b*Y3n1, I*sunC.b*Y30);
            ubo.prtLightSH16_B1 = glm::vec4(I*sunC.b*Y31,  I*sunC.b*Y32,  I*sunC.b*Y33,  0);
        }
        // M11 DDGI：probe 几何 + 启用闸门
        ubo.ddgiCounts = glm::ivec4((int)DdgiResources::kProbesX,
                                     (int)DdgiResources::kProbesY,
                                     (int)DdgiResources::kProbesZ,
                                     m_ddgiEnabled ? 1 : 0);
        ubo.ddgiOrigin = glm::vec4(m_ddgiOrigin, 0);
        ubo.ddgiSpacing = glm::vec4(m_ddgiSpacing, 0);
        ubo.ddgiOctaSizes = glm::ivec4((int)DdgiResources::kOctaIrr,
                                        (int)DdgiResources::kOctaDist, 0, 0);
        m_gbuffer.updateFrame(ubo);
        m_skybox.updateFrame(ubo.invViewProj, m_camera.position);
        // M5.0：每帧把当前 sun + AABB 喂给 RsmGeometryPass。sunDir 的约定
        // 跟主 FrameUBO 一致 —— 光传播方向（normalize 后），updateLight 内
        // 部自己取反算 toSun 用于摆 sun camera。
        m_rsmGeom.updateLight(m_scene.aabbMin, m_scene.aabbMax,
                              glm::normalize(m_sunDir),
                              glm::vec3(1.0f, 0.95f, 0.85f),
                              m_sunIntensity);

        VkCommandBuffer cmd = m_cmds[frame.frameInFlight];
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        // A.2：先读上次这个 in-flight 的结果（acquireNextFrame 已经
        // wait-fence，老 query 安全），再 reset 写新 query。
        uint32_t qBase = frame.frameInFlight * 2;
        if (m_timestampValid[frame.frameInFlight]) {
            uint64_t ts[2] = {0, 0};
            VkResult r = vkGetQueryPoolResults(m_device->device(), m_timestampPool,
                qBase, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            if (r == VK_SUCCESS && ts[1] > ts[0]) {
                float ms = float(ts[1] - ts[0]) * m_device->timestampPeriod() * 1e-6f;
                m_gpuMs = m_gpuMs * 0.9f + ms * 0.1f;
            }
        }
        vkCmdResetQueryPool(cmd, m_timestampPool, qBase, 2);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             m_timestampPool, qBase + 0);

        // === Phase 0: RSM 几何（sun-view MRT） ===
        // 独立于主 camera，不依赖 GBuffer 也无 GBuffer 依赖；放最前面方
        // 便 RenderDoc 单独审视。layout 转换由 m_rsmGeom.record 内部完成
        // （结尾 4 张都到 SHADER_READ_ONLY，给后续 RsmSamplePass 直接用）。
        m_rsmGeom.record(cmd, m_scene, m_sceneGpu);

        // === Phase 1: GBuffer prepass (graphics, MRT) ===
        // 3 GBuffer color images + depth go from UNDEFINED to attachment layout.
        auto toColorAttach = [&](VkImage img) {
            transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        };
        toColorAttach(m_rt.gAlbedoMetal.image());
        toColorAttach(m_rt.gNormalRough.image());
        toColorAttach(m_rt.gEmissiveAO.image());
        transitionImage(cmd, m_rt.depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        m_gbuffer.record(cmd, m_rt, m_scene, m_sceneGpu);

        // === Transition GBuffer / depth to SHADER_READ_ONLY for downstream
        // compute consumers (SSAO + Lighting). They share these inputs, so
        // do the layout flip once.
        auto colorAttachToSampled = [&](VkImage img) {
            transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        };
        colorAttachToSampled(m_rt.gAlbedoMetal.image());
        colorAttachToSampled(m_rt.gNormalRough.image());
        colorAttachToSampled(m_rt.gEmissiveAO.image());
        transitionImage(cmd, m_rt.depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        // === Phase 1.5: AO (compute, SSAO/GTAO 二选一或 None) ===
        // 两种 AO 共享 rt.ssao R8 输出（lighting 端不区分来源）。当 None
        // 时把 AO 清成 1.0，lighting 读到 1 表示无遮蔽。
        bool aoOn = (m_aoMethod != AOMethod::None);
        if (aoOn) {
            transitionImage(cmd, m_rt.ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            if (m_aoMethod == AOMethod::SSAO) {
                m_ssao.record(cmd, m_rt, ubo.proj, glm::inverse(ubo.proj), ubo.view);
            } else {
                m_gtao.record(cmd, m_rt, ubo.proj, ubo.view);
            }
            transitionImage(cmd, m_rt.ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        } else {
            transitionImage(cmd, m_rt.ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue white{};
            white.float32[0] = 1.0f; white.float32[1] = 1.0f;
            white.float32[2] = 1.0f; white.float32[3] = 1.0f;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_rt.ssao.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &range);
            transitionImage(cmd, m_rt.ssao.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }

        // === Phase 1.6: SSR (compute, before lighting) ===
        // Reads gNormalRough/gDepth (already SHADER_READ_ONLY) + hdrPrev
        // (kept SHADER_READ_ONLY by bootstrap + per-frame copy). Writes
        // ssr image (rgba16f). When disabled, clear ssr to (0,0,0,0) so
        // lighting blends iblSpec * (1-0) = iblSpec only.
        if (m_ssr.enabled) {
            transitionImage(cmd, m_rt.ssr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_ssr.record(cmd, m_rt);
            transitionImage(cmd, m_rt.ssr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        } else {
            transitionImage(cmd, m_rt.ssr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_rt.ssr.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, m_rt.ssr.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }

        // === Phase 1.7: SSGI / GTGI (compute, before lighting) ===
        // SSGI / GTGI 共享 rt.ssgi 输出 + ssgiPrev 时序 history，per-frame
        // 二选一 dispatch；lighting.slang 读 gSsgi 不区分来源。
        bool screenGiOn = m_ssgi.enabled || m_gtgi.enabled;
        if (screenGiOn) {
            // 1. ssgi(SHADER_READ_ONLY) → TRANSFER_SRC, ssgiPrev → TRANSFER_DST
            transitionImage(cmd, m_rt.ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            transitionImage(cmd, m_rt.ssgiPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

            // 2. copy
            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.extent = {m_rt.extent.width, m_rt.extent.height, 1};
            vkCmdCopyImage(cmd,
                m_rt.ssgi.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_rt.ssgiPrev.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &region);

            // 3. ssgiPrev → SHADER_READ_ONLY (供 SSGI shader sample)
            transitionImage(cmd, m_rt.ssgiPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            // 4. ssgi → GENERAL (写新值)
            transitionImage(cmd, m_rt.ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            if (m_ssgi.enabled) m_ssgi.record(cmd, m_rt);
            else                m_gtgi.record(cmd, m_rt);
            transitionImage(cmd, m_rt.ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        } else {
            transitionImage(cmd, m_rt.ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_rt.ssgi.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, m_rt.ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }

        // === Phase 1.83: VXGI voxelize → inject → mipmap (compute) ===
        // M7：voxelize 写 mip 0 → RSM inject 覆写 RGB → mipmap 下采样到全
        // mipchain → 转 SHADER_READ_ONLY 给 lighting cone trace。
        // M11 DDGI 也吃 voxel grid 当 ray tracing 替代 → DDGI on 时
        // 强制走 voxelize+inject+mipmap 这条链。
        // SDFGI 也需要 voxel grid（trace 直接采 voxel radiance + 用 SDF
        // 做 sphere-step）。ReSTIR DI shade 阶段也用 voxelGrid alpha 做 visibility。
        bool needVoxelGrid = m_vxgiEnabled || m_ddgiEnabled || m_sdfgiPass.enabled
                          || m_restirPass.enabled;
        if (needVoxelGrid) {
            // 1. clear 整张 mip chain 到 0。barrier 必须覆盖所有 mip（验证
            //    层会检查 vkCmdClearColorImage 的 range 内每一级 mip 都
            //    在 TRANSFER_DST_OPTIMAL）。helper transitionImage 默认只
            //    转 mip 0，本里程碑这里手写覆盖 mipLevels()。
            auto barrierAllMips = [&](VkImageLayout oldL, VkImageLayout newL,
                                       VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                                       VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
                b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
                b.oldLayout = oldL; b.newLayout = newL;
                b.image = m_vxgi.image().image();
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                      0, m_vxgi.mipLevels(), 0, 1};
                VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };
            barrierAllMips(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT,
                                       0, m_vxgi.mipLevels(), 0, 1};
            vkCmdClearColorImage(cmd, m_vxgi.image().image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &rg);
            barrierAllMips(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            // 2. voxelize：scatter 所有 primitive 到 mip 0。
            m_vxgiVoxelize.record(cmd, m_scene, m_sceneGpu,
                m_vxgiGridMin, m_vxgiCellSize, kVxgiResolution);

            // 3. inject：把 RSM (pos, flux) 覆写 mip 0 的 RGB。先做 barrier
            //    让 voxelize 的 write 对 inject 的 read-modify-write 可见
            //    （只对 mip 0；其它 mip 仍是 GENERAL 等 mipmap 写入）。
            VkImageMemoryBarrier2 vbar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            vbar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            vbar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            vbar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            vbar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            vbar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            vbar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            vbar.image = m_vxgi.image().image();
            vbar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo vdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            vdi.imageMemoryBarrierCount = 1; vdi.pImageMemoryBarriers = &vbar;
            vkCmdPipelineBarrier2(cmd, &vdi);

            m_vxgiInject.record(cmd, kVxgiResolution, m_vxgiGridMin, m_vxgiCellSize);

            // 4. mipmap：iterate src mip i → dst mip i+1。pass 内部做 barrier。
            m_vxgiMipmap.record(cmd, m_vxgi);

            // 5. 把所有 voxel mip 转 SHADER_READ_ONLY（最后一级 GENERAL → SR_O）。
            VkImageMemoryBarrier2 fb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            fb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            fb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            fb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            fb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            fb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            fb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            fb.image = m_vxgi.image().image();
            fb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                   m_vxgi.mipLevels() - 1, 1, 0, 1};
            VkDependencyInfo fdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            fdi.imageMemoryBarrierCount = 1; fdi.pImageMemoryBarriers = &fb;
            vkCmdPipelineBarrier2(cmd, &fdi);

            // 6. B.6：构建 anisotropic alpha mipchain。
            //    入口：所有 mip UNDEFINED → SHADER_READ_ONLY（discard）。
            //    aniso pass 内部每 iter 把 dst mip 转 GENERAL 写、再回
            //    SHADER_READ_ONLY；这样描述符 sampled view 总是 SR_O。
            VkImageMemoryBarrier2 ab{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            ab.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            ab.srcAccessMask = 0;
            ab.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            ab.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            ab.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ab.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ab.image = m_vxgi.aniso().image();
            ab.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                   0, m_vxgi.mipLevels(), 0, 1};
            VkDependencyInfo adi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            adi.imageMemoryBarrierCount = 1; adi.pImageMemoryBarriers = &ab;
            vkCmdPipelineBarrier2(cmd, &adi);

            m_vxgiAniso.record(cmd, m_vxgi);
            // pass 结尾所有 mip 已经在 SHADER_READ_ONLY（每 iter 自己转回）。

            // C.2 multi-bounce relight：每 voxel cone-trace 自身 voxel grid
            // gather 间接光 → scratch；然后 copy scratch → voxelGrid mip 0。
            // 注：mip 1+ 不重建（性能优化），cone trace 在远 mip 看 1-bounce
            // 旧值，近 mip 看 1.x-bounce 新值。视觉上多 bounce 效果可见。
            if (m_vxgiRelightEnabled) {
                // 1. scratch UNDEFINED → GENERAL
                VkImageMemoryBarrier2 sb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                sb.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                sb.srcAccessMask = 0;
                sb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                sb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                sb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                sb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                sb.image = m_vxgi.relightScratch().image();
                sb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo sdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                sdi.imageMemoryBarrierCount = 1; sdi.pImageMemoryBarriers = &sb;
                vkCmdPipelineBarrier2(cmd, &sdi);

                m_vxgiRelight.record(cmd, kVxgiResolution, m_vxgi.mipLevels(),
                    m_vxgiCellSize, m_vxgiGridMin, m_vxgiRelightStrength);

                // 2. scratch GENERAL → TRANSFER_SRC
                sb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                sb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                sb.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                sb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                sb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                sb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                vkCmdPipelineBarrier2(cmd, &sdi);

                // 3. voxelGrid mip 0 SHADER_READ_ONLY → TRANSFER_DST
                VkImageMemoryBarrier2 vb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                vb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                vb.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                vb.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                vb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                vb.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                vb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                vb.image = m_vxgi.image().image();
                vb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo vdiR{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                vdiR.imageMemoryBarrierCount = 1; vdiR.pImageMemoryBarriers = &vb;
                vkCmdPipelineBarrier2(cmd, &vdiR);

                // 4. copy scratch → voxelGrid mip 0
                VkImageCopy region{};
                region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.extent = {kVxgiResolution, kVxgiResolution, kVxgiResolution};
                vkCmdCopyImage(cmd,
                    m_vxgi.relightScratch().image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    m_vxgi.image().image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &region);

                // 5. voxelGrid mip 0 TRANSFER_DST → SHADER_READ_ONLY
                vb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                vb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                vb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                vb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                vb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                vb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                vkCmdPipelineBarrier2(cmd, &vdiR);
            }
        } else {
            // VXGI/DDGI 都关：lighting 仍要求 voxelGrid binding 合法。
            // UNDEFINED→SHADER_READ_ONLY（discard 内容；shader 不会读）。
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b.srcAccessMask = 0;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.image = m_vxgi.image().image();
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,
                                  0, m_vxgi.mipLevels(), 0, 1};
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &di);
            // B.6 aniso 也要在 SHADER_READ_ONLY（lighting binding 19）。
            // 走同样的 UNDEFINED→SHADER_READ_ONLY discard。
            b.image = m_vxgi.aniso().image();
            vkCmdPipelineBarrier2(cmd, &di);
        }

        // === Phase 1.835: SDFGI (compute, after VXGI 链) ====================
        // 进入条件：voxelGrid + aniso 均在 SHADER_READ_ONLY（上面 if/else 都
        // 已统一）。SDFGI 跑完写 rt.ssgi（lighting 读）。
        if (m_sdfgiPass.enabled) {
            // 首次激活：seedA/seedB/udf 是 UNDEFINED → 全转 GENERAL。
            if (!m_sdfgiBootstrapped) {
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
                bootstrapToGeneral(m_sdfgi.seedA().image());
                bootstrapToGeneral(m_sdfgi.seedB().image());
                bootstrapToGeneral(m_sdfgi.udf().image());
                m_sdfgiBootstrapped = true;
            }
            // ssgi 当前在 SHADER_READ_ONLY（SSGI/GTGI 关时由"else clear"路径
            // 转过；SDFGI 与 SSGI/GTGI 互斥）→ 转 GENERAL 供 trace 写。
            transitionImage(cmd, m_rt.ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            m_sdfgiPass.record(cmd, m_sdfgi, m_rt, m_frameIndex,
                m_sdfgiPass.seedThreshold, m_sdfgiPass.maxDistCells,
                (uint32_t)m_sdfgiPass.numRays,
                (uint32_t)m_sdfgiPass.maxSteps,
                m_sdfgiPass.rayMaxCells, m_sdfgiPass.hitEpsCells);

            // 写完 ssgi → 转 SHADER_READ_ONLY 供 lighting 读
            transitionImage(cmd, m_rt.ssgi.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }

        // === Phase 1.836: M9 RT GI (compute, HW 光线追踪) ==================
        if (m_rtGiBound && m_giIndexApplied == 10) {
            transitionImage(cmd, m_rt.rtGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_rtGiPass.record(cmd, m_rt);
            transitionImage(cmd, m_rt.rtGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        } else if (m_rtGiInited) {
            // RT GI 关闭时仍须保证 rtGI 在 SHADER_READ_ONLY（lighting 读）。
            // 始终从 UNDEFINED 起：旧内容不必保留——下一步是写全图（clear）
            // 或已经是 valid 数据但上一帧 lighting 读完我们重新 clear。UNDEFINED
            // 总是安全的。
            transitionImage(cmd, m_rt.rtGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_rt.rtGI.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, m_rt.rtGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }

        // === Phase 1.837: ReSTIR DI (compute, after VXGI 链) ================
        // 进入条件：voxelGrid 在 SHADER_READ_ONLY（shade 阶段需 alpha-only
        // visibility ray march）。lights SSBO 已 mapped；CPU 上传 demo 光源。
        if (m_restirPass.enabled) {
            // 每帧上传光源（demo lights，仅 8 个，开销可忽略）
            m_restir.updateLights(m_demoLights);

            // 首次激活：reservoirA/B UNDEFINED → GENERAL。
            if (!m_restirBootstrapped) {
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
                bootstrapToGeneral(m_restir.reservoirA().image());
                bootstrapToGeneral(m_restir.reservoirB().image());
                m_restirBootstrapped = true;
            }
            // rt.restir → GENERAL（shade 写）。oldLayout 看是否首次：
            // 首次 UNDEFINED；之后从上一帧 SR_O。
            VkImageLayout restirOld = m_restirOutInited
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            transitionImage(cmd, m_rt.restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                restirOld, VK_IMAGE_LAYOUT_GENERAL,
                m_restirOutInited ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                   : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                m_restirOutInited ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_restirOutInited = true;

            uint32_t numLights = (uint32_t)m_demoLights.size();
            bool useRtVis = m_rtSupported && m_rtGiBound;
            m_restirPass.record(cmd, m_restir, m_rt,
                numLights,
                (uint32_t)m_restirPass.numCandidates,
                (uint32_t)m_restirPass.numNeighbors,
                m_restirPass.spatialRadius,
                (uint32_t)m_restirPass.shadowSteps,
                m_restirPass.intensityScale,
                m_frameIndex,
                useRtVis);

            // rt.restir → SHADER_READ_ONLY 给 lighting 读
            transitionImage(cmd, m_rt.restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        } else {
            // 关闭：clear rt.restir 到 0（lighting 仍读，但加 0 = 无贡献）。
            VkImageLayout restirOld = m_restirOutInited
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            transitionImage(cmd, m_rt.restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                restirOld, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                m_restirOutInited ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                   : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                m_restirOutInited ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_rt.restir.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, m_rt.restir.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            m_restirOutInited = true;
        }

        // === Phase 1.84: DDGI update + blend (compute) ===
        // 从 voxel grid 投 ray → 写 ray buffer → blend 到 atlas。
        // atlas 转 GENERAL（供 blend 写）→ 转 SHADER_READ_ONLY（供 lighting 读）。
        // 关闭 DDGI 时也得保证 atlas 在 SHADER_READ_ONLY，避免 lighting binding 失效。
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
        if (m_ddgiEnabled) {
            // 1. atlas → GENERAL（首次 UNDEFINED；后续帧 SHADER_READ_ONLY）
            VkImageLayout oldAtlasL = m_ddgiAtlasInited
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            VkAccessFlags2 srcAcc = m_ddgiAtlasInited
                ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0;
            VkPipelineStageFlags2 srcStg = m_ddgiAtlasInited
                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barrierAtlas(m_ddgi.irradiance().image(),
                oldAtlasL, VK_IMAGE_LAYOUT_GENERAL,
                srcStg, srcAcc,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            barrierAtlas(m_ddgi.distance().image(),
                oldAtlasL, VK_IMAGE_LAYOUT_GENERAL,
                srcStg, srcAcc,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            // 2. ddgi update + blend（pass 内部已有 ray buf 写后 read 的 mem barrier）
            float jitterRot = float((m_frameIndex % 360) * 0.0174532925);
            m_ddgiPass.record(cmd, m_ddgi, m_ddgiOrigin, m_ddgiSpacing,
                m_vxgiGridMin, m_vxgiCellSize, kVxgiResolution,
                jitterRot, m_frameIndex);

            // 3. atlas → SHADER_READ_ONLY for lighting
            barrierAtlas(m_ddgi.irradiance().image(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            barrierAtlas(m_ddgi.distance().image(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            m_ddgiAtlasInited = true;
        } else if (!m_ddgiAtlasInited) {
            // DDGI 从未跑过 → atlas 还在 UNDEFINED。lighting binding 16/17
            // 要求 SHADER_READ_ONLY，做一次 discard transition。
            barrierAtlas(m_ddgi.irradiance().image(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            barrierAtlas(m_ddgi.distance().image(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            m_ddgiAtlasInited = true;
        }
        ++m_frameIndex;

        // === Phase 1.85: LPV inject + propagate (compute) ===
        // M6.0+M6.1：先把 RSM VPL 注入到 grid[0]（inject 内部已 clear），
        // 再 ping-pong 跑 N 次 propagate。每次迭代前后做 src/dst 的
        // GENERAL ↔ SHADER_READ_ONLY 切换。最后一次 propagate 写完，把
        // m_lpv.current() 转 SHADER_READ_ONLY 给 lighting 用。
        if (m_lpvEnabled) {
            // 1. inject 写 grid[curIdx=0] + GV，结尾 layout = GENERAL + pending write。
            m_lpvInject.record(cmd, kLpvResolution, m_lpvGridMin, m_lpvCellSize);

            // B.8：GV 从 GENERAL（inject 写完）转 SHADER_READ_ONLY 给
            // propagate sample。每 propagate iter 内 GV 都保持 SR_O。
            transitionImage(cmd, m_lpv.gv().image(), VK_IMAGE_ASPECT_COLOR_BIT,
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

            // 2. ping-pong propagate。inject 之后 grid[curIdx=0] 是 src。
            //    iter 强制偶数：每 2 次完整 ping-pong curIdx 回到 0。这样
            //    lighting 始终从 grid[0] 读最终结果，描述符可以一次绑定。
            int propIter = m_lpvProp.iterations & ~1;
            for (int it = 0; it < propIter; ++it) {
                LpvGrid& src = m_lpv.current();
                LpvGrid& dst = m_lpv.next();

                // src：GENERAL（写后） → SHADER_READ_ONLY（读）。
                barrierLpv(src,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

                // dst：UNDEFINED → GENERAL。第一帧 grid[1] 是 UNDEFINED；
                // 后续帧 dst 是上轮的 src（SHADER_READ_ONLY），都用
                // UNDEFINED 起始 discard —— 因为 propagate 全 cell 覆盖写。
                barrierLpv(dst,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

                m_lpvProp.record(cmd, m_lpv.curIdx(),
                                 kLpvResolution, m_lpvProp.occlusionAmplifier,
                                 m_lpvProp.gvOcclusionStrength);
                m_lpv.swap();
            }

            // 3. 最后一次 propagate 写完后，curIdx 已经回到 0；grid[0] 在
            //    GENERAL + pending write。给 lighting 用要再转 SHADER_READ_ONLY。
            //    特殊情况：iter=0（用户调成 0）时连一次 propagate 都没跑，
            //    grid[0] 还是 inject 写完的 GENERAL —— 同样要转 SHADER_READ_ONLY。
            barrierLpv(m_lpv.current(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        } else {
            // LPV 关闭：grid 永远不会被 inject/propagate 写。lighting 的
            // 描述符仍绑着 grid[0]，shader 端 lpvCounts.y=0 已经把读跳掉，
            // 但描述符必须是合法 layout。第一帧之前 grid 是 UNDEFINED，
            // 把它一次转成 SHADER_READ_ONLY（用 oneShot 不便，这里每帧
            // 做也 OK，因为 oldLayout=UNDEFINED 会 discard 内容；shader
            // 实际不读）。
            VkImage imgs[3] = {m_lpv.current().lpvR.image(),
                               m_lpv.current().lpvG.image(),
                               m_lpv.current().lpvB.image()};
            for (auto img : imgs) {
                transitionImage(cmd, img, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            }
        }

        // === Phase 1.8: RSM gather (compute) ===
        // RSM 几何已在 Phase 0 收尾时转 SHADER_READ_ONLY，可直接采样。
        // 关闭 RSM 时仍要把 rsmGI 抹成 0 + 转 SHADER_READ_ONLY —— 因为
        // LightingPass 的描述符 binding 9 = rsmGI，每帧必须是合法 layout
        // 且内容确定（counts.w=0 时 lerp 已被 gate 掉，但读纹理本身不能
        // 出错）。模式同 SSGI 的 disable-clear。
        if (m_rsmSample.enabled) {
            transitionImage(cmd, m_rt.rsmGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_rsmSample.record(cmd, m_rt);
            transitionImage(cmd, m_rt.rsmGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        } else {
            transitionImage(cmd, m_rt.rsmGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, m_rt.rsmGI.image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            transitionImage(cmd, m_rt.rsmGI.image(), VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }

        // === Phase 2: Lighting (compute) ===
        transitionImage(cmd, m_rt.hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        m_lighting.record(cmd, m_rt);

        // === Phase 3: Skybox (graphics) ===
        // hdrColor: GENERAL → COLOR_ATTACHMENT (test depth, fill where depth==1).
        // depth:    SHADER_READ_ONLY → DEPTH_ATTACHMENT (read-only test).
        transitionImage(cmd, m_rt.hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
        transitionImage(cmd, m_rt.depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
        m_skybox.record(cmd, m_rt);

        // === Phase 3.5: copy hdrColor → hdrPrev (for next frame's SSR) ===
        transitionImage(cmd, m_rt.hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        transitionImage(cmd, m_rt.hdrPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkImageCopy hdrCopy{};
        hdrCopy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        hdrCopy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        hdrCopy.extent = {m_rt.extent.width, m_rt.extent.height, 1};
        vkCmdCopyImage(cmd,
            m_rt.hdrColor.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            m_rt.hdrPrev.image(),  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &hdrCopy);
        transitionImage(cmd, m_rt.hdrPrev.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        // === Phase 4: Tonemap (compute) ===
        // hdrColor finishes phase 3.5 in TRANSFER_SRC, so transition from there.
        transitionImage(cmd, m_rt.hdrColor.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        transitionImage(cmd, m_rt.ldrTonemap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);

        m_tonemap.record(cmd, m_rt);

        // Phase 3: blit ldr → swapchain
        transitionImage(cmd, m_rt.ldrTonemap.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        transitionImage(cmd, frame.image, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {(int32_t)m_rt.extent.width, (int32_t)m_rt.extent.height, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[1] = {(int32_t)frame.extent.width, (int32_t)frame.extent.height, 1};
        vkCmdBlitImage(cmd,
            m_rt.ldrTonemap.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            frame.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        // Phase 4: ImGui on top of swapchain image
        transitionImage(cmd, frame.image, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        m_imgui.render(cmd, frame.view, frame.extent);

        transitionImage(cmd, frame.image, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);

        // A.2：写 end timestamp，标记本 in-flight 的 query 有效。
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                             m_timestampPool, qBase + 1);
        m_timestampValid[frame.frameInFlight] = true;

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

        // B.4 SSGI 时序：保存这帧 viewProj 给下一帧 reproject。
        m_prevViewProj = ubo.viewProj;
    }
}

}
