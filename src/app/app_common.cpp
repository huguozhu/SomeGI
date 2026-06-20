// app_common.cpp — app 模块共享辅助函数的实现
#include "app_common.h"

namespace somegi {

// ════════════════════════════════════════════════════════════════
// 场景列表
// ════════════════════════════════════════════════════════════════
const SceneEntry kScenes[] = {
    { "cube",           "gltf/cube/cube.gltf" },
    { "Sponza",         "gltf/Sponza/Sponza.gltf" },
    { "DamagedHelmet",  "gltf/DamagedHelmet/DamagedHelmet.gltf" },
    { "Bistro",         "gltf/Bistro/Bistro.gltf" },
};
const int kSceneCount = (int)(sizeof(kScenes) / sizeof(kScenes[0]));

// ════════════════════════════════════════════════════════════════
// GI 方法列表
// ════════════════════════════════════════════════════════════════
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
const int kGiCount = (int)(sizeof(kGis) / sizeof(kGis[0]));

const char* giLabel(int i, char* buf, size_t bufSize) {
    if (i < 0 || i >= kGiCount) return "?";
    if (kGis[i].implemented && !kGis[i].requiresRt) return kGis[i].name;
    if (kGis[i].implemented && kGis[i].requiresRt) {
        std::snprintf(buf, bufSize, "%s (RT)", kGis[i].name);
        return buf;
    }
    std::snprintf(buf, bufSize, "%s (未实现)", kGis[i].name);
    return buf;
}

// ════════════════════════════════════════════════════════════════
// 场景状态持久化
// ════════════════════════════════════════════════════════════════
const char* kStatePath = "assets/scene_state.ini";

PersistedAll loadAllSceneStates() {
    PersistedAll out;
    std::ifstream f(kStatePath);
    if (!f) return out;
    std::string current;
    std::string line;
    while (std::getline(f, line)) {
        // 去除首尾空白
        size_t b = line.find_first_not_of(" \t\r");
        if (b == std::string::npos) continue;
        size_t e = line.find_last_not_of(" \t\r");
        line = line.substr(b, e - b + 1);

        // 节标题 [SceneName]
        if (line.size() > 2 && line[0] == '[' && line.back() == ']') {
            current = line.substr(1, line.size() - 2);
            continue;
        }
        if (current.empty()) continue;

        // key value 行
        std::istringstream is(line);
        std::string key;
        float v;
        if (!(is >> key >> v)) continue;

        SceneState& st = out.scenes[current];
        if (key == "cam.pos.x")     { st.camPos.x = v;   st.camValid = true; }
        else if (key == "cam.pos.y"){ st.camPos.y = v;   st.camValid = true; }
        else if (key == "cam.pos.z"){ st.camPos.z = v;   st.camValid = true; }
        else if (key == "cam.yaw")  { st.yaw = v;        st.camValid = true; }
        else if (key == "cam.pitch"){ st.pitch = v;      st.camValid = true; }
        else if (key == "cam.fov")  { st.fov = v;        st.camValid = true; }
        else if (key == "sun.dir.x")     st.sunDir.x = v;
        else if (key == "sun.dir.y")     st.sunDir.y = v;
        else if (key == "sun.dir.z")     st.sunDir.z = v;
        else if (key == "sun.intensity") st.sunIntensity = v;
        else if (key == "amb.x")         st.ambient.x = v;
        else if (key == "amb.y")         st.ambient.y = v;
        else if (key == "amb.z")         st.ambient.z = v;
        else if (key == "taa.blend")     st.taaBlendAlpha = v;
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

// ════════════════════════════════════════════════════════════════
// AppSettings 持久化
// ════════════════════════════════════════════════════════════════
const char* kAppSettingsPath = "assets/app_settings.ini";

AppSettings loadAppSettings() {
    AppSettings cfg;
    std::ifstream f(kAppSettingsPath);
    if (!f) return cfg;
    std::string line;
    while (std::getline(f, line)) {
        size_t b = line.find_first_not_of(" \t\r");
        if (b == std::string::npos || line[b] == '#') continue;
        size_t e = line.find_last_not_of(" \t\r");
        std::string s = line.substr(b, e - b + 1);
        std::istringstream is(s);
        std::string key;
        if (!(is >> key)) continue;
        if (key == "gi.index")           is >> cfg.giIndex;
        else if (key == "shadow.index")  is >> cfg.shadowIndex;
        else if (key == "ao.method")     is >> cfg.aoMethod;
        else if (key == "aa.method")     is >> cfg.aaMethod;
        else if (key == "rendering")     is >> cfg.renderingMode;
        else if (key == "msaa.samples")  is >> cfg.msaaSamples;
        else if (key == "fg.enabled")    is >> cfg.useFrameGraph;
        else if (key == "cull.gpu")      is >> cfg.useGpuCulling;
        else if (key == "cull.hiz")      is >> cfg.useHiZOcclusion;
        else if (key == "mipmaps")       is >> cfg.useMipmaps;
        else if (key == "mesh.shader")   is >> cfg.useMeshShader;
        else if (key == "taa.blend")     is >> cfg.taaBlendAlpha;
        else if (key == "shadow.rt.rays")   is >> cfg.shadowRtRays;
        else if (key == "shadow.rt.radius") is >> cfg.shadowRtRadius;
    }
    return cfg;
}

void saveAppSettings(const AppSettings& cfg) {
    std::ofstream f(kAppSettingsPath);
    if (!f) return;
    f << "# AppSettings — 全局渲染设置（自动保存）\n";
    f << "gi.index "       << cfg.giIndex       << "\n";
    f << "shadow.index "   << cfg.shadowIndex   << "\n";
    f << "ao.method "      << cfg.aoMethod      << "\n";
    f << "aa.method "      << cfg.aaMethod      << "\n";
    f << "rendering "      << cfg.renderingMode << "\n";
    f << "msaa.samples "   << cfg.msaaSamples   << "\n";
    f << "fg.enabled "     << (cfg.useFrameGraph ? 1 : 0) << "\n";
    f << "cull.gpu "       << (cfg.useGpuCulling ? 1 : 0) << "\n";
    f << "cull.hiz "       << (cfg.useHiZOcclusion ? 1 : 0) << "\n";
    f << "mipmaps "        << (cfg.useMipmaps ? 1 : 0) << "\n";
    f << "mesh.shader "    << (cfg.useMeshShader ? 1 : 0) << "\n";
    f << "taa.blend "      << cfg.taaBlendAlpha << "\n";
    f << "shadow.rt.rays " << cfg.shadowRtRays << "\n";
    f << "shadow.rt.radius " << cfg.shadowRtRadius << "\n";
}

} // namespace somegi
