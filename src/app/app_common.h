// app_common.h — app 模块内共享的类型、常量和辅助函数
// 供 app.cpp / app_ui.cpp / app_fg.cpp 等文件共用
#pragma once
#include "app.h"
#include <cstdio>
#include <string>
#include <map>
#include <fstream>
#include <sstream>

namespace somegi {

// ════════════════════════════════════════════════════════════════
// 场景列表
// ════════════════════════════════════════════════════════════════
struct SceneEntry {
    const char* name;
    const char* relPath;
};
extern const SceneEntry kScenes[];
extern const int kSceneCount;

// ════════════════════════════════════════════════════════════════
// GI 方法列表
// ════════════════════════════════════════════════════════════════
struct GiEntry {
    const char* name;
    bool implemented;
    bool requiresRt = false;
};
extern GiEntry kGis[];
extern const int kGiCount;

// 获取 GI 条目的显示标签（未实现项标注"（未实现）"，需 RT 项标注"（RT）"）
const char* giLabel(int i, char* buf, size_t bufSize);

// ════════════════════════════════════════════════════════════════
// 场景状态持久化
// ════════════════════════════════════════════════════════════════
extern const char* kStatePath;

struct PersistedAll {
    std::map<std::string, SceneState> scenes;
    std::string lastScene;   // 上次关闭时激活的场景
};

// 从 scene_state.ini 读取所有场景的持久化状态
PersistedAll loadAllSceneStates();

// 将所有场景状态写回 scene_state.ini
void saveAllSceneStates(const std::map<std::string, SceneState>& states,
                        const std::string& lastScene);

// ════════════════════════════════════════════════════════════════
// AppSettings — 全局渲染设置持久化（GI/AA/MSAA/阴影 等）
// ════════════════════════════════════════════════════════════════
extern const char* kAppSettingsPath;

struct AppSettings {
    int giIndex        = 1;   // GI 方法（0=None, 1=IBL, ...）
    int shadowIndex    = 1;   // 阴影方法
    int aoMethod       = 1;   // AO 方法（0=None, 1=SSAO, 2=GTAO）
    int aaMethod       = 1;   // AA 方法（0=None, 1=MSAA, 2=TAA, 3=SMAA）
    int renderingMode  = 0;   // 0=Deferred, 1=Forward
    int msaaSamples    = 4;   // MSAA 采样数（1/2/4/8/16）
    bool useFrameGraph = true;
    bool useGpuCulling = false;
    bool useHiZOcclusion = false;
    bool useMipmaps    = true;
    bool useMeshShader = false;
    float taaBlendAlpha = 0.92f;
    int shadowRtRays   = 8;       // RT Soft shadow 采样数
    float shadowRtRadius = 0.05f; // RT Soft shadow 太阳半径
};

// 从 app_settings.ini 加载全局渲染设置
AppSettings loadAppSettings();

// 将全局渲染设置保存到 app_settings.ini
void saveAppSettings(const AppSettings& cfg);

} // namespace somegi
