#include "app.h"
#include "app_common.h"
#include "core/device.h"
#include "core/swapchain.h"
#include "scene/upload.h"
#include "rhi/vulkan/vk_context.h"
#include "scene/scene_gpu.h"
#include <imgui.h>

namespace somegi {

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
                                   m_currentSceneIndex < kSceneCount)
                                  ? kScenes[m_currentSceneIndex].name : "?";
            if (ImGui::BeginCombo("glTF", curName)) {
                for (int i = 0; i < kSceneCount; ++i) {
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
                uploadScene(*m_device, static_cast<rhi::VkContext&>(*m_context).vkCommandPool(), m_scene, m_sceneGpu, m_useMipmaps, m_renderer.rhiDevice());
                m_renderer.gbuffer().bindScene(*m_device, m_sceneGpu, (uint32_t)m_sceneGpu.images.size());
                m_renderer.rsmGeom().bindScene(m_sceneGpu, (uint32_t)m_sceneGpu.images.size());
                m_renderer.vxgiVoxelize().bindScene(m_sceneGpu, (uint32_t)m_sceneGpu.images.size(), m_renderer.vxgi());
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
                                m_renderer.taa().bindResources(m_renderer.rt(), 0);
                                m_renderer.smaa().bindResources(m_renderer.rt());
                            } else {
                                m_renderer.rt().destroyAaResources();
                                m_renderer.tonemap().bindOutput(m_renderer.rt().ldrTonemap.view(), 0);
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
        {
            // 图形 API 后端选择（需重启生效）
            const char* backends[] = {"Vulkan", "D3D12"};
            const char* backendLabels[] = {"Vulkan", "D3D12 (experimental)"};
            int backendIdx = (m_backendName == "d3d12") ? 1 : 0;
            ImGui::Text("Graphics API");
            ImGui::SameLine();
            if (ImGui::BeginCombo("##backend", backendLabels[backendIdx])) {
                for (int i = 0; i < 2; ++i) {
                    bool sel = (backendIdx == i);
                    if (ImGui::Selectable(backendLabels[i], sel)) {
                        m_backendName = backends[i];
                        // 后端切换需要重启应用生效
                    }
                    if (sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            if (backendIdx == 1) {
                ImGui::SameLine();
                ImGui::TextDisabled("(restart required)");
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

                // ---- 资源布局时间线 ----
                if (!fgDebug.timelines.empty() && ImGui::CollapsingHeader("Layout Timeline")) {
                    auto& tls = fgDebug.timelines;
                    auto& names = fgDebug.timelinePassNames;

                    // 布局→颜色和缩写
                    auto layoutInfo = [](VkImageLayout l) -> std::pair<ImU32, const char*> {
                        switch (l) {
                            case VK_IMAGE_LAYOUT_UNDEFINED:              return {0xFF333333, "---"};
                            case VK_IMAGE_LAYOUT_GENERAL:                return {0xFF4488CC, "GEN"};
                            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: return {0xFFCC6644, "COL"};
                            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                            case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL: return {0xFF44AA66, "DEP"};
                            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return {0xFFAA44CC, "SRO"};
                            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:   return {0xFFCCAA44, "TSR"};
                            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:   return {0xFF44AACC, "TDT"};
                            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:        return {0xFFCCCC44, "PRE"};
                            default: return {0xFF555555, "???"};
                        }
                    };

                    if (ImGui::BeginTable("##fgtimeline", (int)names.size() + 1,
                            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                            ImGuiTableFlags_SizingFixedFit)) {
                        ImGui::TableSetupColumn("Resource");
                        for (auto& n : names)
                            ImGui::TableSetupColumn(n.c_str());
                        ImGui::TableHeadersRow();

                        for (auto& tl : tls) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Text("%s", tl.resourceName.c_str());

                            for (size_t pi = 0; pi < names.size(); ++pi) {
                                ImGui::TableNextColumn();
                                if (pi < tl.snapshots.size()) {
                                    auto& s = tl.snapshots[pi];
                                    auto [color, label] = layoutInfo(s.layout);
                                    if (s.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
                                        ImGui::TextDisabled("---");
                                    } else {
                                        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(color), "%s%s",
                                            label, s.barrierEmitted ? " B" : "");
                                    }
                                    if (ImGui::IsItemHovered()) {
                                        ImGui::BeginTooltip();
                                        ImGui::Text("Pass: %s", pi < names.size() ? names[pi].c_str() : "?");
                                        ImGui::Text("Layout: 0x%X", (uint32_t)s.layout);
                                        ImGui::Text("Access: 0x%X", (uint32_t)s.access);
                                        ImGui::Text("Barrier: %s", s.barrierEmitted ? "Y" : "N");
                                        ImGui::EndTooltip();
                                    }
                                } else {
                                    ImGui::TextDisabled("-");
                                }
                            }
                        }
                        ImGui::EndTable();
                    }
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

    // 全局渲染设置有变更即保存
    {
        AppSettings cfg;
        cfg.giIndex        = m_currentGiIndex;
        cfg.shadowIndex    = m_currentShadowIndex;
        cfg.aoMethod       = (int)m_aoMethod;
        cfg.aaMethod       = (int)m_aaMethod;
        cfg.renderingMode  = (int)m_renderingMode;
        cfg.msaaSamples    = (int)m_msaaSamples;
        cfg.useFrameGraph  = m_useFrameGraph;
        cfg.useGpuCulling  = m_useGpuCulling;
        cfg.useHiZOcclusion = m_useHiZOcclusion;
        cfg.useMipmaps     = m_useMipmaps;
        cfg.useMeshShader  = m_renderer.useMeshShader();
        cfg.taaBlendAlpha  = m_taaBlendAlpha;
        cfg.shadowRtRays   = m_renderer.shadow().rtRayCount();
        cfg.shadowRtRadius = m_renderer.shadow().rtSunRadius();
        saveAppSettings(cfg);
    }
}

} // namespace somegi
