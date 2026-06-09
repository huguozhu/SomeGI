#pragma once
#include "renderer/core/render_targets.h"
#include "renderer/core/gbuffer_pass.h"
#include "renderer/core/forward_pass.h"
#include "renderer/gi/rsm/rsm_geometry_pass.h"
#include "renderer/gi/rsm/rsm_sample_pass.h"
#include "renderer/gi/lpv/lpv_grid.h"         // LpvResources + LpvGrid
#include "renderer/gi/lpv/lpv_inject_pass.h"
#include "renderer/gi/lpv/lpv_propagate_pass.h"
#include "renderer/gi/vxgi/vxgi_resources.h"
#include "renderer/gi/vxgi/vxgi_voxelize_pass.h"
#include "renderer/gi/vxgi/vxgi_inject_pass.h"
#include "renderer/gi/vxgi/vxgi_mipmap_pass.h"
#include "renderer/gi/vxgi/vxgi_aniso_pass.h"
#include "renderer/gi/vxgi/vxgi_relight_pass.h"
#include "renderer/gi/vxgi/vxgi_resolve_6axis_pass.h"
#include "renderer/gi/sdfgi/sdfgi_resources.h"
#include "renderer/gi/sdfgi/sdfgi_pass.h"
#include "renderer/gi/restir/restir_resources.h"
#include "renderer/gi/restir/restir_pass.h"
#include "renderer/gi/prt/prt_resources.h"
#include "renderer/gi/prt/prt_bake_pass.h"
#include "renderer/gi/ddgi/ddgi_resources.h"
#include "renderer/gi/ddgi/ddgi_pass.h"
#include "renderer/gi/ndgi/ndgi_resources.h"
#include "renderer/gi/ndgi/ndgi_pass.h"
#include "renderer/core/lighting_pass.h"
#include "renderer/ao/ssao_pass.h"
#include "renderer/ao/gtao_pass.h"
#include "renderer/screenspace/ssr_pass.h"
#include "renderer/screenspace/ssgi_pass.h"
#include "renderer/screenspace/gtgi_pass.h"
#include "renderer/core/skybox_pass.h"
#include "renderer/core/tonemap_pass.h"
#include "renderer/core/taa_pass.h"
#include "renderer/core/smaa_pass.h"
#include "renderer/core/imgui_pass.h"
#include "renderer/gi/rt/scene_rt_as.h"
#include "renderer/gi/rt/rt_gi_pass.h"
#include "renderer/gi/lumen/lumen_resources.h"
#include "renderer/gi/lumen/lumen_probe_pass.h"
#include "renderer/gi/lumen/lumen_gather_pass.h"
#include "renderer/gi/lumen/lumen_filter_pass.h"
#include "renderer/core/barrier_manager.h"
#include "renderer/core/render_pipeline.h"
#include "gi/ibl_baker.h"
#include "gi/gi_technique.h"
#include "scene/scene.h"
#include <glm/glm.hpp>
#include <memory>

namespace somegi {

class Device;

class FrameRenderer {
public:
    struct BenchResult { int gi, aa, ao; float fps, gpuMs; };

    void init(Device& d, VkCommandPool pool, VkExtent2D extent,
              VkSampleCountFlagBits msaaSamples, bool rtSupported,
              VkFormat swapchainFmt, GLFWwindow* window);
    void destroy();

    void onResize(Device& d, VkExtent2D newExtent,
                  VkSampleCountFlagBits msaaSamples,
                  VkFormat swapchainFmt, GLFWwindow* window);

    void bindScenePasses(Device& d, const SceneGpu& gpu, uint32_t textureCount);

    void setGiTechnique(IGITechnique* tech);
    void applyGiFlags(int effectiveGiIndex);

    // Bootstrap
    void bootstrapHdrPrev();
    void bootstrapSsgiTemporal();

    // Accessors — App needs these for buildUI() and run()
    RenderTargets&       rt()          { return m_rt; }
    GBufferPass&         gbuffer()     { return m_gbuffer; }
    ForwardPass&         forward()     { return m_forward; }
    RsmGeometryPass&     rsmGeom()     { return m_rsmGeom; }
    RsmSamplePass&       rsmSample()   { return m_rsmSample; }
    LpvResources&        lpv()         { return m_lpv; }
    LpvInjectPass&       lpvInject()   { return m_lpvInject; }
    LpvPropagatePass&    lpvProp()     { return m_lpvProp; }
    VxgiResources&       vxgi()        { return m_vxgi; }
    VxgiVoxelizePass&    vxgiVoxelize(){ return m_vxgiVoxelize; }
    VxgiInjectPass&      vxgiInject()  { return m_vxgiInject; }
    VxgiMipmapPass&      vxgiMipmap()  { return m_vxgiMipmap; }
    VxgiAnisoPass&       vxgiAniso()   { return m_vxgiAniso; }
    VxgiRelightPass&     vxgiRelight() { return m_vxgiRelight; }
    VxgiResolve6AxisPass& vxgi6Axis()  { return m_vxgiResolve6Axis; }
    SdfgiResources&      sdfgi()       { return m_sdfgi; }
    SdfgiPass&           sdfgiPass()   { return m_sdfgiPass; }
    RestirResources&     restir()      { return m_restir; }
    RestirPass&          restirPass()  { return m_restirPass; }
    PrtResources&        prt()         { return m_prt; }
    PrtBakePass&         prtBake()     { return m_prtBake; }
    DdgiResources&       ddgi()        { return m_ddgi; }
    DdgiPass&            ddgiPass()    { return m_ddgiPass; }
    NdgiResources&       ndgi()        { return m_ndgi; }
    NdgiPass&            ndgiPass()    { return m_ndgiPass; }
    LightingPass&        lighting()    { return m_lighting; }
    SsaoPass&            ssao()        { return m_ssao; }
    GtaoPass&            gtao()        { return m_gtao; }
    SsrPass&             ssr()         { return m_ssr; }
    SsgiPass&            ssgi()        { return m_ssgi; }
    GtgiPass&            gtgi()        { return m_gtgi; }
    SkyboxPass&          skybox()      { return m_skybox; }
    TonemapPass&         tonemap()     { return m_tonemap; }
    TaaPass&             taa()         { return m_taa; }
    SmaaPass&            smaa()        { return m_smaa; }
    ImGuiPass&           imgui()       { return m_imgui; }
    SceneRtAS&           rtAS()        { return m_rtAS; }
    RtGiPass&            rtGi()        { return m_rtGiPass; }
    LumenResources&      lumen()       { return m_lumen; }
    LumenProbePass&      lumenProbe()  { return m_lumenProbePass; }
    LumenFilterPass&     lumenFilter() { return m_lumenFilterPass; }
    LumenGatherPass&     lumenGather() { return m_lumenGatherPass; }
    BarrierManager&      barriers()    { return m_barriers; }
    RenderPipeline&      pipeline()    { return m_pipeline; }

    IblResources&        envIbl()      { return m_envIbl; }
    std::unique_ptr<IGITechnique>& giTech() { return m_giTech; }

    // GI flags
    bool& lpvEnabled()          { return m_lpvEnabled; }
    bool& vxgiEnabled()         { return m_vxgiEnabled; }
    bool& prtEnabled()          { return m_prtEnabled; }
    bool& prtBaked()            { return m_prtBaked; }
    int&  prtShOrder()          { return m_prtShOrder; }
    bool& ddgiEnabled()         { return m_ddgiEnabled; }
    bool& ndgiEnabled()         { return m_ndgiEnabled; }
    bool& ndgiInited()          { return m_ndgiInited; }
    bool& lumenEnabled()        { return m_lumenEnabled; }
    bool& vxgiRelightEnabled()  { return m_vxgiRelightEnabled; }
    bool& vxgiSixAxisInited()   { return m_vxgiSixAxisInited; }
    float& vxgiRelightStrength(){ return m_vxgiRelightStrength; }
    int&  lumenDebugMode()      { return m_lumenDebugMode; }
    int&  demoLightCount()      { return m_demoLightCount; }
    float& demoLightIntensity() { return m_demoLightIntensity; }
    std::vector<PointLightCpu>& demoLights() { return m_demoLights; }

    // Bootstrap flags
    bool& ddgiAtlasInited()     { return m_ddgiAtlasInited; }
    bool& restirBootstrapped()  { return m_restirBootstrapped; }
    bool& restirOutInited()     { return m_restirOutInited; }
    bool& sdfgiBootstrapped()   { return m_sdfgiBootstrapped; }
    bool& lumenAtlasInited()    { return m_lumenAtlasInited; }
    bool& lumenProbeInited()    { return m_lumenProbeInited; }
    bool& lumenFilterInited()   { return m_lumenFilterInited; }
    bool& lumenGatherInited()   { return m_lumenGatherInited; }
    bool& lumenOutInited()      { return m_lumenOutInited; }
    bool& aaHistoryNeedsInit()  { return m_aaHistoryNeedsInit; }
    bool& rtGiInited()          { return m_rtGiInited; }
    bool& rtGiBound()           { return m_rtGiBound; }

    // Grid params
    glm::vec3& lpvGridMin()     { return m_lpvGridMin; }
    float&     lpvCellSize()    { return m_lpvCellSize; }
    glm::vec3& vxgiGridMin()    { return m_vxgiGridMin; }
    float&     vxgiCellSize()   { return m_vxgiCellSize; }
    glm::vec3& prtGridMin()     { return m_prtGridMin; }
    float&     prtCellSize()    { return m_prtCellSize; }
    glm::vec3& ddgiOrigin()     { return m_ddgiOrigin; }
    glm::vec3& ddgiSpacing()    { return m_ddgiSpacing; }

    // Profiling
    float& gpuMs() { return m_gpuMs; }
    float  gpuMs() const { return m_gpuMs; }
    float* passTimes(uint32_t fi) { return m_passMs[fi]; }
    const char* const* passNames() const { return m_passNames; }

    // GPU timestamp
    VkQueryPool timestampPool() const { return m_timestampPool; }
    bool& timestampValid(uint32_t i)  { return m_timestampValid[i]; }
    void writeTimestamp(VkCommandBuffer cmd, uint32_t slot);

    // Frame state
    uint32_t& frameIndex() { return m_frameIndex; }

    // Benchmark
    bool& benchRunning()    { return m_benchRunning; }
    bool& benchCollecting() { return m_benchCollecting; }
    int&  benchGi()         { return m_benchGi; }
    int&  benchAa()         { return m_benchAa; }
    int&  benchAo()         { return m_benchAo; }
    float& benchTimer()     { return m_benchTimer; }
    int&   benchFrameCount(){ return m_benchFrameCount; }
    float& benchFpsSum()    { return m_benchFpsSum; }
    float& benchGpuSum()    { return m_benchGpuSum; }
    auto&  benchResults()   { return m_benchResults; }

    // RT
    bool rtSupported() const { return m_rtSupported; }

    // Constants
    static constexpr uint32_t kLpvResolution  = 32;
    static constexpr uint32_t kVxgiResolution = 128;
    static constexpr uint32_t kPrtResolution  = 32;
    static constexpr uint32_t kSdfgiResolution = 128;
    static constexpr uint32_t kTimestampSlots = 9;
    static constexpr uint32_t kRestirMaxLights = 64;

    enum TimestampSlot : uint32_t {
        kTsStart = 0, kTsGBuffer, kTsAO, kTsVoxelGI, kTsLighting,
        kTsSkybox, kTsTonemap, kTsAA, kTsEnd, kTsCount = kTsEnd
    };

    void registerPipelineSteps();
    void buildPipelineTable();
    void rebuildDemoLights(const SceneCpu& cpu);

private:
    Device* m_device = nullptr;
    VkCommandPool m_pool = VK_NULL_HANDLE;

    RenderTargets m_rt;
    BarrierManager m_barriers;
    RenderPipeline m_pipeline;

    GBufferPass     m_gbuffer;
    ForwardPass     m_forward;
    RsmGeometryPass m_rsmGeom;
    RsmSamplePass   m_rsmSample;
    LpvResources    m_lpv;
    LpvInjectPass   m_lpvInject;
    LpvPropagatePass m_lpvProp;
    VxgiResources        m_vxgi;
    VxgiVoxelizePass     m_vxgiVoxelize;
    VxgiInjectPass       m_vxgiInject;
    VxgiMipmapPass       m_vxgiMipmap;
    VxgiAnisoPass        m_vxgiAniso;
    VxgiRelightPass      m_vxgiRelight;
    VxgiResolve6AxisPass m_vxgiResolve6Axis;
    SdfgiResources  m_sdfgi;
    SdfgiPass       m_sdfgiPass;
    RestirResources m_restir;
    RestirPass      m_restirPass;
    PrtResources    m_prt;
    PrtBakePass     m_prtBake;
    DdgiResources   m_ddgi;
    DdgiPass        m_ddgiPass;
    NdgiResources   m_ndgi;
    NdgiPass        m_ndgiPass;
    LightingPass    m_lighting;
    SsaoPass        m_ssao;
    GtaoPass        m_gtao;
    SsrPass         m_ssr;
    SsgiPass        m_ssgi;
    GtgiPass        m_gtgi;
    SkyboxPass      m_skybox;
    SceneRtAS       m_rtAS;
    RtGiPass        m_rtGiPass;
    LumenResources  m_lumen;
    LumenProbePass  m_lumenProbePass;
    LumenFilterPass m_lumenFilterPass;
    LumenGatherPass m_lumenGatherPass;
    TonemapPass     m_tonemap;
    TaaPass         m_taa;
    SmaaPass        m_smaa;
    ImGuiPass       m_imgui;

    IblResources m_envIbl;
    std::unique_ptr<IGITechnique> m_giTech;

    // GI flags
    bool m_lpvEnabled = false;
    bool m_vxgiEnabled = false;
    bool m_prtEnabled = false;
    bool m_prtBaked = false;
    int  m_prtShOrder = 0;
    bool m_ddgiEnabled = false;
    bool m_ndgiEnabled = false;
    bool m_ndgiInited = false;
    bool m_lumenEnabled = false;
    bool m_vxgiRelightEnabled = false;
    float m_vxgiRelightStrength = 1.0f;
    bool m_vxgiSixAxisInited = false;

    // Bootstrap flags
    bool m_ddgiAtlasInited = false;
    bool m_restirBootstrapped = false;
    bool m_restirOutInited = false;
    bool m_sdfgiBootstrapped = false;
    bool m_lumenAtlasInited = false;
    bool m_lumenProbeInited = false;
    bool m_lumenFilterInited = false;
    bool m_lumenGatherInited = false;
    bool m_lumenOutInited = false;
    bool m_aaHistoryNeedsInit = false;
    int  m_lumenDebugMode = 0;
    bool m_rtGiInited = false;
    bool m_rtGiBound = false;
    bool m_rtSupported = false;

    // Grid geometry
    glm::vec3 m_lpvGridMin{0};
    float     m_lpvCellSize = 1.0f;
    glm::vec3 m_vxgiGridMin{0};
    float     m_vxgiCellSize = 1.0f;
    glm::vec3 m_prtGridMin{0};
    float     m_prtCellSize = 1.0f;
    glm::vec3 m_ddgiOrigin{0};
    glm::vec3 m_ddgiSpacing{1};

    // ReSTIR demo lights
    std::vector<PointLightCpu> m_demoLights;
    int   m_demoLightCount = 8;
    float m_demoLightIntensity = 8.0f;

    // Benchmark
    bool m_benchRunning = false;
    int  m_benchGi = 0, m_benchAa = 0, m_benchAo = 0;
    float m_benchTimer = 0;
    int   m_benchFrameCount = 0;
    float m_benchFpsSum = 0, m_benchGpuSum = 0;
    std::vector<BenchResult> m_benchResults;
    bool m_benchCollecting = false;

    // GPU timestamp
    VkQueryPool m_timestampPool = VK_NULL_HANDLE;
    bool m_timestampValid[kFramesInFlight]{};
    float m_gpuMs = 0.0f;
    float m_passMs[kFramesInFlight][kTimestampSlots]{};
    const char* m_passNames[kTimestampSlots]{};

    // Frame state
    uint32_t m_frameIndex = 0;
};

} // namespace somegi
