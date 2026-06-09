#pragma once
#include "core/vk_common.h"
#include <glm/glm.hpp>
#include <string>

// GI 抽象层 —— 所有间接光技术（IBL/SSGI/VXGI/PRT/RT/ReSTIR）实现 IGITechnique
// 接口，由 App 持有一个 std::unique_ptr 在运行时切换。下游的 LightingPass /
// ForwardPass 通过 set=1 描述符消费当前 technique 的资源。

namespace somegi {

class Device;
struct IblResources;

// onAttach 时由 App 提供的一次性上下文。
struct GIContext {
    Device* device = nullptr;
    VkCommandPool oneShotPool = VK_NULL_HANDLE;
    // skybox.hdr 预烘焙的 env 资源（envCube / diffuseCube / specularCube /
    // brdfLut / linearSampler）。归 App 所有，跨 GI 切换不重烘，IBLTechnique
    // 只是借用。M5+ 的 VXGI 等也可以把它当 fallback。
    const IblResources* iblBaked = nullptr;
    // M3 之后会加 SceneGpu* / VkImageView gbufferNormals 等。
};

// 每帧上下文（保留接口位，M5+ 实现需要逐帧 prepare 时启用）。
struct FrameContext {
    uint32_t frameIndex = 0;
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::vec3 cameraPos{0};
};

// 所有 GI 技术的虚基类。生命周期：App 构造时 onAttach，析构时 onDetach。
// LightingPass 每帧通过 descriptorSet() / descriptorSetLayout()
// 取当前 technique 的资源。
class IGITechnique {
public:
    virtual ~IGITechnique() = default;

    virtual const char* name() const = 0;

    // 一次性资源初始化（M3 IBL：占用 GIContext.iblBaked；M5+ VXGI：体素化、
    // 注入光照、生成 mip 都在这里）。
    virtual void onAttach(const GIContext& ctx) = 0;
    virtual void onDetach() = 0;

    // 逐帧 prepare（M3 IBL 是 no-op；M5+ VXGI 每帧重新 inject 光照）。
    virtual void prepare(VkCommandBuffer /*cmd*/, const FrameContext& /*fc*/) {}

    // 下游 pass 绑定 set=1 时使用。
    virtual VkDescriptorSetLayout descriptorSetLayout() const = 0;
    virtual VkDescriptorSet descriptorSet() const = 0;

    // ImGui 自定义子面板（由 App 在 GI 父面板中调用）。
    virtual void drawUI() {}
};

}
