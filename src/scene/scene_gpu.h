#pragma once
#include "scene.h"
#include <glm/glm.hpp>

namespace somegi {
class Device;

// 前向声明 RHI 类型（somegi::rhi 命名空间）
namespace rhi { class RHIDevice; }

struct MaterialGpu {
    glm::vec4 baseColorFactor;
    glm::vec3 emissiveFactor;
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;
    float alphaCutoff;
    int baseColorTex, mrTex, normalTex, occlusionTex, emissiveTex;
    uint32_t alphaMode;
    uint32_t doubleSided;
    uint32_t _pad0;
};

void uploadScene(Device& d, VkCommandPool pool, const SceneCpu& cpu, SceneGpu& out, bool useMipmaps = true,
                 rhi::RHIDevice* rhiDevice = nullptr);
void destroySceneSamplers(Device& d, SceneGpu& gpu);

}
