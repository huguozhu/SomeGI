#pragma once
#include "scene.h"
#include <glm/glm.hpp>

namespace somegi {
class Device;

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

void uploadScene(Device& d, VkCommandPool pool, const SceneCpu& cpu, SceneGpu& out);
void destroySceneSamplers(Device& d, SceneGpu& gpu);

}
