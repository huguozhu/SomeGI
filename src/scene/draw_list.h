#pragma once
#include "scene/scene.h"
#include <glm/glm.hpp>
#include <cstdint>
#include <vector>
namespace somegi {
struct DrawEntry {
    glm::mat4 worldTransform; int32_t materialIndex; uint32_t firstIndex; uint32_t indexCount;
    int32_t vertexOffset; glm::vec3 aabbMin; uint32_t _pad0; glm::vec3 aabbMax; uint32_t _pad1;
};
static_assert(sizeof(DrawEntry) == 112);
void buildDrawList(const SceneCpu& cpu, std::vector<DrawEntry>& out);
}
