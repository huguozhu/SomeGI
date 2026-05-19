#pragma once
#include "core/buffer.h"
#include "core/image.h"
#include <glm/glm.hpp>
#include <vector>

namespace somegi {

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;
    glm::vec2 uv0;
};

struct MaterialDesc {
    glm::vec4 baseColorFactor{1.0f};
    glm::vec3 emissiveFactor{0.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float normalScale = 1.0f;
    float occlusionStrength = 1.0f;
    float alphaCutoff = 0.5f;
    int baseColorTex = -1;
    int mrTex = -1;
    int normalTex = -1;
    int occlusionTex = -1;
    int emissiveTex = -1;
    uint32_t alphaMode = 0;
    uint32_t doubleSided = 0;
};

struct Primitive {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t  vertexOffset = 0;
    int32_t  materialIndex = -1;
};

struct Mesh {
    std::vector<Primitive> primitives;
    glm::vec3 localAabbMin{0};
    glm::vec3 localAabbMax{0};
};

struct Node {
    glm::mat4 worldTransform{1.0f};
    int meshIndex = -1;
};

struct TextureCpu {
    int width = 0, height = 0, channels = 4;
    std::vector<uint8_t> rgba;
    bool isSrgb = false;
};

struct SceneCpu {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Mesh> meshes;
    std::vector<Node> nodes;
    std::vector<MaterialDesc> materials;
    std::vector<TextureCpu> textures;
    glm::vec3 aabbMin{0}, aabbMax{0};
};

struct SceneGpu {
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer materialBuffer;
    std::vector<Image> images;
    Image whiteTex;
    Image normalTex;
    VkSampler linearSampler = VK_NULL_HANDLE;
};

}
