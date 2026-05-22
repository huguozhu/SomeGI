#pragma once
#include "nanite_types.h"
#include "scene/scene.h"       // for Mesh, Primitive, Vertex
#include <vector>
#include <string>

namespace somegi::nanite {

// Built cluster data ready for serialization or GPU upload.
struct NaniteMesh {
    std::vector<ClusterHeader> clusters;
    std::vector<PackedVertex>  vertices;
    std::vector<uint32_t>      indices;     // 3 * triangleCount per cluster
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
};

class ClusterBuilder {
public:
    // Build nanite mesh data from a single scene mesh.
    // The mesh must have unique (non-index-shared) vertices for correct SV_VertexID/3 mapping.
    static NaniteMesh buildFromMesh(const Mesh& mesh, const std::vector<Vertex>& vertices,
                                    uint32_t maxTrianglesPerCluster = 128);

    // Build from multiple primitives (all merged into one nanite mesh).
    static NaniteMesh buildFromPrimitives(const std::vector<Primitive>& prims,
                                          const std::vector<Vertex>& vertices,
                                          const std::vector<uint32_t>& indices,
                                          uint32_t maxTrianglesPerCluster = 128);

    // Serialize to .nanite binary format.
    static bool writeToFile(const NaniteMesh& mesh, const std::string& path);

    // Read from .nanite binary format.
    static bool readFromFile(const std::string& path, NaniteMesh& outMesh);

private:
    // Compute tight bounding sphere using Ritter's algorithm.
    static glm::vec4 computeBoundingSphere(const glm::vec3* positions, uint32_t count);

    // Pack vertex into GPU format.
    static PackedVertex packVertex(const Vertex& v, const glm::mat4& transform);
};

} // namespace somegi::nanite
