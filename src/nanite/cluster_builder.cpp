#include "cluster_builder.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstring>

namespace somegi::nanite {

// ---- Bounding sphere (Ritter's algorithm) ---------------------------------
glm::vec4 ClusterBuilder::computeBoundingSphere(const glm::vec3* positions, uint32_t count) {
    if (count == 0) return glm::vec4(0, 0, 0, 0);
    if (count == 1) return glm::vec4(positions[0], 0.0f);

    // Initial guess: bounding box center
    glm::vec3 bmin = positions[0], bmax = positions[0];
    for (uint32_t i = 1; i < count; ++i) {
        bmin = glm::min(bmin, positions[i]);
        bmax = glm::max(bmax, positions[i]);
    }
    glm::vec3 center = (bmin + bmax) * 0.5f;
    float radius = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        float d = glm::distance(center, positions[i]);
        if (d > radius) radius = d;
    }

    // Refine: expand sphere to include farthest point
    for (uint32_t iter = 0; iter < 3; ++iter) {
        uint32_t farthest = 0;
        float maxD2 = 0;
        for (uint32_t i = 0; i < count; ++i) {
            float d2 = glm::dot(positions[i] - center, positions[i] - center);
            if (d2 > maxD2) { maxD2 = d2; farthest = i; }
        }
        float d = std::sqrt(maxD2);
        if (d <= radius + 1e-6f) break;
        // Move center toward farthest point, grow radius
        center = center + (positions[farthest] - center) * ((d - radius) / (2.0f * d));
        radius = (radius + d) * 0.5f;
    }

    return glm::vec4(center, radius);
}

// ---- Vertex packing -------------------------------------------------------
PackedVertex ClusterBuilder::packVertex(const Vertex& v, const glm::mat4& transform) {
    PackedVertex pv;
    glm::vec3 wp = glm::vec3(transform * glm::vec4(v.position, 1.0f));
    pv.px = wp.x; pv.py = wp.y; pv.pz = wp.z;

    // Octahedron-encode normal into 2 x snorm16
    glm::vec3 wn = glm::normalize(glm::vec3(transform * glm::vec4(v.normal, 0.0f)));
    float absSum = std::abs(wn.x) + std::abs(wn.y) + std::abs(wn.z);
    glm::vec2 oct(0.0f);
    if (absSum > 1e-10f) {
        oct = glm::vec2(wn.x, wn.y) / absSum;
        if (wn.z < 0) {
            float tx = (1.0f - std::abs(oct.y)) * (oct.x >= 0 ? 1.0f : -1.0f);
            float ty = (1.0f - std::abs(oct.x)) * (oct.y >= 0 ? 1.0f : -1.0f);
            oct = glm::vec2(tx, ty);
        }
    }
    int16_t nx = (int16_t)std::clamp((int32_t)(oct.x * 32767.0f), -32767, 32767);
    int16_t ny = (int16_t)std::clamp((int32_t)(oct.y * 32767.0f), -32767, 32767);
    pv.normal_uv = (uint32_t)(uint16_t)nx | ((uint32_t)(uint16_t)ny << 16);

    return pv;
}

// ---- Build from Mesh ------------------------------------------------------
NaniteMesh ClusterBuilder::buildFromMesh(const Mesh& mesh, const std::vector<Vertex>& vertices,
                                         uint32_t maxTrisPerCluster) {
    NaniteMesh result;
    glm::vec3 bmin(1e30f), bmax(-1e30f);

    std::vector<glm::vec3> allPositions;
    std::vector<glm::vec3> clusterTris; // temp per-cluster

    // Collect all triangles from all primitives
    struct TriInfo { uint32_t i0, i1, i2; int32_t matIdx; };
    std::vector<TriInfo> allTris;

    for (auto& p : mesh.primitives) {
        uint32_t triCount = p.indexCount / 3;
        for (uint32_t t = 0; t < triCount; ++t) {
            TriInfo ti;
            ti.i0 = t * 3 + 0;
            ti.i1 = t * 3 + 1;
            ti.i2 = t * 3 + 2;
            ti.matIdx = p.materialIndex;
            allTris.push_back(ti);
        }
    }

    // Partition into clusters
    uint32_t numClusters = (uint32_t(allTris.size()) + maxTrisPerCluster - 1) / maxTrisPerCluster;
    result.clusters.reserve(numClusters);
    result.indices.reserve(allTris.size() * 3);

    uint32_t globalVertIdx = 0;

    for (uint32_t ci = 0; ci < numClusters; ++ci) {
        uint32_t startTri = ci * maxTrisPerCluster;
        uint32_t endTri = std::min(startTri + maxTrisPerCluster, (uint32_t)allTris.size());
        uint32_t triCount = endTri - startTri;

        // Collect vertex positions for bounding sphere
        std::vector<glm::vec3> positions;
        for (uint32_t t = startTri; t < endTri; ++t) {
            auto& tri = allTris[t];
            positions.push_back(vertices[tri.i0].position);
            positions.push_back(vertices[tri.i1].position);
            positions.push_back(vertices[tri.i2].position);
        }

        ClusterHeader hdr{};
        hdr.boundingSphere = computeBoundingSphere(positions.data(), (uint32_t)positions.size());
        hdr.triangleOffset = startTri * 3;
        hdr.triangleCount = triCount;
        hdr.vertexOffset = globalVertIdx;
        hdr.vertexCount = triCount * 3;
        hdr.lodParentIndex = 0xFFFFFFFF;
        hdr.lodError = 0;

        result.clusters.push_back(hdr);
        globalVertIdx += triCount * 3;

        // Copy indices (just sequential: 0,1,2, 3,4,5, ...)
        for (uint32_t t = startTri; t < endTri; ++t) {
            auto& tri = allTris[t];
            result.indices.push_back(tri.i0);
            result.indices.push_back(tri.i1);
            result.indices.push_back(tri.i2);
        }

        // Update AABB
        for (auto& p : positions) {
            bmin = glm::min(bmin, p);
            bmax = glm::max(bmax, p);
        }
    }

    // Pack vertices (identity transform for now — world-space baking)
    glm::mat4 ident(1.0f);
    result.vertices.reserve(globalVertIdx);
    for (auto& tri : allTris) {
        result.vertices.push_back(packVertex(vertices[tri.i0], ident));
        result.vertices.push_back(packVertex(vertices[tri.i1], ident));
        result.vertices.push_back(packVertex(vertices[tri.i2], ident));
    }

    result.aabbMin = bmin;
    result.aabbMax = bmax;
    return result;
}

// ---- Build from primitives (with shared index buffer) ---------------------
NaniteMesh ClusterBuilder::buildFromPrimitives(const std::vector<Primitive>& prims,
                                                const std::vector<Vertex>& vertices,
                                                const std::vector<uint32_t>& indices,
                                                uint32_t maxTrisPerCluster) {
    NaniteMesh result;
    glm::vec3 bmin(1e30f), bmax(-1e30f);
    uint32_t globalVertIdx = 0;

    for (auto& prim : prims) {
        uint32_t triCount = prim.indexCount / 3;
        uint32_t numClusters = (triCount + maxTrisPerCluster - 1) / maxTrisPerCluster;

        for (uint32_t ci = 0; ci < numClusters; ++ci) {
            uint32_t startTri = ci * maxTrisPerCluster;
            uint32_t endTri = std::min(startTri + maxTrisPerCluster, triCount);
            uint32_t clusterTriCount = endTri - startTri;

            std::vector<glm::vec3> positions;
            for (uint32_t t = startTri; t < endTri; ++t) {
                uint32_t base = prim.firstIndex + t * 3;
                for (int v = 0; v < 3; ++v) {
                    uint32_t vi = indices[base + v];
                    positions.push_back(vertices[vi + prim.vertexOffset].position);
                }
            }

            ClusterHeader hdr{};
            hdr.boundingSphere = computeBoundingSphere(positions.data(), (uint32_t)positions.size());
            hdr.triangleOffset = (uint32_t)result.indices.size();
            hdr.triangleCount = clusterTriCount;
            hdr.vertexOffset = globalVertIdx;
            hdr.vertexCount = clusterTriCount * 3;
            hdr.lodParentIndex = 0xFFFFFFFF;
            hdr.lodError = 0;
            result.clusters.push_back(hdr);

            // Copy indices and pack vertices
            glm::mat4 ident(1.0f);
            for (uint32_t t = startTri; t < endTri; ++t) {
                uint32_t base = prim.firstIndex + t * 3;
                for (int v = 0; v < 3; ++v) {
                    uint32_t vi = indices[base + v];
                    result.indices.push_back(globalVertIdx + (t - startTri) * 3 + v);
                    result.vertices.push_back(packVertex(vertices[vi + prim.vertexOffset], ident));
                }
            }
            globalVertIdx += clusterTriCount * 3;

            for (auto& p : positions) {
                bmin = glm::min(bmin, p);
                bmax = glm::max(bmax, p);
            }
        }
    }

    result.aabbMin = bmin;
    result.aabbMax = bmax;
    return result;
}

// ---- Binary serialization -------------------------------------------------
bool ClusterBuilder::writeToFile(const NaniteMesh& mesh, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    NaniteFileHeader hdr{};
    hdr.magic   = kNaniteMagic;
    hdr.version  = 1;
    hdr.clusterCount = (uint32_t)mesh.clusters.size();
    hdr.vertexCount  = (uint32_t)mesh.vertices.size();
    hdr.indexCount   = (uint32_t)mesh.indices.size();
    hdr.rootClusterCount = hdr.clusterCount; // all clusters at root level for now
    hdr.aabbMin[0] = mesh.aabbMin.x; hdr.aabbMin[1] = mesh.aabbMin.y; hdr.aabbMin[2] = mesh.aabbMin.z;
    hdr.aabbMax[0] = mesh.aabbMax.x; hdr.aabbMax[1] = mesh.aabbMax.y; hdr.aabbMax[2] = mesh.aabbMax.z;
    hdr._pad = 0;

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(mesh.clusters.data()), mesh.clusters.size() * sizeof(ClusterHeader));
    f.write(reinterpret_cast<const char*>(mesh.vertices.data()), mesh.vertices.size() * sizeof(PackedVertex));
    f.write(reinterpret_cast<const char*>(mesh.indices.data()), mesh.indices.size() * sizeof(uint32_t));

    return f.good();
}

bool ClusterBuilder::readFromFile(const std::string& path, NaniteMesh& outMesh) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    NaniteFileHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic != kNaniteMagic || hdr.version != 1) return false;

    outMesh.clusters.resize(hdr.clusterCount);
    outMesh.vertices.resize(hdr.vertexCount);
    outMesh.indices.resize(hdr.indexCount);

    f.read(reinterpret_cast<char*>(outMesh.clusters.data()), hdr.clusterCount * sizeof(ClusterHeader));
    f.read(reinterpret_cast<char*>(outMesh.vertices.data()), hdr.vertexCount * sizeof(PackedVertex));
    f.read(reinterpret_cast<char*>(outMesh.indices.data()), hdr.indexCount * sizeof(uint32_t));

    outMesh.aabbMin = glm::vec3(hdr.aabbMin[0], hdr.aabbMin[1], hdr.aabbMin[2]);
    outMesh.aabbMax = glm::vec3(hdr.aabbMax[0], hdr.aabbMax[1], hdr.aabbMax[2]);

    return f.good();
}

} // namespace somegi::nanite
