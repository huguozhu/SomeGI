#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// Nanite GPU-consumable data structures.
// Shared between the offline ClusterBuilder and the runtime renderer.

namespace somegi::nanite {

static constexpr uint32_t kMaxTrianglesPerCluster = 128;
static constexpr uint32_t kClusterVertexAlignment = 64; // bytes

// GPU-side cluster header (SSBO element).
// 32 bytes, 16-byte aligned for GPU efficiency.
struct alignas(16) ClusterHeader {
    glm::vec4 boundingSphere;       // xyz = center, w = radius
    uint32_t triangleOffset;        // offset into the global index buffer (indices)
    uint32_t triangleCount;         // number of triangles (3 * triangleCount = index count)
    uint32_t vertexOffset;          // offset into the global vertex buffer (vertices)
    uint32_t vertexCount;           // number of vertices
    uint32_t lodParentIndex;        // parent cluster in the LOD DAG (0xFFFFFFFF = root)
    uint32_t lodError;              // screen-space error metric (float as uint)
    uint32_t _pad;                  // alignment
};
static_assert(sizeof(ClusterHeader) == 48, "ClusterHeader must be 48 bytes");

// Per-vertex data stored in the packed vertex buffer.
// Position: 12 bytes (3 floats)
// Octahedron-encoded normal: 4 bytes (2 x snorm16)
// UV: 4 bytes (2 x float16)
// Total: 20 bytes per vertex
struct alignas(4) PackedVertex {
    float px, py, pz;               // position (world space, 12 bytes)
    uint32_t normal_uv;             // normal (2 x int16_t snorm) | uv (2 x half)
};
static_assert(sizeof(PackedVertex) == 16, "PackedVertex must be 16 bytes");

// .nanite file header
struct NaniteFileHeader {
    uint32_t magic;                 // 'NANT'
    uint32_t version;               // 1
    uint32_t clusterCount;
    uint32_t vertexCount;           // packed vertex count
    uint32_t indexCount;            // total index count (3 * total triangles)
    uint32_t rootClusterCount;      // number of root (LOD 0) clusters
    float aabbMin[3];
    float aabbMax[3];
    uint32_t _pad;
};
static constexpr uint32_t kNaniteMagic = 0x544E414Eu; // 'NANT' little-endian

} // namespace somegi::nanite
