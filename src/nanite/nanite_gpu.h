#pragma once
#include "core/device.h"
#include "nanite_types.h"
#include "core/buffer.h"
#include "cluster_builder.h"
#include <vector>

namespace somegi::nanite {

// GPU-side nanite mesh resources.
struct NaniteGpu {
    Buffer clusterBuffer;
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer drawArgsBuffer;
    uint32_t clusterCount = 0;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;

    bool valid() const { return clusterCount > 0; }

    // Defined in cpp
    void destroy();
};

// Defined in cpp
void createNaniteGpu(Device* dev, const NaniteMesh& mesh, NaniteGpu& out);
void destroyNaniteGpu(NaniteGpu& gpu);

} // namespace somegi::nanite
