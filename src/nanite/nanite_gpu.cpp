#include "nanite_gpu.h"
#include "core/device.h"
#include "cluster_builder.h"

namespace somegi::nanite {

void createNaniteGpu(Device* dev, const NaniteMesh& mesh, NaniteGpu& out) {
    if (mesh.clusters.empty()) return;

    out.clusterCount = (uint32_t)mesh.clusters.size();
    out.vertexCount  = (uint32_t)mesh.vertices.size();
    out.indexCount   = (uint32_t)mesh.indices.size();

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    out.clusterBuffer = Buffer(*dev, VkDeviceSize(mesh.clusters.size() * sizeof(ClusterHeader)), usage, memProps);
    out.vertexBuffer  = Buffer(*dev, VkDeviceSize(mesh.vertices.size() * sizeof(PackedVertex)),  usage, memProps);
    out.indexBuffer   = Buffer(*dev, VkDeviceSize(mesh.indices.size() * sizeof(uint32_t)),        usage, memProps);

    VkBufferUsageFlags drawUsage = usage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    out.drawArgsBuffer = Buffer(*dev, VkDeviceSize(mesh.clusters.size()) * VkDeviceSize(sizeof(VkDrawIndexedIndirectCommand)),
                                drawUsage, memProps);
}

void NaniteGpu::destroy() {
    clusterBuffer.reset();
    vertexBuffer.reset();
    indexBuffer.reset();
    drawArgsBuffer.reset();
    clusterCount = 0;
    vertexCount = 0;
    indexCount = 0;
}

void destroyNaniteGpu(NaniteGpu& gpu) {
    gpu.destroy();
}

} // namespace somegi::nanite
