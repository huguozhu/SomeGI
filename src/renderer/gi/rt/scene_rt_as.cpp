#include "renderer/gi/rt/scene_rt_as.h"
#include "core/device.h"
#include "scene/scene_gpu.h"
#include "scene/upload.h"
#include <vector>
#include <cstring>

namespace somegi {

namespace {
constexpr VkDeviceSize alignUp(VkDeviceSize val, VkDeviceSize align) {
    return (val + align - 1) & ~(align - 1);
}

// Helper: get VkDeviceAddress from a VkBuffer handle.
VkDeviceAddress getBufferAddress(VkDevice dev, VkBuffer buf) {
    VkBufferDeviceAddressInfo bdi{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bdi.buffer = buf;
    return vkGetBufferDeviceAddress(dev, &bdi);
}
}

void SceneRtAS::swap(SceneRtAS& o) noexcept {
    std::swap(m_device, o.m_device);
    std::swap(m_blases, o.m_blases);
    std::swap(m_tlas, o.m_tlas);
    m_blasBuf.swap(o.m_blasBuf);
    m_tlasBuf.swap(o.m_tlasBuf);
    m_scratchBuf.swap(o.m_scratchBuf);
    m_instanceBuf.swap(o.m_instanceBuf);
    m_instanceDataBuf.swap(o.m_instanceDataBuf);
    std::swap(m_instanceCount, o.m_instanceCount);
}

void SceneRtAS::destroy() {
    if (!m_device) return;
    auto& dispatch = m_device->dispatch();
    for (auto as : m_blases) {
        if (as) dispatch.destroyAccelerationStructureKHR(as, nullptr);
    }
    if (m_tlas) dispatch.destroyAccelerationStructureKHR(m_tlas, nullptr);
    m_blases.clear();
    m_tlas = VK_NULL_HANDLE;
    m_blasBuf.reset();
    m_tlasBuf.reset();
    m_scratchBuf.reset();
    m_instanceBuf.reset();
    m_instanceDataBuf.reset();
    m_instanceCount = 0;
    m_device = nullptr;
}

VkWriteDescriptorSetAccelerationStructureKHR SceneRtAS::tlasWriteInfo() const {
    VkWriteDescriptorSetAccelerationStructureKHR ai{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    ai.accelerationStructureCount = 1;
    ai.pAccelerationStructures = &m_tlas;
    return ai;
}

void SceneRtAS::build(Device& d, VkCommandPool pool,
                      const SceneCpu& scene, const SceneGpu& sceneGpu) {
    destroy();
    m_device = &d;
    auto dev = d.device();
    auto& dispatch = d.dispatch();

    VkDeviceAddress vertAddr = getBufferAddress(dev, sceneGpu.vertexBuffer.handle());
    VkDeviceAddress idxAddr  = getBufferAddress(dev, sceneGpu.indexBuffer.handle());

    // ---- Step 1: enumerate primitives ----
    struct PrimInfo {
        uint32_t firstIndex;
        uint32_t indexCount;
        int32_t vertexOffset;
        int32_t materialIndex;
    };
    std::vector<PrimInfo> prims;
    for (auto& mesh : scene.meshes) {
        for (auto& prim : mesh.primitives) {
            prims.push_back({prim.firstIndex, prim.indexCount, prim.vertexOffset, prim.materialIndex});
        }
    }
    uint32_t primCount = (uint32_t)prims.size();
    if (primCount == 0) return;

    // ---- Step 2: create geometry descriptions and query sizes ----
    struct BlasEntry {
        VkAccelerationStructureGeometryKHR geometry;
        VkAccelerationStructureBuildRangeInfoKHR range;
        VkAccelerationStructureBuildSizesInfoKHR sizes;
        VkDeviceSize accelOffset;   // offset into m_blasBuf
    };
    std::vector<BlasEntry> entries(primCount);

    VkDeviceSize totalAccelSize = 0;
    VkDeviceSize maxScratchSize = 0;

    for (uint32_t i = 0; i < primCount; ++i) {
        auto& e = entries[i];
        auto& prim = prims[i];

        VkAccelerationStructureGeometryTrianglesDataKHR tri{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        tri.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
        tri.vertexData.deviceAddress = vertAddr;
        tri.vertexStride             = sizeof(Vertex);
        tri.maxVertex                = (uint32_t)scene.vertices.size() - 1;
        tri.indexType                = VK_INDEX_TYPE_UINT32;
        tri.indexData.deviceAddress  = idxAddr;
        tri.transformData.deviceAddress = 0;

        e.geometry = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        e.geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        e.geometry.geometry.triangles = tri;
        e.geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        e.range = {};
        e.range.primitiveCount  = prim.indexCount / 3;

        e.range.primitiveOffset = prim.firstIndex * sizeof(uint32_t);
        e.range.firstVertex     = (uint32_t)std::max(0, prim.vertexOffset);
        e.range.transformOffset = 0;

        VkAccelerationStructureBuildGeometryInfoKHR bi{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        bi.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bi.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        bi.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = 1;
        bi.pGeometries   = &entries[i].geometry;

        e.sizes = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        uint32_t pc = prim.indexCount / 3;
        dispatch.getAccelerationStructureBuildSizesKHR(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                       &bi, &pc, &e.sizes);

        e.accelOffset = totalAccelSize;
        totalAccelSize += alignUp(e.sizes.accelerationStructureSize, 256);

        maxScratchSize = std::max(maxScratchSize, e.sizes.buildScratchSize);
    }

    // ---- Step 3: allocate backing buffers ----
    auto bufACCEL = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    auto bufSRC   = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    m_blasBuf = Buffer(d, totalAccelSize, bufACCEL, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_scratchBuf = Buffer(d, maxScratchSize, bufSRC, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceAddress scratchAddr = m_scratchBuf.deviceAddress();

    // ---- Step 4: create BLAS objects + query device addresses ----
    m_blases.resize(primCount, VK_NULL_HANDLE);
    std::vector<VkDeviceAddress> blasAddresses(primCount);
    for (uint32_t i = 0; i < primCount; ++i) {
        VkAccelerationStructureCreateInfoKHR aci{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        aci.buffer = m_blasBuf.handle();
        aci.offset = entries[i].accelOffset;
        aci.size   = entries[i].sizes.accelerationStructureSize;
        aci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        VK_CHECK(dispatch.createAccelerationStructureKHR(&aci, nullptr, &m_blases[i]));
        VkAccelerationStructureDeviceAddressInfoKHR adi{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        adi.accelerationStructure = m_blases[i];
        blasAddresses[i] = dispatch.getAccelerationStructureDeviceAddressKHR(&adi);
    }

    // ---- Step 5: build BLAS one-by-one to avoid TDR ----
    for (uint32_t i = 0; i < primCount; ++i) {
        oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
            VkAccelerationStructureBuildGeometryInfoKHR bi{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
            bi.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            bi.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
            bi.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            bi.dstAccelerationStructure  = m_blases[i];
            bi.geometryCount = 1;
            bi.pGeometries   = &entries[i].geometry;
            bi.scratchData.deviceAddress = scratchAddr;

            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &entries[i].range;
            dispatch.cmdBuildAccelerationStructuresKHR(cmd, 1, &bi, &pRange);
        });
    }

    // ---- Step 6: build TLAS ----
    // Each (node, primitive) pair → one instance.
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    std::vector<RtInstanceData> instanceData;

    for (auto& node : scene.nodes) {
        if (node.meshIndex < 0 || node.meshIndex >= (int)scene.meshes.size()) continue;
        auto& mesh = scene.meshes[node.meshIndex];
        for (uint32_t pIdx = 0; pIdx < (uint32_t)mesh.primitives.size(); ++pIdx) {
            uint32_t globalPrimIdx = 0;
            for (int m = 0; m < node.meshIndex; ++m) {
                globalPrimIdx += (uint32_t)scene.meshes[m].primitives.size();
            }
            globalPrimIdx += pIdx;

            auto& prim = mesh.primitives[pIdx];

            // Transpose glm::mat4 (column-major 4×4) → VkTransformMatrixKHR (row-major 3×4).
            VkAccelerationStructureInstanceKHR inst{};
            auto& tfm = inst.transform;
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 4; ++col)
                    tfm.matrix[row][col] = node.worldTransform[col][row];

            inst.instanceCustomIndex = (uint32_t)instanceData.size();
            inst.mask = 0xFF;
            inst.instanceShaderBindingTableRecordOffset = 0;
            inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            inst.accelerationStructureReference = blasAddresses[globalPrimIdx];

            instances.push_back(inst);

            RtInstanceData id{};
            id.materialIndex = prim.materialIndex;
            id.vertexOffset  = prim.vertexOffset;
            id.firstIndex    = prim.firstIndex;
            instanceData.push_back(id);
        }
    }

    m_instanceCount = (uint32_t)instances.size();
    if (m_instanceCount == 0) return;

    // Upload instance buffer + instance data buffer.
    VkDeviceSize instBufSize = m_instanceCount * sizeof(VkAccelerationStructureInstanceKHR);
    VkDeviceSize dataBufSize = m_instanceCount * sizeof(RtInstanceData);

    // For TLAS build, instance buffer needs ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY.
    // For shader read, instanceDataBuf needs STORAGE.
    m_instanceBuf = Buffer(d, instBufSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_instanceDataBuf = Buffer(d, dataBufSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Upload via staging.
    Buffer instStaging(d, instBufSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(instStaging.mapped(), instances.data(), (size_t)instBufSize);

    Buffer dataStaging(d, dataBufSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(dataStaging.mapped(), instanceData.data(), (size_t)dataBufSize);

    // ---- Step 7: query TLAS build size ----
    VkAccelerationStructureGeometryKHR tlasGeom{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeom.geometry.instances = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
    tlasGeom.geometry.instances.data.deviceAddress = m_instanceBuf.deviceAddress();
    tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    tlasBuildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasBuildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlasBuildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tlasBuildInfo.geometryCount = 1;
    tlasBuildInfo.pGeometries   = &tlasGeom;

    VkAccelerationStructureBuildSizesInfoKHR tlasSizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    uint32_t tlasPrimCount = m_instanceCount;
    dispatch.getAccelerationStructureBuildSizesKHR(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                    &tlasBuildInfo, &tlasPrimCount, &tlasSizes);

    // Update scratch buffer to handle TLAS scratch if larger.
    if (tlasSizes.buildScratchSize > maxScratchSize) {
        m_scratchBuf.reset();
        m_scratchBuf = Buffer(d, tlasSizes.buildScratchSize, bufSRC,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        scratchAddr = m_scratchBuf.deviceAddress();
    }

    // Allocate TLAS backing buffer + create TLAS.
    m_tlasBuf = Buffer(d, tlasSizes.accelerationStructureSize, bufACCEL,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkAccelerationStructureCreateInfoKHR tlasACI{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    tlasACI.buffer = m_tlasBuf.handle();
    tlasACI.offset = 0;
    tlasACI.size   = tlasSizes.accelerationStructureSize;
    tlasACI.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    VK_CHECK(dispatch.createAccelerationStructureKHR(&tlasACI, nullptr, &m_tlas));

    // ---- Step 8: build TLAS (one-shot) ----
    oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
        // Copy instances and data.
        VkBufferCopy ic{0, 0, instBufSize};
        vkCmdCopyBuffer(cmd, instStaging.handle(), m_instanceBuf.handle(), 1, &ic);
        VkBufferCopy dc{0, 0, dataBufSize};
        vkCmdCopyBuffer(cmd, dataStaging.handle(), m_instanceDataBuf.handle(), 1, &dc);

        // Barrier: ensure copies finish before AS build reads.
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);

        // Build TLAS.
        VkAccelerationStructureBuildGeometryInfoKHR bi = tlasBuildInfo;
        bi.dstAccelerationStructure = m_tlas;
        bi.scratchData.deviceAddress = scratchAddr;

        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = m_instanceCount;
        range.primitiveOffset = 0;
        range.firstVertex = 0;
        range.transformOffset = 0;
        const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

        dispatch.cmdBuildAccelerationStructuresKHR(cmd, 1, &bi, &pRange);
    });

    std::printf("[SceneRtAS] built %u BLAS + TLAS (%u instances), accel=%.1f MB scratch=%.1f MB\n",
                primCount, m_instanceCount,
                (totalAccelSize + tlasSizes.accelerationStructureSize) / (1024.0 * 1024.0),
                maxScratchSize / (1024.0 * 1024.0));
}

}
