#pragma once
#include "core/buffer.h"
#include "scene/scene.h"
#include "rhi/base/acceleration_structure.h"
#include <memory>
#include <vulkan/vulkan.h>

namespace somegi {

class Device;
namespace rhi { class RHIDevice; }
struct SceneGpu;

// Per-instance data used by the RT shader to fetch material/textures on hit.
struct RtInstanceData {
    int32_t materialIndex = -1;
    int32_t vertexOffset  = 0;
    uint32_t firstIndex   = 0;
    int32_t _pad0         = 0;
};

// SceneRtAS — builds BLAS per-primitive + single TLAS for HW ray tracing.
// Created once per scene load, destroyed on scene switch or shutdown.
// Only valid when Device::features().accelStruct && .rayQuery are true.
class SceneRtAS {
public:
    ~SceneRtAS() { destroy(); }

    SceneRtAS() = default;
    SceneRtAS(const SceneRtAS&) = delete;
    SceneRtAS& operator=(const SceneRtAS&) = delete;
    SceneRtAS(SceneRtAS&& o) noexcept { swap(o); }
    SceneRtAS& operator=(SceneRtAS&& o) noexcept { if (this != &o) { reset(); swap(o); } return *this; }

    void swap(SceneRtAS& o) noexcept;

    // Build all acceleration structures from scene data.
    // Must be called with a one-shot-compatible pool (oneShotSubmit).
    void build(Device& d, rhi::RHIDevice& rhiDevice, VkCommandPool pool, const SceneCpu& scene, const SceneGpu& sceneGpu);

    void destroy();
    void reset() { destroy(); }

    // Accessors for descriptor binding.
    const rhi::RHIAccelerationStructure* tlas() const { return m_tlasRHI.get(); }

    // Buffer containing RtInstanceData[] — one entry per TLAS instance.
    VkBuffer instanceDataBuffer() const { return m_instanceDataBuf.handle(); }
    uint32_t instanceCount() const { return m_instanceCount; }

private:
    Device* m_device = nullptr;

    // Backing storage for all BLAS and the TLAS.
    std::vector<VkAccelerationStructureKHR> m_blases;
    std::unique_ptr<rhi::RHIAccelerationStructure> m_tlasRHI;

    // Backing buffers (one large allocation each).
    Buffer m_blasBuf;       // VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
    Buffer m_tlasBuf;       // VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
    Buffer m_scratchBuf;    // max(BLAS scratch, TLAS scratch)
    Buffer m_instanceBuf;   // VkAccelerationStructureInstanceKHR[]
    Buffer m_instanceDataBuf; // RtInstanceData[] — uploaded alongside instance buffer

    uint32_t m_instanceCount = 0;
};

}
