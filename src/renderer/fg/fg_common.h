// src/renderer/fg/fg_common.h
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>

namespace somegi {
namespace fg {

// ============================================================
// FGHandle: 轻量级不透明资源句柄
// ============================================================
struct FGHandle {
    uint32_t index = UINT32_MAX;      // 内部资源数组下标
    uint32_t generation = 0;          // 代数，用于 debug 时检测悬空 handle

    bool valid() const { return index != UINT32_MAX; }
    bool operator==(FGHandle o) const {
        return index == o.index && generation == o.generation;
    }
    bool operator!=(FGHandle o) const { return !(*this == o); }
};

// ============================================================
// FGPassType: Pass 类型，决定 Barrier 推导时的 pipeline stage
// ============================================================
enum class FGPassType {
    Compute,        // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
    Graphics,       // VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT + FRAGMENT_SHADER_BIT
    MeshShading,    // VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT + MESH_SHADER_BIT_EXT
    RayTracing,     // VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
};

// ============================================================
// FGResourceType: 资源类型枚举
// ============================================================
enum class FGResourceType {
    Texture,
    Buffer,
};

// ============================================================
// FGTextureDesc: 纹理资源描述符
// ============================================================
struct FGTextureDesc {
    VkExtent3D extent{1, 1, 1};
    VkFormat   format = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t   mipLevels = 1;
    uint32_t   arrayLayers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageUsageFlags usage = 0;
    bool isCubemap = false;
};

// ============================================================
// FGBufferDesc: Buffer 资源描述符
// ============================================================
struct FGBufferDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
};

// ============================================================
// FGResourceDesc: 统一资源描述符
// ============================================================
struct FGResourceDesc {
    FGResourceType type = FGResourceType::Texture;
    const char* debugName = nullptr;

    union {
        FGTextureDesc texture;
        FGBufferDesc  buffer;
    };

    // 显式默认构造：匿名 union 成员含有 NSDMI，隐式默认构造被 delete
    FGResourceDesc() {
        new (&texture) FGTextureDesc();
    }

    // 便捷工厂
    static FGResourceDesc textureDesc(const char* name,
                                       VkExtent3D extent,
                                       VkFormat format,
                                       VkImageUsageFlags usage,
                                       uint32_t mipLevels = 1,
                                       VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) {
        FGResourceDesc d;
        d.type = FGResourceType::Texture;
        d.debugName = name;
        d.texture = {extent, format, mipLevels, 1, samples, usage, false};
        return d;
    }

    static FGResourceDesc bufferDesc(const char* name,
                                      VkDeviceSize size,
                                      VkBufferUsageFlags usage) {
        FGResourceDesc d;
        d.type = FGResourceType::Buffer;
        d.debugName = name;
        d.buffer = {size, usage};
        return d;
    }
};

} // namespace fg
} // namespace somegi
