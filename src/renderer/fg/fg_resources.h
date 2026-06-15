// src/renderer/fg/fg_resources.h
#pragma once
#include "fg_common.h"
#include <vulkan/vulkan.h>
#include <vector>

namespace somegi {
namespace fg {

// ============================================================
// FGResources: 提供给 pass execute 回调的只读资源视图
//
// 内部缓存了 handle → VkImageView / VkBuffer 的映射，
// 在 FrameGraph::execute() 开始时一次性填充。
// ============================================================
class FGResources {
public:
    // 获取纹理视图（mip=0, layer=0 默认）
    VkImageView getTextureView(FGHandle handle,
                                uint32_t mip = 0,
                                uint32_t layer = 0) const;

    // 获取 Buffer handle + offset
    VkBuffer getBuffer(FGHandle handle,
                       VkDeviceSize* outOffset = nullptr) const;

    // 获取资源 extent
    VkExtent3D extent(FGHandle handle) const;

private:
    friend class FrameGraph;

    struct TextureView {
        FGHandle handle;
        VkImageView view = VK_NULL_HANDLE;
        VkExtent3D extent{1, 1, 1};
    };

    struct BufferView {
        FGHandle handle;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
    };

    std::vector<TextureView> m_textures;
    std::vector<BufferView>  m_buffers;
};

} // namespace fg
} // namespace somegi
