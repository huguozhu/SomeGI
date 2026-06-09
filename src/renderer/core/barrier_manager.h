#pragma once
#include "core/vk_common.h"
#include <unordered_map>
#include <vector>

namespace somegi {

// Tracks layout/access state for a single image and automates barrier insertion.
// Usage:
//   1. registerImage() for each image that participates in the pipeline
//   2. transition() to move an image from its current layout to a new one
//   3. The manager automatically inserts VkImageMemoryBarrier2 when needed
class BarrierManager {
public:
    struct State {
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkAccessFlags2 access = 0;
        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    };

    // Register an image. All transitions must be for registered images.
    void registerImage(VkImage image, VkImageAspectFlags aspect,
                       VkImageLayout initial = VK_IMAGE_LAYOUT_UNDEFINED,
                       VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                       VkAccessFlags2 access = 0);

    // Transition to a new layout. Automatically inserts barrier if needed.
    // Returns true if a barrier was written.
    bool transition(VkCommandBuffer cmd, VkImage image,
                    VkImageLayout newLayout,
                    VkPipelineStageFlags2 dstStage,
                    VkAccessFlags2 dstAccess);

    // Convenience shorthands — auto-pick stages and access from layout
    bool toColorAttachment(VkCommandBuffer cmd, VkImage image);
    bool toDepthAttachment(VkCommandBuffer cmd, VkImage image);
    bool toSampled(VkCommandBuffer cmd, VkImage image);
    bool toStorage(VkCommandBuffer cmd, VkImage image);
    bool toTransferSrc(VkCommandBuffer cmd, VkImage image);
    bool toTransferDst(VkCommandBuffer cmd, VkImage image);
    bool toPresent(VkCommandBuffer cmd, VkImage image);

    // Query current state
    State current(VkImage image) const;

private:
    std::unordered_map<VkImage, State> m_states;
};

} // namespace somegi
