#include "barrier_manager.h"
#include <cassert>

namespace somegi {

void BarrierManager::registerImage(VkImage image, VkImageAspectFlags aspect,
                                    VkImageLayout initial, VkPipelineStageFlags2 stage,
                                    VkAccessFlags2 access) {
    m_states[image] = {initial, stage, access, aspect};
}

BarrierManager::State BarrierManager::current(VkImage image) const {
    auto it = m_states.find(image);
    assert(it != m_states.end() && "Image not registered");
    return it->second;
}

bool BarrierManager::transition(VkCommandBuffer cmd, VkImage image,
                                 VkImageLayout newLayout,
                                 VkPipelineStageFlags2 dstStage,
                                 VkAccessFlags2 dstAccess) {
    auto it = m_states.find(image);
    assert(it != m_states.end() && "Image not registered");

    State& s = it->second;
    VkImageLayout oldLayout = s.layout;

    // Same layout → no barrier needed
    if (oldLayout == newLayout && s.stage == dstStage && s.access == dstAccess)
        return false;

    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask  = s.stage;
    b.srcAccessMask = s.access;
    b.dstStageMask  = dstStage;
    b.dstAccessMask = dstAccess;
    b.oldLayout = oldLayout;
    b.newLayout = newLayout;
    b.image = image;
    b.subresourceRange = {s.aspect, 0, 1, 0, 1};

    // If transitioning from UNDEFINED, skip execution barrier (contents discarded)
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
        b.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        b.srcAccessMask = 0;
    }

    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);

    s.layout = newLayout;
    s.stage  = dstStage;
    s.access = dstAccess;
    return true;
}

bool BarrierManager::toColorAttachment(VkCommandBuffer cmd, VkImage image) {
    return transition(cmd, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
}

bool BarrierManager::toDepthAttachment(VkCommandBuffer cmd, VkImage image) {
    return transition(cmd, image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
}

bool BarrierManager::toSampled(VkCommandBuffer cmd, VkImage image) {
    return transition(cmd, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

bool BarrierManager::toStorage(VkCommandBuffer cmd, VkImage image) {
    return transition(cmd, image, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

bool BarrierManager::toTransferSrc(VkCommandBuffer cmd, VkImage image) {
    return transition(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_COPY_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT);
}

bool BarrierManager::toTransferDst(VkCommandBuffer cmd, VkImage image) {
    return transition(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_COPY_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

bool BarrierManager::toPresent(VkCommandBuffer cmd, VkImage image) {
    return transition(cmd, image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
}

} // namespace somegi
