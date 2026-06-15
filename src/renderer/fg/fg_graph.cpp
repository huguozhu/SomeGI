// fg_graph.cpp - FrameGraph 顶层 API 实现
// TODO: Phase 1 implementation

#include "fg_graph.h"

namespace somegi {
namespace fg {

FrameGraph::FrameGraph() = default;
FrameGraph::~FrameGraph() = default;

void FrameGraph::init(Device& device) {
    m_device = &device;
}

FGHandle FrameGraph::importTexture(const char* /*name*/,
                                    VkImage /*image*/,
                                    const FGTextureDesc& /*desc*/,
                                    VkImageLayout /*initialLayout*/) {
    return {};
}

FGHandle FrameGraph::createTexture(const char* /*name*/, const FGTextureDesc& /*desc*/) {
    return {};
}

FGHandle FrameGraph::createBuffer(const char* /*name*/, const FGBufferDesc& /*desc*/) {
    return {};
}

void FrameGraph::addPass(const char* /*name*/, std::function<void(FGBuilder&)> /*setup*/) {
}

void FrameGraph::compile() {
}

void FrameGraph::execute(VkCommandBuffer /*cmd*/) {
}

VkImageView FrameGraph::getTextureView(FGHandle /*handle*/,
                                        uint32_t /*mip*/,
                                        uint32_t /*layer*/) const {
    return VK_NULL_HANDLE;
}

VkBuffer FrameGraph::getBuffer(FGHandle /*handle*/,
                                VkDeviceSize* outOffset) const {
    if (outOffset) *outOffset = 0;
    return VK_NULL_HANDLE;
}

void FrameGraph::reset() {
}

FGHandle FrameGraph::addManagedResource(const FGResourceDesc& /*desc*/) {
    return {};
}

FGResourceNode* FrameGraph::findResource(FGHandle /*handle*/) {
    return nullptr;
}

const FGResourceNode* FrameGraph::findResource(FGHandle /*handle*/) const {
    return nullptr;
}

void FrameGraph::populateViewCache() {
}

VkPipelineStageFlags2 FrameGraph::stageForPassType(FGPassType /*type*/) {
    return VK_PIPELINE_STAGE_2_NONE;
}

VkPipelineStageFlags2 FrameGraph::readStageForPassType(FGPassType /*type*/) {
    return VK_PIPELINE_STAGE_2_NONE;
}

} // namespace fg
} // namespace somegi
