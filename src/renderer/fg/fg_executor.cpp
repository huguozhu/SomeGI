// fg_executor.cpp - FrameGraph 执行器实现
// TODO: Phase 3 implementation

#include "fg_executor.h"
#include "fg_pass_node.h"
#include "fg_resource_node.h"
#include "fg_resources.h"

namespace somegi {
namespace fg {

void FGExecutor::init(Device& device) {
    m_device = &device;
}

void FGExecutor::destroy() {
}

void FGExecutor::execute(VkCommandBuffer /*cmd*/,
                          FGCompiler::CompiledGraph& /*compiled*/,
                          const FGResources& /*viewCache*/) {
}

VkImageLayout FGExecutor::derivedLayout(FGPassType /*passType*/,
                                         VkImageUsageFlags /*usage*/,
                                         bool /*isWrite*/) {
    return VK_IMAGE_LAYOUT_UNDEFINED;
}

VkAccessFlags2 FGExecutor::derivedAccess(FGPassType /*passType*/,
                                          VkImageUsageFlags /*usage*/,
                                          bool /*isWrite*/,
                                          bool /*isReadWrite*/) {
    return 0;
}

VkPipelineStageFlags2 FGExecutor::derivedStage(FGPassType /*passType*/,
                                                VkImageUsageFlags /*usage*/,
                                                bool /*isWrite*/) {
    return VK_PIPELINE_STAGE_2_NONE;
}

void FGExecutor::allocateAliasGroup(const FGCompiler::AliasGroup& /*group*/,
                                     std::vector<FGResourceNode*>& /*resources*/) {
}

Image* FGExecutor::allocateTexture(const FGResourceDesc& /*desc*/) {
    return nullptr;
}

Buffer* FGExecutor::allocateBuffer(const FGResourceDesc& /*desc*/) {
    return nullptr;
}

void FGExecutor::emitBarriers(VkCommandBuffer /*cmd*/,
                               const FGPassNode& /*pass*/,
                               std::vector<FGResourceNode*>& /*resources*/,
                               const FGResources& /*viewCache*/) {
}

void FGExecutor::updateResourceStates(const FGPassNode& /*pass*/,
                                       std::vector<FGResourceNode*>& /*resources*/) {
}

void FGExecutor::recycleUnused(uint64_t /*threshold*/) {
}

} // namespace fg
} // namespace somegi
