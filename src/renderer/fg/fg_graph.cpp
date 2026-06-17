// src/renderer/fg/fg_graph.cpp
#include "fg_graph.h"
#include "fg_pass_node.h"
#include "fg_resource_node.h"
#include "core/device.h"

namespace somegi {
namespace fg {

// ============================================================
// FrameGraph
// ============================================================

FrameGraph::FrameGraph() = default;
FrameGraph::~FrameGraph() = default;

void FrameGraph::init(Device& device) {
    m_device = &device;
    m_executor.init(device);
}

void FrameGraph::destroy() {
    m_executor.destroy();
}

// ---- 资源声明 ----

FGHandle FrameGraph::importTexture(const char* name,
                                    VkImage image,
                                    const FGTextureDesc& desc,
                                    VkImageLayout initialLayout) {
    uint32_t idx = (uint32_t)m_resources.size();
    FGHandle h{idx, m_resourceGeneration};

    FGResourceNode node;
    node.handle = h;
    node.desc = FGResourceDesc::textureDesc(name, desc.extent, desc.format,
        desc.usage, desc.mipLevels, desc.samples);
    node.desc.texture = desc;
    node.isImported = true;
    node.importedImage = image;  // 保存 VkImage 供 barrier 发射使用
    node.state.layout = initialLayout;

    m_resources.push_back(std::move(node));
    m_resourceNameMap[name] = idx;
    return h;
}

FGHandle FrameGraph::createTexture(const char* name, const FGTextureDesc& desc) {
    return addManagedResource(
        FGResourceDesc::textureDesc(name, desc.extent, desc.format,
            desc.usage, desc.mipLevels, desc.samples));
}

FGHandle FrameGraph::createBuffer(const char* name, const FGBufferDesc& desc) {
    return addManagedResource(
        FGResourceDesc::bufferDesc(name, desc.size, desc.usage));
}

FGHandle FrameGraph::addManagedResource(const FGResourceDesc& desc) {
    uint32_t idx = (uint32_t)m_resources.size();
    FGHandle h{idx, m_resourceGeneration};

    FGResourceNode node;
    node.handle = h;
    node.desc = desc;
    node.isImported = false;

    m_resources.push_back(std::move(node));
    if (desc.debugName) {
        m_resourceNameMap[desc.debugName] = idx;
    }
    return h;
}

// ---- Pass 声明 ----

void FrameGraph::addPass(const char* name, std::function<void(FGBuilder&)> setup) {
    FGPassNode node;
    node.name = name;
    node.passType = FGPassType::Compute;

    m_passes.push_back(std::move(node));
    FGPassNode& passNode = m_passes.back();

    FGBuilder builder(*this, passNode);
    setup(builder);

    // 自动推导 pass 类型：若 write 包含 COLOR_ATTACHMENT 或 DEPTH_STENCIL，
    // 且未显式设置类型，则切换为 Graphics
    bool hasAttachment = false;
    for (auto& ref : passNode.writes) {
        if (ref.resource && ref.resource->desc.type == FGResourceType::Texture) {
            auto usage = ref.resource->desc.texture.usage;
            if (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
                hasAttachment = true;
            }
        }
    }
    if (hasAttachment && passNode.passType == FGPassType::Compute) {
        passNode.passType = FGPassType::Graphics;
    }
}

// ---- 编译 ----

void FrameGraph::compile() {
    std::vector<FGPassNode*> passPtrs;
    passPtrs.reserve(m_passes.size());
    for (auto& p : m_passes) passPtrs.push_back(&p);

    std::vector<FGResourceNode*> resPtrs;
    resPtrs.reserve(m_resources.size());
    for (auto& r : m_resources) resPtrs.push_back(&r);

    m_compiled = m_compiler.compile(passPtrs, resPtrs);
    m_debug.populate(m_compiled, m_passes, m_resources);
    m_compiledThisFrame = true;
}

// ---- 执行 ----

void FrameGraph::execute(VkCommandBuffer cmd) {
    if (!m_compiledThisFrame) return;
    populateViewCache();
    m_executor.execute(cmd, m_compiled, m_viewCache);
    ++m_frameIndex;
}

// ---- 查询 ----

VkImageView FrameGraph::getTextureView(FGHandle handle, uint32_t mip, uint32_t layer) const {
    (void)mip; (void)layer;
    if (handle.index < m_viewCache.m_textureByIndex.size())
        return m_viewCache.m_textureByIndex[handle.index].view;
    return VK_NULL_HANDLE;
}

VkBuffer FrameGraph::getBuffer(FGHandle handle, VkDeviceSize* outOffset) const {
    if (handle.index < m_viewCache.m_bufferByIndex.size()) {
        auto& bv = m_viewCache.m_bufferByIndex[handle.index];
        if (outOffset) *outOffset = bv.offset;
        return bv.buffer;
    }
    return VK_NULL_HANDLE;
}

// ---- 帧管理 ----

void FrameGraph::reset() {
    m_passes.clear();
    m_resources.clear();
    m_resourceNameMap.clear();
    m_compiled = FGCompiler::CompiledGraph{};
    m_compiledThisFrame = false;
    m_viewCache = FGResources{};
    ++m_resourceGeneration;  // 递增代数，使旧 FGHandle 失效
}

// ---- 资源查找 ----

FGResourceNode* FrameGraph::findResource(FGHandle handle) {
    if (!handle.valid() || handle.index >= m_resources.size()) return nullptr;
    auto& res = m_resources[handle.index];
    // 验证代数：若 handle 来自旧帧（reset 后 generation 递增），则不匹配
    if (res.handle.generation != handle.generation) return nullptr;
    return &res;
}

const FGResourceNode* FrameGraph::findResource(FGHandle handle) const {
    if (!handle.valid() || handle.index >= m_resources.size()) return nullptr;
    auto& res = m_resources[handle.index];
    if (res.handle.generation != handle.generation) return nullptr;
    return &res;
}

// ---- 视图缓存填充 ----

void FrameGraph::populateViewCache() {
    uint32_t n = (uint32_t)m_resources.size();
    m_viewCache.m_textures.clear();
    m_viewCache.m_buffers.clear();
    m_viewCache.m_textureByIndex.assign(n, {});
    m_viewCache.m_bufferByIndex.assign(n, {});

    for (auto& res : m_resources) {
        uint32_t idx = res.handle.index;
        if (res.desc.type == FGResourceType::Texture && res.physicalTexture && idx < n) {
            FGResources::TextureView tv;
            tv.handle = res.handle;
            tv.view = res.physicalTexture->view();
            tv.extent = res.physicalTexture->extent();
            m_viewCache.m_textures.push_back(tv);
            m_viewCache.m_textureByIndex[idx] = tv;
        } else if (res.desc.type == FGResourceType::Buffer && res.physicalBuffer && idx < n) {
            FGResources::BufferView bv;
            bv.handle = res.handle;
            bv.buffer = res.physicalBuffer->handle();
            bv.size = res.physicalBuffer->size();
            m_viewCache.m_buffers.push_back(bv);
            m_viewCache.m_bufferByIndex[idx] = bv;
        }
    }
}

// ---- Pipeline stage 推导 ----

VkPipelineStageFlags2 FrameGraph::stageForPassType(FGPassType type) {
    switch (type) {
        case FGPassType::Compute:     return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        case FGPassType::Graphics:    return VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case FGPassType::MeshShading: return VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                             VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case FGPassType::RayTracing:  return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    }
    return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
}

VkPipelineStageFlags2 FrameGraph::readStageForPassType(FGPassType type) {
    switch (type) {
        case FGPassType::Compute:     return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        case FGPassType::Graphics:    return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case FGPassType::MeshShading: return VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case FGPassType::RayTracing:  return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    }
    return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
}

// ============================================================
// FGBuilder
// ============================================================

FGBuilder::FGBuilder(FrameGraph& graph, FGPassNode& passNode)
    : m_graph(graph), m_passNode(&passNode) {}

FGBuilder& FGBuilder::setPassType(FGPassType type) {
    m_passNode->passType = type;
    return *this;
}

FGHandle FGBuilder::read(FGHandle handle) {
    auto* res = m_graph.findResource(handle);
    if (!res) return handle;

    FGPassNode::ResourceRef ref;
    ref.handle = handle;
    ref.resource = res;

    bool isTexture = (res->desc.type == FGResourceType::Texture);
    VkImageUsageFlags usage = isTexture ? res->desc.texture.usage : (VkImageUsageFlags)0;
    ref.access = FGExecutor::derivedAccess(m_passNode->passType, usage, false, false);
    ref.stages = FrameGraph::readStageForPassType(m_passNode->passType);
    ref.requiredLayout = isTexture ? FGExecutor::derivedLayout(
        m_passNode->passType, usage, false) : VK_IMAGE_LAYOUT_UNDEFINED;

    m_passNode->reads.push_back(ref);
    return handle;
}

FGHandle FGBuilder::write(FGHandle handle) {
    auto* res = m_graph.findResource(handle);
    if (!res) return handle;

    FGPassNode::ResourceRef ref;
    ref.handle = handle;
    ref.resource = res;

    bool isTexture = (res->desc.type == FGResourceType::Texture);
    VkImageUsageFlags usage = isTexture ? res->desc.texture.usage : (VkImageUsageFlags)0;
    ref.access = FGExecutor::derivedAccess(m_passNode->passType, usage, true, false);
    ref.stages = FGExecutor::derivedStage(m_passNode->passType, usage, true);
    ref.requiredLayout = isTexture ? FGExecutor::derivedLayout(
        m_passNode->passType, usage, true) : VK_IMAGE_LAYOUT_UNDEFINED;

    m_passNode->writes.push_back(ref);
    return handle;
}

FGHandle FGBuilder::readWrite(FGHandle handle) {
    auto* res = m_graph.findResource(handle);
    if (!res) return handle;

    FGPassNode::ResourceRef ref;
    ref.handle = handle;
    ref.resource = res;

    bool isTexture = (res->desc.type == FGResourceType::Texture);
    VkImageUsageFlags usage = isTexture ? res->desc.texture.usage : (VkImageUsageFlags)0;
    ref.access = FGExecutor::derivedAccess(m_passNode->passType, usage, true, true);
    ref.stages = FGExecutor::derivedStage(m_passNode->passType, usage, true);
    ref.requiredLayout = isTexture ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;

    m_passNode->reads.push_back(ref);
    m_passNode->writes.push_back(ref);
    return handle;
}

// ---- 显式 Layout 重载 ----

// 根据显式 layout 推导 access 和 stage（用于非标准 shader read/write layout）
static void deriveFromLayout(VkImageLayout layout, VkAccessFlags2& access, VkPipelineStageFlags2& stages) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            access  = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            stages  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            access  = VK_ACCESS_2_TRANSFER_READ_BIT;
            stages  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            access  = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            stages  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
            break;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            access  = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
            stages  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            break;
        default:
            // 非标准 layout — 保留自动推导
            access = 0;
            stages = 0;
            break;
    }
}

FGHandle FGBuilder::read(FGHandle handle, VkImageLayout explicitLayout) {
    auto* res = m_graph.findResource(handle);
    if (!res) return handle;

    FGPassNode::ResourceRef ref;
    ref.handle = handle;
    ref.resource = res;
    ref.requiredLayout = explicitLayout;

    bool isTexture = (res->desc.type == FGResourceType::Texture);
    VkImageUsageFlags usage = isTexture ? res->desc.texture.usage : (VkImageUsageFlags)0;

    VkAccessFlags2 derivedAcc = 0;
    VkPipelineStageFlags2 derivedStg = 0;
    deriveFromLayout(explicitLayout, derivedAcc, derivedStg);

    ref.access  = derivedAcc ? derivedAcc : FGExecutor::derivedAccess(m_passNode->passType, usage, false, false);
    ref.stages  = derivedStg ? derivedStg : FrameGraph::readStageForPassType(m_passNode->passType);

    m_passNode->reads.push_back(ref);
    return handle;
}

FGHandle FGBuilder::write(FGHandle handle, VkImageLayout explicitLayout) {
    auto* res = m_graph.findResource(handle);
    if (!res) return handle;

    FGPassNode::ResourceRef ref;
    ref.handle = handle;
    ref.resource = res;
    ref.requiredLayout = explicitLayout;

    bool isTexture = (res->desc.type == FGResourceType::Texture);
    VkImageUsageFlags usage = isTexture ? res->desc.texture.usage : (VkImageUsageFlags)0;

    VkAccessFlags2 derivedAcc = 0;
    VkPipelineStageFlags2 derivedStg = 0;
    deriveFromLayout(explicitLayout, derivedAcc, derivedStg);

    ref.access  = derivedAcc ? derivedAcc : FGExecutor::derivedAccess(m_passNode->passType, usage, true, false);
    ref.stages  = derivedStg ? derivedStg : FGExecutor::derivedStage(m_passNode->passType, usage, true);

    m_passNode->writes.push_back(ref);
    return handle;
}

FGHandle FGBuilder::createTexture(const char* name, const FGTextureDesc& desc) {
    return m_graph.createTexture(name, desc);
}

FGHandle FGBuilder::createBuffer(const char* name, const FGBufferDesc& desc) {
    return m_graph.createBuffer(name, desc);
}

FGBuilder& FGBuilder::setExitLayout(FGHandle handle, VkImageLayout layout) {
    // 在 write 列表中查找 handle，设置其 exitLayout
    for (auto& ref : m_passNode->writes) {
        if (ref.handle == handle) {
            ref.exitLayout = layout;
            return *this;
        }
    }
    // 如果不在 write 列表中，也在 read 列表中查找
    for (auto& ref : m_passNode->reads) {
        if (ref.handle == handle) {
            ref.exitLayout = layout;
            return *this;
        }
    }
    return *this;
}

// ============================================================
// FGResources
// ============================================================

VkImageView FGResources::getTextureView(FGHandle handle, uint32_t mip, uint32_t layer) const {
    (void)mip; (void)layer;
    if (handle.index < m_textureByIndex.size())
        return m_textureByIndex[handle.index].view;
    return VK_NULL_HANDLE;
}

VkBuffer FGResources::getBuffer(FGHandle handle, VkDeviceSize* outOffset) const {
    if (handle.index < m_bufferByIndex.size()) {
        auto& bv = m_bufferByIndex[handle.index];
        if (outOffset) *outOffset = bv.offset;
        return bv.buffer;
    }
    return VK_NULL_HANDLE;
}

VkExtent3D FGResources::extent(FGHandle handle) const {
    if (handle.index < m_textureByIndex.size()) {
        auto& tv = m_textureByIndex[handle.index];
        if (tv.view) return tv.extent;
    }
    return {1, 1, 1};
}

} // namespace fg
} // namespace somegi
