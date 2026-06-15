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

// ---- 资源声明 ----

FGHandle FrameGraph::importTexture(const char* name,
                                    VkImage image,
                                    const FGTextureDesc& desc,
                                    VkImageLayout initialLayout) {
    (void)image;
    uint32_t idx = (uint32_t)m_resources.size();
    FGHandle h{idx, 0};

    FGResourceNode node;
    node.handle = h;
    node.desc = FGResourceDesc::textureDesc(name, desc.extent, desc.format,
        desc.usage, desc.mipLevels, desc.samples);
    node.desc.texture = desc;
    node.isImported = true;
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
    FGHandle h{idx, 0};

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
    for (auto& tv : m_viewCache.m_textures) {
        if (tv.handle == handle) return tv.view;
    }
    return VK_NULL_HANDLE;
}

VkBuffer FrameGraph::getBuffer(FGHandle handle, VkDeviceSize* outOffset) const {
    for (auto& bv : m_viewCache.m_buffers) {
        if (bv.handle == handle) {
            if (outOffset) *outOffset = bv.offset;
            return bv.buffer;
        }
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
}

// ---- 资源查找 ----

FGResourceNode* FrameGraph::findResource(FGHandle handle) {
    if (!handle.valid() || handle.index >= m_resources.size()) return nullptr;
    return &m_resources[handle.index];
}

const FGResourceNode* FrameGraph::findResource(FGHandle handle) const {
    if (!handle.valid() || handle.index >= m_resources.size()) return nullptr;
    return &m_resources[handle.index];
}

// ---- 视图缓存填充 ----

void FrameGraph::populateViewCache() {
    m_viewCache.m_textures.clear();
    m_viewCache.m_buffers.clear();

    for (auto& res : m_resources) {
        if (res.desc.type == FGResourceType::Texture && res.physicalTexture) {
            FGResources::TextureView tv;
            tv.handle = res.handle;
            tv.view = res.physicalTexture->view();
            tv.extent = res.physicalTexture->extent();
            m_viewCache.m_textures.push_back(tv);
        } else if (res.desc.type == FGResourceType::Buffer && res.physicalBuffer) {
            FGResources::BufferView bv;
            bv.handle = res.handle;
            bv.buffer = res.physicalBuffer->handle();
            bv.size = res.physicalBuffer->size();
            m_viewCache.m_buffers.push_back(bv);
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

FGHandle FGBuilder::createTexture(const char* name, const FGTextureDesc& desc) {
    return m_graph.createTexture(name, desc);
}

FGHandle FGBuilder::createBuffer(const char* name, const FGBufferDesc& desc) {
    return m_graph.createBuffer(name, desc);
}

// ============================================================
// FGResources
// ============================================================

VkImageView FGResources::getTextureView(FGHandle handle, uint32_t mip, uint32_t layer) const {
    (void)mip; (void)layer;
    for (auto& tv : m_textures) {
        if (tv.handle == handle) return tv.view;
    }
    return VK_NULL_HANDLE;
}

VkBuffer FGResources::getBuffer(FGHandle handle, VkDeviceSize* outOffset) const {
    for (auto& bv : m_buffers) {
        if (bv.handle == handle) {
            if (outOffset) *outOffset = bv.offset;
            return bv.buffer;
        }
    }
    return VK_NULL_HANDLE;
}

VkExtent3D FGResources::extent(FGHandle handle) const {
    for (auto& tv : m_textures) {
        if (tv.handle == handle) return tv.extent;
    }
    return {1, 1, 1};
}

} // namespace fg
} // namespace somegi
