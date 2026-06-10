#pragma once
#include "core/vk_common.h"
#include <vector>
#include <filesystem>

namespace somegi {

class Device;

// Mesh Shader pipeline 创建描述
struct MeshPipelineDesc {
    std::filesystem::path meshSpv;    // Mesh Shader SPV 路径
    std::filesystem::path taskSpv;    // Task Shader SPV 路径（可为空，跳过 Task Stage）
    std::filesystem::path fragSpv;    // Fragment Shader SPV 路径
    const char* meshEntry = "ms_main";
    const char* taskEntry = "ts_main";
    const char* fragEntry = "ps_main";
    std::vector<VkDescriptorSetLayout> setLayouts;  // set=0..N
    std::vector<VkFormat> colorFormats;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    bool depthTest  = true;
    bool depthWrite = true;
};

// mesh pipeline 创建结果（pipeline + layout 需同步管理生命周期）
struct MeshPipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
};

// 创建 Mesh Shader graphics pipeline（task + mesh + fragment 三阶段）
MeshPipeline createMeshPipeline(Device& d, const MeshPipelineDesc& desc);

} // namespace somegi
