// IBL 烘焙器实现 —— 详见 ibl_baker.h 顶部对整个 pipeline 的描述。
// 本文件按"小工具 → 阶段 dispatch"的顺序组织：
//   1) anon namespace：常量 + barrier 工具 + view/image 分配 + sampler +
//      ComputeKernel（一个 compute pipeline 的 RAII 集合）+ 描述符池工具。
//   2) IblResources::destroy：释放 sampler。
//   3) IblBaker::bake：按阶段顺序 dispatch 所有 compute kernel，并在阶段
//      间插入正确的 layout / 内存屏障。

#include "ibl_baker.h"
#include "core/device.h"
#include "core/buffer.h"
#include "core/shader.h"
#include "scene/upload.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_sampler.h"
#include <array>
#include <cstring>
#include <stdexcept>

namespace somegi {

namespace {

// IBL 资源默认尺寸。提高 size / mips 会显著增加显存与烘焙耗时；
// 256³ envCube 仅约几 MB，但 specular prefilter 是 dispatch 大头。
constexpr uint32_t kEnvCubeSize       = 512;   // envCube 一面边长（mip 0）
constexpr uint32_t kEnvCubeMips       = 6;     // envCube mip 数（供 specular prefilter 取样）
constexpr uint32_t kDiffuseSize       = 32;    // diffuseCube 一面边长，1 mip
constexpr uint32_t kSpecularSize      = 256;   // specularCube mip 0 边长
constexpr uint32_t kSpecularMips      = 6;     // specular mip 数（mip 0=镜面，mip N-1=完全粗糙）
constexpr uint32_t kBrdfLutSize       = 256;   // BRDF LUT 二维分辨率

// 构造一个 VkImageMemoryBarrier2 而不立刻提交（让调用方批量打到 VkDependencyInfo）。
// 由于烘焙过程会大量做 layout 转换（UNDEFINED → GENERAL → TRANSFER_*
// → SHADER_READ_ONLY），抽出来便于复用。
VkImageMemoryBarrier2 imgBarrier(VkImage img,
    VkImageLayout oldL, VkImageLayout newL,
    VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
    VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc,
    uint32_t baseMip, uint32_t mipCount,
    uint32_t baseLayer, uint32_t layerCount)
{
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
    b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
    b.oldLayout = oldL; b.newLayout = newL;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, baseLayer, layerCount};
    return b;
}

void pipelineBarrier(VkCommandBuffer cmd, std::vector<VkImageMemoryBarrier2>& b) {
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = (uint32_t)b.size();
    di.pImageMemoryBarriers = b.data();
    vkCmdPipelineBarrier2(cmd, &di);
}

// 给某个 mip 创建一个 2D_ARRAY 视图（layerCount=6 时是 cube 的 6 面），
// 用于 compute shader 的 RWTexture2DArray 写出。一个 image view 只能覆盖
// 一个 mip，所以多 mip 处理（如 specular prefilter）需要为每个 mip 各
// 创建一个 view。
VkImageView createPerMipArrayView(Device& d, VkImage img, VkFormat fmt,
                                  uint32_t mip, uint32_t layerCount) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img;
    vi.viewType = (layerCount == 6) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, layerCount};
    VkImageView v = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(d.device(), &vi, nullptr, &v));
    return v;
}

// 分配一个 cubemap（6 layer）。
// CUBE_COMPATIBLE_BIT 是关键：让默认创建的 view 自动用 VK_IMAGE_VIEW_TYPE_CUBE，
// shader 端就能 TextureCube.Sample 用方向向量取样。
// usage 包含 SAMPLED + STORAGE + TRANSFER_SRC + TRANSFER_DST：
// - SAMPLED: prefilter 阶段当源采样
// - STORAGE: compute 写 mip 0（imageStore 通过 RWTexture2DArray）
// - TRANSFER_SRC + TRANSFER_DST: vkCmdBlitImage 生成 envCube 的 mip 链
VkImage allocCube(Device& d, uint32_t size, uint32_t mips, VkFormat fmt,
                  VkImageUsageFlags extraUsage, Image& out) {
    ImageDesc id{};
    id.format = fmt;
    id.extent = {size, size, 1};
    id.mipLevels = mips;
    id.arrayLayers = 6;
    id.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    id.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
             | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
             | extraUsage;
    out = Image(d, id);
    return out.image();
}

// 把 CPU 上的 equirect HDR float 数据上传到一张 2D image 上，layout
// 转到 SHADER_READ_ONLY_OPTIMAL。后续 equi_to_cube compute kernel 用它
// 当源采样、写到 envCube 的 mip 0。
Image uploadEquirect(Device& d, VkCommandPool pool, const EnvCpu& env) {
    ImageDesc id{};
    id.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    id.extent = {(uint32_t)env.width, (uint32_t)env.height, 1};
    id.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    Image img(d, id);

    Buffer staging(d, env.rgbaF32.size() * sizeof(float),
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(staging.mapped(), env.rgbaF32.data(), env.rgbaF32.size() * sizeof(float));

    oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
        std::vector<VkImageMemoryBarrier2> bs;
        bs.push_back(imgBarrier(img.image(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            0, 1, 0, 1));
        pipelineBarrier(cmd, bs);

        VkBufferImageCopy c{};
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = id.extent;
        vkCmdCopyBufferToImage(cmd, staging.handle(), img.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);

        bs.clear();
        bs.push_back(imgBarrier(img.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            0, 1, 0, 1));
        pipelineBarrier(cmd, bs);
    });

    return img;
}

// 共享给所有 IBL 资源的线性 sampler：cube 立方贴图、equirect、二维 LUT
// 都用同一个 sampler，节约描述符。clamp 模式给 cubemap / LUT 用，repeat
// 模式给 equirect 横向接缝用（虽然这里我们都用 clamp，因为 cubemap 不
// 应该 wrap）。
VkSampler makeLinearSampler(Device& d, bool clamp, float maxLod) {
    VkSamplerCreateInfo s{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    s.magFilter = VK_FILTER_LINEAR;
    s.minFilter = VK_FILTER_LINEAR;
    s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    s.addressModeU = clamp ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                           : VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.addressModeV = s.addressModeU;
    s.addressModeW = s.addressModeU;
    s.maxLod = maxLod;
    VkSampler sm = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(d.device(), &s, nullptr, &sm));
    return sm;
}

// 一个 compute pipeline 的 RAII 集合 —— DSL + PipelineLayout + Pipeline
// 三件套打包在一起，烘焙的每个阶段（equi_to_cube / prefilter_diffuse / ...）
// 都用一个独立 ComputeKernel。烘焙结束统一 destroy。
struct ComputeKernel {
    Device* d = nullptr;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;

    void destroy() {
        if (!d) return;
        if (pipe) vkDestroyPipeline(d->device(), pipe, nullptr);
        if (pl)   vkDestroyPipelineLayout(d->device(), pl, nullptr);
        if (dsl)  vkDestroyDescriptorSetLayout(d->device(), dsl, nullptr);
        pipe = VK_NULL_HANDLE; pl = VK_NULL_HANDLE; dsl = VK_NULL_HANDLE;
    }
};

// 给定 spv 路径 + descriptor 绑定布局 + push constant 大小，构造完整
// compute pipeline。bindings 描述 shader 使用的各类资源；pushConstantBytes
// 为 0 时不带 push constant。
ComputeKernel makeKernel(Device& d, const std::filesystem::path& spvPath,
                         const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                         uint32_t pushConstantBytes) {
    ComputeKernel k; k.d = &d;

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)bindings.size();
    li.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &k.dsl));

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = pushConstantBytes;

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &k.dsl;
    if (pushConstantBytes > 0) { plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc; }
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &k.pl));

    ShaderModule cs(d, spvPath);
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ss.module = cs.handle();
    ss.pName = "cs_main";

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = ss; cpi.layout = k.pl;
    VK_CHECK(vkCreateComputePipelines(d.device(), VK_NULL_HANDLE, 1, &cpi, nullptr, &k.pipe));
    return k;
}

// 烘焙阶段用的一次性描述符池 —— 烘焙结束后一起 destroy。每个阶段的
// 描述符 set 都从这种池里 allocate，不复用。
VkDescriptorPool makeOneShotDescPool(Device& d, uint32_t maxSets,
                                     const std::vector<VkDescriptorPoolSize>& sizes) {
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = maxSets;
    pci.poolSizeCount = (uint32_t)sizes.size();
    pci.pPoolSizes = sizes.data();
    VkDescriptorPool p = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &p));
    return p;
}

VkDescriptorSet allocDescSet(Device& d, VkDescriptorPool pool, VkDescriptorSetLayout dsl) {
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &dsl;
    VkDescriptorSet s = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &ai, &s));
    return s;
}

}  // anon

void IblResources::destroy(Device& d) {
    if (linear) vkDestroySampler(d.device(), linear, nullptr);
    linear = VK_NULL_HANDLE;
    rhiLinear.reset();
    envCube.reset();
    diffuseCube.reset();
    specularCube.reset();
    brdfLut.reset();
}

// IBL 烘焙主流程。按阶段顺序：
//   阶段 0：上传 equirect、分配四张目标 image、所有目标 image 转 GENERAL。
//   阶段 1：equi_to_cube compute 把 equirect 投影到 envCube mip 0。
//   阶段 2：vkCmdBlitImage 链生成 envCube mip 1..N（线性 box filter）。
//   阶段 3：prefilter_diffuse compute 对 envCube 做 cosine-weighted 半球
//          积分得 diffuseCube。
//   阶段 4：prefilter_specular compute 按 mip 各 dispatch 一次（每个 mip
//          对应一个 roughness）得 specularCube。
//   阶段 5：brdf_lut compute 烘 split-sum BRDF 二维表。
//   阶段 6：所有结果 image 转 SHADER_READ_ONLY，bake 结束。
//
// 阶段间的 layout 转换 + 内存屏障在每段代码末尾就近处理。oneShotSubmit
// 会做 vkQueueWaitIdle，所以阶段间天然有 GPU sync。
void IblBaker::bake(Device& d, VkCommandPool pool, const EnvCpu& env, IblResources& out,
                    rhi::RHIDevice* rhiDevice) {
    out.specularMipCount = kSpecularMips;

    // 1. 把 HDR equirect 上传 GPU（layout 已 → SHADER_READ_ONLY）。
    Image equi = uploadEquirect(d, pool, env);
    // 2. 共享线性 sampler；maxLod = envCubeMips（specular prefilter 阶段
    //    要按算出来的 LOD 在 envCube 上 SampleLevel）。
    out.linear = makeLinearSampler(d, /*clamp*/true, /*maxLod*/(float)kEnvCubeMips);
    // 同时创建 RHI sampler 包装（若提供了 RHI 设备）
    if (rhiDevice) {
        out.rhiLinear = rhi::VkRHISampler::createNonOwning(
            static_cast<rhi::VkRHIDevice&>(*rhiDevice), out.linear);
    }

    // 3. 分配三个 cubemap + 一个 BRDF LUT。
    allocCube(d, kEnvCubeSize,  kEnvCubeMips,  VK_FORMAT_R16G16B16A16_SFLOAT, 0, out.envCube);
    allocCube(d, kDiffuseSize,  1,             VK_FORMAT_R16G16B16A16_SFLOAT, 0, out.diffuseCube);
    allocCube(d, kSpecularSize, kSpecularMips, VK_FORMAT_R16G16B16A16_SFLOAT, 0, out.specularCube);

    {
        // BRDF LUT 不是 cubemap，单独按 2D image 分配。R16G16 两通道存
        // split-sum 的 (A, B) 系数。
        ImageDesc lutDesc{};
        lutDesc.format = VK_FORMAT_R16G16_SFLOAT;
        lutDesc.extent = {kBrdfLutSize, kBrdfLutSize, 1};
        lutDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        out.brdfLut = Image(d, lutDesc);
    }

    // 4. 把 4 张目标 image 全部转到 GENERAL（compute storage 写需要）。
    //    equirect 已经在 SHADER_READ_ONLY，不需要再动。
    oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
        std::vector<VkImageMemoryBarrier2> bs;
        auto pushUndefToGeneral = [&](VkImage img, uint32_t mips, uint32_t layers) {
            bs.push_back(imgBarrier(img,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                0, mips, 0, layers));
        };
        pushUndefToGeneral(out.envCube.image(),      kEnvCubeMips,  6);
        pushUndefToGeneral(out.diffuseCube.image(),  1,             6);
        pushUndefToGeneral(out.specularCube.image(), kSpecularMips, 6);
        pushUndefToGeneral(out.brdfLut.image(),      1,             1);
        pipelineBarrier(cmd, bs);
    });

    // ===== 阶段 1：equirect → cube mip 0 =====
    // equi_to_cube.slang 给每个目标 cube 像素，根据其面 + UV 算出 3D 方向
    // 向量，反查 equirect 的 (phi, theta) UV，用线性 sampler 取色写出。
    {
        ComputeKernel k = makeKernel(
            d, shaderDir() / "gi" / "ibl" / "equi_to_cube.spv",
            {
                {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                {1, VK_DESCRIPTOR_TYPE_SAMPLER,       1, VK_SHADER_STAGE_COMPUTE_BIT},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
            },
            /*pushBytes*/ 16
        );

        VkDescriptorPool dp = makeOneShotDescPool(d, 1, {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        });
        VkDescriptorSet s = allocDescSet(d, dp, k.dsl);

        VkImageView equiView   = equi.view();
        VkImageView cubeMip0   = createPerMipArrayView(d, out.envCube.image(),
                                    VK_FORMAT_R16G16B16A16_SFLOAT, 0, 6);

        VkDescriptorImageInfo srcInfo{};
        srcInfo.imageView = equiView; srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo smpInfo{}; smpInfo.sampler = out.linear;
        VkDescriptorImageInfo dstInfo{};
        dstInfo.imageView = cubeMip0; dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 3> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = s; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &srcInfo;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = s; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[1].pImageInfo = &smpInfo;
        w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[2].dstSet = s; w[2].dstBinding = 2; w[2].descriptorCount = 1;
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo = &dstInfo;
        vkUpdateDescriptorSets(d.device(), 3, w.data(), 0, nullptr);

        oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pl, 0, 1, &s, 0, nullptr);
            struct PC { uint32_t cubeSize, p0, p1, p2; } pc{kEnvCubeSize, 0, 0, 0};
            vkCmdPushConstants(cmd, k.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            uint32_t g = (kEnvCubeSize + 7) / 8;
            vkCmdDispatch(cmd, g, g, 6);
        });

        vkDestroyImageView(d.device(), cubeMip0, nullptr);
        vkDestroyDescriptorPool(d.device(), dp, nullptr);
        k.destroy();
    }

    // ===== 阶段 2：用 vkCmdBlitImage 生成 envCube mip 1..N =====
    // 简单线性下采样链：mip k → mip k+1（半边长，VK_FILTER_LINEAR 做 box
    // 平均）。比 compute pipeline 简单，适合 specular prefilter 的 mip 0
    // 输入需求（不需要高质量过滤，反正 prefilter 自己会按粗糙度做卷积）。
    oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
        int32_t s = (int32_t)kEnvCubeSize;
        for (uint32_t lvl = 1; lvl < kEnvCubeMips; ++lvl) {
            // src mip lvl-1：第一次循环时来自 compute 写出（GENERAL），后续
            // 循环来自上一轮 blit 的 dst（TRANSFER_DST_OPTIMAL）。所以 src
            // 旧 layout / stage / access 要按 lvl 分支选。
            VkImageLayout srcOldLayout = (lvl == 1)
                ? VK_IMAGE_LAYOUT_GENERAL
                : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            VkPipelineStageFlags2 srcOldStage = (lvl == 1)
                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_2_BLIT_BIT;
            VkAccessFlags2 srcOldAccess = (lvl == 1)
                ? VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                : VK_ACCESS_2_TRANSFER_WRITE_BIT;

            std::vector<VkImageMemoryBarrier2> bs;
            bs.push_back(imgBarrier(out.envCube.image(),
                srcOldLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                srcOldStage, srcOldAccess,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                lvl - 1, 1, 0, 6));
            // dst mip lvl: GENERAL → TRANSFER_DST
            bs.push_back(imgBarrier(out.envCube.image(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                lvl, 1, 0, 6));
            pipelineBarrier(cmd, bs);

            VkImageBlit2 blit{VK_STRUCTURE_TYPE_IMAGE_BLIT_2};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lvl - 1, 0, 6};
            blit.srcOffsets[1]   = {s, s, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lvl, 0, 6};
            blit.dstOffsets[1]   = {std::max(s / 2, 1), std::max(s / 2, 1), 1};

            VkBlitImageInfo2 bi{VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2};
            bi.srcImage = out.envCube.image();
            bi.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            bi.dstImage = out.envCube.image();
            bi.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            bi.regionCount = 1; bi.pRegions = &blit;
            bi.filter = VK_FILTER_LINEAR;
            vkCmdBlitImage2(cmd, &bi);

            s = std::max(s / 2, 1);
        }

        // mip 链生成完毕。把所有 mip 一次性转回 SHADER_READ_ONLY_OPTIMAL：
        //   - mip 0..N-2：之前是 TRANSFER_SRC（被前一轮当 src 用过）
        //   - mip N-1   ：之前是 TRANSFER_DST（最后一轮的 dst）
        // 后续 prefilter 阶段把 envCube 整体当 SampledCube 取样。
        std::vector<VkImageMemoryBarrier2> bs;
        bs.push_back(imgBarrier(out.envCube.image(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            0, kEnvCubeMips - 1, 0, 6));
        bs.push_back(imgBarrier(out.envCube.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            kEnvCubeMips - 1, 1, 0, 6));
        pipelineBarrier(cmd, bs);
    });

    // ===== 阶段 3：diffuse 辐照度卷积 =====
    // prefilter_diffuse.slang 对 envCube 做 cosine-weighted 半球积分
    // (Lambert 兰伯特卷积)，逐 phi/theta 累加 Riemann 求和。结果写到
    // diffuseCube（仅 1 mip，32×32 足够）。运行时 evalIBLDiffuse
    // 只需 SampleLevel 0 取样即可。
    {
        ComputeKernel k = makeKernel(
            d, shaderDir() / "gi" / "ibl" / "prefilter_diffuse.spv",
            {
                {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                {1, VK_DESCRIPTOR_TYPE_SAMPLER,       1, VK_SHADER_STAGE_COMPUTE_BIT},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
            },
            /*pushBytes*/ 16);

        VkDescriptorPool dp = makeOneShotDescPool(d, 1, {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        });
        VkDescriptorSet s = allocDescSet(d, dp, k.dsl);

        // envCube view (整 mip) — cube view 让 shader 用 TextureCube 采样
        VkImageViewCreateInfo cvi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        cvi.image = out.envCube.image();
        cvi.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        cvi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        cvi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, kEnvCubeMips, 0, 6};
        VkImageView envCubeView = VK_NULL_HANDLE;
        VK_CHECK(vkCreateImageView(d.device(), &cvi, nullptr, &envCubeView));

        VkImageView diffMip0 = createPerMipArrayView(d, out.diffuseCube.image(),
                                  VK_FORMAT_R16G16B16A16_SFLOAT, 0, 6);

        VkDescriptorImageInfo srcInfo{};
        srcInfo.imageView = envCubeView;
        srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo smpInfo{}; smpInfo.sampler = out.linear;
        VkDescriptorImageInfo dstInfo{};
        dstInfo.imageView = diffMip0; dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 3> w{};
        w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[0].dstSet = s; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &srcInfo;
        w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[1].dstSet = s; w[1].dstBinding = 1; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[1].pImageInfo = &smpInfo;
        w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[2].dstSet = s; w[2].dstBinding = 2; w[2].descriptorCount = 1;
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo = &dstInfo;
        vkUpdateDescriptorSets(d.device(), 3, w.data(), 0, nullptr);

        oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pl, 0, 1, &s, 0, nullptr);
            struct PC { uint32_t outSize, p0, p1, p2; } pc{kDiffuseSize, 0, 0, 0};
            vkCmdPushConstants(cmd, k.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            uint32_t g = (kDiffuseSize + 7) / 8;
            vkCmdDispatch(cmd, g, g, 6);
        });

        vkDestroyImageView(d.device(), diffMip0, nullptr);
        vkDestroyImageView(d.device(), envCubeView, nullptr);
        vkDestroyDescriptorPool(d.device(), dp, nullptr);
        k.destroy();
    }

    // ===== 阶段 4：specular prefilter（每个 mip 一次 dispatch）=====
    // prefilter_specular.slang 用 GGX importance sampling（Hammersley 序列
    // 1024 样本）对 envCube 在某个粗糙度下做 split-sum 第一项的卷积。
    // mip k 对应 roughness = k / (mipCount-1)，共享同一个 compute pipeline，
    // 只是 push constant 的 roughness + outSize 不同。
    // 这是烘焙最耗时的阶段（kSpecularMips × dispatch × 1024 样本）。
    {
        ComputeKernel k = makeKernel(
            d, shaderDir() / "gi" / "ibl" / "prefilter_specular.spv",
            {
                {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                {1, VK_DESCRIPTOR_TYPE_SAMPLER,       1, VK_SHADER_STAGE_COMPUTE_BIT},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
            },
            /*pushBytes*/ 16);

        VkDescriptorPool dp = makeOneShotDescPool(d, kSpecularMips, {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kSpecularMips},
            {VK_DESCRIPTOR_TYPE_SAMPLER, kSpecularMips},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSpecularMips},
        });

        // env cube view (full mips)
        VkImageViewCreateInfo cvi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        cvi.image = out.envCube.image();
        cvi.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        cvi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        cvi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, kEnvCubeMips, 0, 6};
        VkImageView envCubeView = VK_NULL_HANDLE;
        VK_CHECK(vkCreateImageView(d.device(), &cvi, nullptr, &envCubeView));

        std::vector<VkImageView> mipViews(kSpecularMips);
        for (uint32_t lvl = 0; lvl < kSpecularMips; ++lvl) {
            mipViews[lvl] = createPerMipArrayView(d, out.specularCube.image(),
                                VK_FORMAT_R16G16B16A16_SFLOAT, lvl, 6);
        }

        oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
            for (uint32_t lvl = 0; lvl < kSpecularMips; ++lvl) {
                VkDescriptorSet s = allocDescSet(d, dp, k.dsl);

                VkDescriptorImageInfo srcInfo{};
                srcInfo.imageView = envCubeView;
                srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkDescriptorImageInfo smpInfo{}; smpInfo.sampler = out.linear;
                VkDescriptorImageInfo dstInfo{};
                dstInfo.imageView = mipViews[lvl]; dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                std::array<VkWriteDescriptorSet, 3> w{};
                w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w[0].dstSet = s; w[0].dstBinding = 0; w[0].descriptorCount = 1;
                w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[0].pImageInfo = &srcInfo;
                w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w[1].dstSet = s; w[1].dstBinding = 1; w[1].descriptorCount = 1;
                w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[1].pImageInfo = &smpInfo;
                w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w[2].dstSet = s; w[2].dstBinding = 2; w[2].descriptorCount = 1;
                w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo = &dstInfo;
                vkUpdateDescriptorSets(d.device(), 3, w.data(), 0, nullptr);

                uint32_t mipSize = std::max(kSpecularSize >> lvl, 1u);
                float roughness = (kSpecularMips == 1) ? 0.0f : float(lvl) / float(kSpecularMips - 1);

                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pl, 0, 1, &s, 0, nullptr);
                struct PC { uint32_t outSize, envCubeSize; float roughness; uint32_t _p; }
                    pc{mipSize, kEnvCubeSize, roughness, 0};
                vkCmdPushConstants(cmd, k.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                uint32_t g = (mipSize + 7) / 8;
                vkCmdDispatch(cmd, g, g, 6);
            }
        });

        for (auto v : mipViews) vkDestroyImageView(d.device(), v, nullptr);
        vkDestroyImageView(d.device(), envCubeView, nullptr);
        vkDestroyDescriptorPool(d.device(), dp, nullptr);
        k.destroy();
    }

    // ===== 阶段 5：BRDF split-sum LUT =====
    // brdf_lut.slang 烘 split-sum 第二项的二维表：(NoV, roughness) →
    // (A, B) 系数。运行时 evalIBLSpecular 用 (NoV, roughness) 当 UV
    // 取样，结合 prefilter 第一项做 split-sum 近似。LUT 与材质 / 光源
    // 无关，理论上可序列化保存到磁盘只烘一次；目前每次启动都重烘。
    {
        ComputeKernel k = makeKernel(
            d, shaderDir() / "gi" / "ibl" / "brdf_lut.spv",
            {
                {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
            },
            /*pushBytes*/ 16);

        VkDescriptorPool dp = makeOneShotDescPool(d, 1, {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        });
        VkDescriptorSet s = allocDescSet(d, dp, k.dsl);

        VkDescriptorImageInfo dstInfo{};
        dstInfo.imageView = out.brdfLut.view();
        dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = s; w.dstBinding = 0; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w.pImageInfo = &dstInfo;
        vkUpdateDescriptorSets(d.device(), 1, &w, 0, nullptr);

        oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pl, 0, 1, &s, 0, nullptr);
            struct PC { uint32_t size, p0, p1, p2; } pc{kBrdfLutSize, 0, 0, 0};
            vkCmdPushConstants(cmd, k.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            uint32_t g = (kBrdfLutSize + 7) / 8;
            vkCmdDispatch(cmd, g, g, 1);
        });

        vkDestroyDescriptorPool(d.device(), dp, nullptr);
        k.destroy();
    }

    // ===== 阶段 6：收尾 layout 转换 GENERAL → SHADER_READ_ONLY_OPTIMAL =====
    // diffuseCube / specularCube / brdfLut 还在 GENERAL（compute 写出后没
    // 转过），统一转 SHADER_READ_ONLY 供运行时 fragment / compute 阶段
    // 采样。envCube 在阶段 2 mip 链生成结束时已经转好了。
    oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
        std::vector<VkImageMemoryBarrier2> bs;
        auto toSampled = [&](VkImage img, uint32_t mips, uint32_t layers) {
            bs.push_back(imgBarrier(img,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                0, mips, 0, layers));
        };
        toSampled(out.diffuseCube.image(),  1,             6);
        toSampled(out.specularCube.image(), kSpecularMips, 6);
        toSampled(out.brdfLut.image(),      1,             1);
        pipelineBarrier(cmd, bs);
    });
}

}  // namespace somegi
