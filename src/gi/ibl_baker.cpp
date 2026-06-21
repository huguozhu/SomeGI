// IBL 烘焙器实现 —— 详见 ibl_baker.h 顶部对整个 pipeline 的描述。
// 本文件按"小工具 → 阶段 dispatch"的顺序组织：
//   1) anon namespace：常量 + barrier 工具 + view/image 分配 +
//      compute pipeline 与描述符通过 RHI 创建。
//   2) IblResources::destroy：释放 sampler。
//   3) IblBaker::bake：按阶段顺序 dispatch 所有 compute kernel，并在阶段
//      间插入正确的 layout / 内存屏障。

#include "ibl_baker.h"
#include "core/device.h"
#include "core/buffer.h"
#include "core/shader.h"
#include "scene/upload.h"
#include "rhi/base/device.h"
#include "rhi/base/descriptor.h"
#include "rhi/base/pipeline_state.h"
#include "rhi/base/command_buffer.h"
#include "rhi/base/sampler.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_sampler.h"
#include "rhi/vulkan/vk_pso.h"
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
// 烘焙过程中 vkCmdBlitImage2 链需要复杂 barrier（多 mip / 条件 layout），此工具保留
// 原生 Vulkan。
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
// 由于 RHI 没有 copyBufferToTexture，此函数保留原生 Vulkan。
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

}  // anon

void IblResources::destroy(Device& d) {
    rhiLinear.reset();  // RHI sampler 自动销毁
    linear = VK_NULL_HANDLE;
    envCube.reset();
    diffuseCube.reset();
    specularCube.reset();
    brdfLut.reset();
}

// IBL 烘焙主流程。按阶段顺序：
//   阶段 0：上传 equirect、分配四张目标 image、所有目标 image 转 GENERAL（原生 Vulkan）。
//   阶段 1：equi_to_cube compute 把 equirect 投影到 envCube mip 0。（RHI）
//   阶段 2：vkCmdBlitImage 链生成 envCube mip 1..N（保留原生 Vulkan）。
//   阶段 3：prefilter_diffuse compute 对 envCube 做 cosine-weighted 半球
//          积分得 diffuseCube。（RHI）
//   阶段 4：prefilter_specular compute 按 mip 各 dispatch 一次（每个 mip
//          对应一个 roughness）得 specularCube。（RHI）
//   阶段 5：brdf_lut compute 烘 split-sum BRDF 二维表。（RHI）
//   阶段 6：所有结果 image 转 SHADER_READ_ONLY（原生 Vulkan），bake 结束。
//
// 阶段间的 layout 转换 + 内存屏障在每段代码末尾就近处理。oneShotSubmit
// 会做 vkQueueWaitIdle，所以阶段间天然有 GPU sync。
void IblBaker::bake(Device& d, VkCommandPool pool, const EnvCpu& env, IblResources& out,
                    rhi::RHIDevice* rhiDevice) {
    out.specularMipCount = kSpecularMips;
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(*rhiDevice);

    // 1. 把 HDR equirect 上传 GPU（layout 已 → SHADER_READ_ONLY，原生 Vulkan）。
    Image equi = uploadEquirect(d, pool, env);

    // 2. 创建共享线性 sampler（RHI）；同时填充 VkSampler 供原生 Vulkan 阶段使用。
    {
        rhi::SamplerDesc sd;
        sd.magFilter = rhi::Filter::Linear;
        sd.minFilter = rhi::Filter::Linear;
        sd.mipmapMode = rhi::SamplerMipmapMode::Linear;
        sd.addressU = rhi::SamplerAddressMode::ClampToEdge;
        sd.addressV = rhi::SamplerAddressMode::ClampToEdge;
        sd.addressW = rhi::SamplerAddressMode::ClampToEdge;
        sd.maxLod = (float)kEnvCubeMips;
        out.rhiLinear = rhiDevice->createSampler(sd);
        out.linear = static_cast<VkSampler>(out.rhiLinear->nativeHandle());
    }

    // 3. 分配三个 cubemap + 一个 BRDF LUT（原生 core::Image）。
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
    //    保留原生 Vulkan（无 RHI 等效项，且 barrier 一次性批处理更简单）。
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
        // RHI: 描述符布局
        rhi::DescSetLayoutDesc layoutDesc; layoutDesc.debugName = "IBL_EquiToCube";
        layoutDesc.bindings = {
            {0, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {1, rhi::DescriptorType::Sampler,       1, rhi::ShaderStage::Compute},
            {2, rhi::DescriptorType::StorageImage,  1, rhi::ShaderStage::Compute},
        };
        auto dsl = rhiDevice->createDescriptorSetLayout(layoutDesc);

        // RHI: compute PSO
        rhi::ShaderDesc sd; sd.stage = rhi::ShaderStage::Compute; sd.entryPoint = "cs_main";
        auto shader = rhi::VkRHIShader::createFromFile(vkDev, sd,
            shaderDir() / "gi" / "ibl" / "equi_to_cube.spv");
        rhi::ComputePSODesc psoDesc; psoDesc.debugName = "IBL_EquiToCube";
        psoDesc.computeShader = shader.get();
        psoDesc.descriptorSetLayouts = {dsl.get()};
        psoDesc.pushConstants = {{rhi::ShaderStage::Compute, 0, 16}};
        auto pso = rhiDevice->createComputePSO(psoDesc);

        // RHI: 描述符集
        auto descSet = rhiDevice->createDescriptorSet(*dsl);

        // RHI: 图像视图
        auto equiView = rhi::VkRHITextureView::createNonOwning(vkDev, equi.view());
        VkImageViewCreateInfo cubeMip0CI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        cubeMip0CI.image = out.envCube.image();
        cubeMip0CI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        cubeMip0CI.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        cubeMip0CI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
        auto cubeMip0 = rhi::VkRHITextureView::createNonOwning(vkDev.vkDevice(), cubeMip0CI);

        // RHI: 描述符写入
        descSet->write({
            {0, rhi::DescriptorType::SampledImage, equiView.get()},
            {1, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, out.rhiLinear.get()},
            {2, rhi::DescriptorType::StorageImage, cubeMip0.get()},
        });

        // RHI: 提交 compute dispatch
        oneShotSubmitRHI(*rhiDevice, d, pool, [&](rhi::RHICommandBuffer& cmd) {
            cmd.bindPipelineState(*pso);
            cmd.bindDescriptorSet(0, *descSet);
            struct PC { uint32_t cubeSize, p0, p1, p2; } pc{kEnvCubeSize, 0, 0, 0};
            cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
            uint32_t g = (kEnvCubeSize + 7) / 8;
            cmd.dispatch(g, g, 6);
        });
    }

    // ===== 阶段 2：用 vkCmdBlitImage 生成 envCube mip 1..N =====
    // 简单线性下采样链：mip k → mip k+1（半边长，VK_FILTER_LINEAR 做 box
    // 平均）。比 compute pipeline 简单，适合 specular prefilter 的 mip 0
    // 输入需求（不需要高质量过滤，反正 prefilter 自己会按粗糙度做卷积）。
    // RHI 无 blit 等效项，保留原生 Vulkan。
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
        // RHI: 描述符布局
        rhi::DescSetLayoutDesc layoutDesc; layoutDesc.debugName = "IBL_PrefilterDiffuse";
        layoutDesc.bindings = {
            {0, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {1, rhi::DescriptorType::Sampler,       1, rhi::ShaderStage::Compute},
            {2, rhi::DescriptorType::StorageImage,  1, rhi::ShaderStage::Compute},
        };
        auto dsl = rhiDevice->createDescriptorSetLayout(layoutDesc);

        // RHI: compute PSO
        rhi::ShaderDesc sd; sd.stage = rhi::ShaderStage::Compute; sd.entryPoint = "cs_main";
        auto shader = rhi::VkRHIShader::createFromFile(vkDev, sd,
            shaderDir() / "gi" / "ibl" / "prefilter_diffuse.spv");
        rhi::ComputePSODesc psoDesc; psoDesc.debugName = "IBL_PrefilterDiffuse";
        psoDesc.computeShader = shader.get();
        psoDesc.descriptorSetLayouts = {dsl.get()};
        psoDesc.pushConstants = {{rhi::ShaderStage::Compute, 0, 16}};
        auto pso = rhiDevice->createComputePSO(psoDesc);

        // RHI: 描述符集
        auto descSet = rhiDevice->createDescriptorSet(*dsl);

        // RHI: envCube cube view (full mips, CUBE type)
        VkImageViewCreateInfo cvi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        cvi.image = out.envCube.image();
        cvi.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        cvi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        cvi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, kEnvCubeMips, 0, 6};
        auto envCubeView = rhi::VkRHITextureView::createNonOwning(vkDev.vkDevice(), cvi);

        // RHI: diffuseCube per-mip array view
        VkImageViewCreateInfo diffMip0CI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        diffMip0CI.image = out.diffuseCube.image();
        diffMip0CI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        diffMip0CI.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        diffMip0CI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
        auto diffMip0 = rhi::VkRHITextureView::createNonOwning(vkDev.vkDevice(), diffMip0CI);

        // RHI: 描述符写入
        descSet->write({
            {0, rhi::DescriptorType::SampledImage, envCubeView.get()},
            {1, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, out.rhiLinear.get()},
            {2, rhi::DescriptorType::StorageImage, diffMip0.get()},
        });

        // RHI: 提交 compute dispatch
        oneShotSubmitRHI(*rhiDevice, d, pool, [&](rhi::RHICommandBuffer& cmd) {
            cmd.bindPipelineState(*pso);
            cmd.bindDescriptorSet(0, *descSet);
            struct PC { uint32_t outSize, p0, p1, p2; } pc{kDiffuseSize, 0, 0, 0};
            cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
            uint32_t g = (kDiffuseSize + 7) / 8;
            cmd.dispatch(g, g, 6);
        });
    }

    // ===== 阶段 4：specular prefilter（每个 mip 一次 dispatch）=====
    // prefilter_specular.slang 用 GGX importance sampling（Hammersley 序列
    // 1024 样本）对 envCube 在某个粗糙度下做 split-sum 第一项的卷积。
    // mip k 对应 roughness = k / (mipCount-1)，共享同一个 compute pipeline，
    // 只是 push constant 的 roughness + outSize 不同。
    // 这是烘焙最耗时的阶段（kSpecularMips × dispatch × 1024 样本）。
    {
        // RHI: 描述符布局
        rhi::DescSetLayoutDesc layoutDesc; layoutDesc.debugName = "IBL_PrefilterSpecular";
        layoutDesc.bindings = {
            {0, rhi::DescriptorType::SampledImage, 1, rhi::ShaderStage::Compute},
            {1, rhi::DescriptorType::Sampler,       1, rhi::ShaderStage::Compute},
            {2, rhi::DescriptorType::StorageImage,  1, rhi::ShaderStage::Compute},
        };
        auto dsl = rhiDevice->createDescriptorSetLayout(layoutDesc);

        // RHI: compute PSO
        rhi::ShaderDesc sd; sd.stage = rhi::ShaderStage::Compute; sd.entryPoint = "cs_main";
        auto shader = rhi::VkRHIShader::createFromFile(vkDev, sd,
            shaderDir() / "gi" / "ibl" / "prefilter_specular.spv");
        rhi::ComputePSODesc psoDesc; psoDesc.debugName = "IBL_PrefilterSpecular";
        psoDesc.computeShader = shader.get();
        psoDesc.descriptorSetLayouts = {dsl.get()};
        psoDesc.pushConstants = {{rhi::ShaderStage::Compute, 0, 16}};
        auto pso = rhiDevice->createComputePSO(psoDesc);

        // RHI: env cube view (full mips, CUBE type)
        VkImageViewCreateInfo cvi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        cvi.image = out.envCube.image();
        cvi.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        cvi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        cvi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, kEnvCubeMips, 0, 6};
        auto envCubeView = rhi::VkRHITextureView::createNonOwning(vkDev.vkDevice(), cvi);

        // RHI: 为每个 mip 创建视图 + 描述符集（预先分配，避免在 submit 内分配）
        std::vector<std::unique_ptr<rhi::RHITextureView>> mipViews(kSpecularMips);
        std::vector<std::unique_ptr<rhi::RHIDescriptorSet>> descSets(kSpecularMips);
        for (uint32_t lvl = 0; lvl < kSpecularMips; ++lvl) {
            VkImageViewCreateInfo mipCI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            mipCI.image = out.specularCube.image();
            mipCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            mipCI.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            mipCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, lvl, 1, 0, 6};
            mipViews[lvl] = rhi::VkRHITextureView::createNonOwning(vkDev.vkDevice(), mipCI);

            descSets[lvl] = rhiDevice->createDescriptorSet(*dsl);
            descSets[lvl]->write({
                {0, rhi::DescriptorType::SampledImage, envCubeView.get()},
                {1, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, out.rhiLinear.get()},
                {2, rhi::DescriptorType::StorageImage, mipViews[lvl].get()},
            });
        }

        // RHI: 提交所有 mip 的 dispatch
        oneShotSubmitRHI(*rhiDevice, d, pool, [&](rhi::RHICommandBuffer& cmd) {
            cmd.bindPipelineState(*pso);
            for (uint32_t lvl = 0; lvl < kSpecularMips; ++lvl) {
                cmd.bindDescriptorSet(0, *descSets[lvl]);

                uint32_t mipSize = std::max(kSpecularSize >> lvl, 1u);
                float roughness = (kSpecularMips == 1) ? 0.0f : float(lvl) / float(kSpecularMips - 1);

                struct PC { uint32_t outSize, envCubeSize; float roughness; uint32_t _p; }
                    pc{mipSize, kEnvCubeSize, roughness, 0};
                cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
                uint32_t g = (mipSize + 7) / 8;
                cmd.dispatch(g, g, 6);
            }
        });
    }

    // ===== 阶段 5：BRDF split-sum LUT =====
    // brdf_lut.slang 烘 split-sum 第二项的二维表：(NoV, roughness) →
    // (A, B) 系数。运行时 evalIBLSpecular 用 (NoV, roughness) 当 UV
    // 取样，结合 prefilter 第一项做 split-sum 近似。LUT 与材质 / 光源
    // 无关，理论上可序列化保存到磁盘只烘一次；目前每次启动都重烘。
    {
        // RHI: 描述符布局（仅一个 storage image）
        rhi::DescSetLayoutDesc layoutDesc; layoutDesc.debugName = "IBL_BRDFLut";
        layoutDesc.bindings = {
            {0, rhi::DescriptorType::StorageImage, 1, rhi::ShaderStage::Compute},
        };
        auto dsl = rhiDevice->createDescriptorSetLayout(layoutDesc);

        // RHI: compute PSO
        rhi::ShaderDesc sd; sd.stage = rhi::ShaderStage::Compute; sd.entryPoint = "cs_main";
        auto shader = rhi::VkRHIShader::createFromFile(vkDev, sd,
            shaderDir() / "gi" / "ibl" / "brdf_lut.spv");
        rhi::ComputePSODesc psoDesc; psoDesc.debugName = "IBL_BRDFLut";
        psoDesc.computeShader = shader.get();
        psoDesc.descriptorSetLayouts = {dsl.get()};
        psoDesc.pushConstants = {{rhi::ShaderStage::Compute, 0, 16}};
        auto pso = rhiDevice->createComputePSO(psoDesc);

        // RHI: 描述符集
        auto descSet = rhiDevice->createDescriptorSet(*dsl);

        // RHI: BRDF LUT view（非拥有型包装，Image 管理生命周期）
        auto brdfView = rhi::VkRHITextureView::createNonOwning(vkDev, out.brdfLut.view());

        // RHI: 描述符写入
        descSet->write({
            {0, rhi::DescriptorType::StorageImage, brdfView.get()},
        });

        // RHI: 提交 compute dispatch
        oneShotSubmitRHI(*rhiDevice, d, pool, [&](rhi::RHICommandBuffer& cmd) {
            cmd.bindPipelineState(*pso);
            cmd.bindDescriptorSet(0, *descSet);
            struct PC { uint32_t size, p0, p1, p2; } pc{kBrdfLutSize, 0, 0, 0};
            cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
            uint32_t g = (kBrdfLutSize + 7) / 8;
            cmd.dispatch(g, g, 1);
        });
    }

    // ===== 阶段 6：收尾 layout 转换 GENERAL → SHADER_READ_ONLY_OPTIMAL =====
    // diffuseCube / specularCube / brdfLut 还在 GENERAL（compute 写出后没
    // 转过），统一转 SHADER_READ_ONLY 供运行时 fragment / compute 阶段
    // 采样。envCube 在阶段 2 mip 链生成结束时已经转好了。
    // 保留原生 Vulkan（简单批处理 barrier，RHI 无等效批处理）。
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
