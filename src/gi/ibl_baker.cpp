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
#include "rhi/vulkan/vk_command.h"
#include "rhi/vulkan/vk_buffer.h"
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

// 创建 TextureBarrierDesc（辅助批量屏障构造）
static rhi::RHICommandBuffer::TextureBarrierDesc mkBar(
    const rhi::RHITexture* tex, rhi::TextureLayout oldL, rhi::TextureLayout newL,
    rhi::PipelineStage srcStg, rhi::BufferAccess srcAcc,
    rhi::PipelineStage dstStg, rhi::BufferAccess dstAcc,
    uint32_t baseMip, uint32_t mipCount, uint32_t baseLayer, uint32_t layerCount)
{
    rhi::RHICommandBuffer::TextureBarrierDesc d{};
    d.texture = tex; d.oldLayout = oldL; d.newLayout = newL;
    d.srcStage = srcStg; d.srcAccess = srcAcc;
    d.dstStage = dstStg; d.dstAccess = dstAcc;
    d.range = {baseMip, mipCount, baseLayer, layerCount};
    return d;
}

// 分配一个 cubemap（6 layer）。
// CUBE_COMPATIBLE_BIT 是关键：让默认创建的 view 自动用 VK_IMAGE_VIEW_TYPE_CUBE，
// shader 端就能 TextureCube.Sample 用方向向量取样。
// usage 包含 SAMPLED + STORAGE + TRANSFER_SRC + TRANSFER_DST：
// - SAMPLED: prefilter 阶段当源采样
// - STORAGE: compute 写 mip 0（imageStore 通过 RWTexture2DArray）
// - TRANSFER_SRC + TRANSFER_DST: blit 生成 envCube 的 mip 链
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
// 全部使用 RHI API（barrier + copy）。
Image uploadEquirect(rhi::RHIDevice& rhiDevice, Device& d, const EnvCpu& env) {
    ImageDesc id{};
    id.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    id.extent = {(uint32_t)env.width, (uint32_t)env.height, 1};
    id.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    Image img(d, id);

    Buffer staging(d, env.rgbaF32.size() * sizeof(float),
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(staging.mapped(), env.rgbaF32.data(), env.rgbaF32.size() * sizeof(float));

    oneShotSubmitRHI(rhiDevice, [&](rhi::RHICommandBuffer& cmd) {
        auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiDevice);

        auto srcBuf = rhi::VkRHIBuffer::createNonOwning(vkDev, staging.handle(), staging.size());
        auto dstTex = rhi::VkRHITexture::createNonOwning(vkDev, img.image(),
            rhi::toRhiFormat(id.format), id.extent.width, id.extent.height);
        cmd.textureBarrier(*dstTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::TransferDst);

        {
            rhi::BufferTextureCopyRegion r;
            r.bufferOffset = 0;
            r.extentWidth = id.extent.width;
            r.extentHeight = id.extent.height;
            cmd.copyBufferToTexture(*srcBuf, *dstTex, r);
        }

        cmd.textureBarrier(*dstTex, rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
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
//   阶段 0：上传 equirect（RHI）、分配四张目标 image、所有目标 image 转 GENERAL（原生 Vulkan）。
//   阶段 1：equi_to_cube compute 把 equirect 投影到 envCube mip 0。（RHI）
//   阶段 2：RHI cmd.blitTexture 链生成 envCube mip 1..N。
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

    // 1. 把 HDR equirect 上传 GPU（layout 已 → SHADER_READ_ONLY，RHI）。
    Image equi = uploadEquirect(*rhiDevice, d, env);

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

    // 4. 把 4 张目标 image 全部转到 GENERAL（compute storage 写需要）。RHI 批量屏障。
    oneShotSubmitRHI(*rhiDevice, [&](rhi::RHICommandBuffer& cmd) {
        auto envTex = rhi::VkRHITexture::createNonOwning(vkDev, out.envCube.image(),
            rhi::toRhiFormat(VK_FORMAT_R16G16B16A16_SFLOAT), kEnvCubeSize, kEnvCubeSize, kEnvCubeMips);
        auto diffTex = rhi::VkRHITexture::createNonOwning(vkDev, out.diffuseCube.image(),
            rhi::toRhiFormat(VK_FORMAT_R16G16B16A16_SFLOAT), kDiffuseSize, kDiffuseSize, 1);
        auto specTex = rhi::VkRHITexture::createNonOwning(vkDev, out.specularCube.image(),
            rhi::toRhiFormat(VK_FORMAT_R16G16B16A16_SFLOAT), kSpecularSize, kSpecularSize, kSpecularMips);
        auto brdfTex = rhi::VkRHITexture::createNonOwning(vkDev, out.brdfLut.image(),
            rhi::toRhiFormat(VK_FORMAT_R16G16_SFLOAT), kBrdfLutSize, kBrdfLutSize, 1);

        rhi::RHICommandBuffer::TextureBarrierDesc bs[4];
        bs[0] = mkBar(envTex.get(),  rhi::TextureLayout::Undefined, rhi::TextureLayout::General,
                      rhi::PipelineStage::TopOfPipe, rhi::BufferAccess::None,
                      rhi::PipelineStage::ComputeShader, rhi::BufferAccess::StorageWrite,
                      0, kEnvCubeMips,  0, 6);
        bs[1] = mkBar(diffTex.get(), rhi::TextureLayout::Undefined, rhi::TextureLayout::General,
                      rhi::PipelineStage::TopOfPipe, rhi::BufferAccess::None,
                      rhi::PipelineStage::ComputeShader, rhi::BufferAccess::StorageWrite,
                      0, 1,             0, 6);
        bs[2] = mkBar(specTex.get(), rhi::TextureLayout::Undefined, rhi::TextureLayout::General,
                      rhi::PipelineStage::TopOfPipe, rhi::BufferAccess::None,
                      rhi::PipelineStage::ComputeShader, rhi::BufferAccess::StorageWrite,
                      0, kSpecularMips, 0, 6);
        bs[3] = mkBar(brdfTex.get(), rhi::TextureLayout::Undefined, rhi::TextureLayout::General,
                      rhi::PipelineStage::TopOfPipe, rhi::BufferAccess::None,
                      rhi::PipelineStage::ComputeShader, rhi::BufferAccess::StorageWrite,
                      0, 1,             0, 1);
        cmd.textureBarriers(4, bs);
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
        auto cubeMip0 = rhi::VkRHITextureView::createNonOwning(vkDev, cubeMip0CI);

        // RHI: 描述符写入
        descSet->write({
            {0, rhi::DescriptorType::SampledImage, equiView.get()},
            {1, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, out.rhiLinear.get()},
            {2, rhi::DescriptorType::StorageImage, cubeMip0.get()},
        });

        // RHI: 提交 compute dispatch
        oneShotSubmitRHI(*rhiDevice, [&](rhi::RHICommandBuffer& cmd) {
            cmd.bindPipelineState(*pso);
            cmd.bindDescriptorSet(0, *descSet);
            struct PC { uint32_t cubeSize, p0, p1, p2; } pc{kEnvCubeSize, 0, 0, 0};
            cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
            uint32_t g = (kEnvCubeSize + 7) / 8;
            cmd.dispatch(g, g, 6);
        });
    }

    // ===== 阶段 2：用 RHI cmd.blitTexture 生成 envCube mip 1..N =====
    // 简单线性下采样链：mip k → mip k+1（半边长，linear filter 做 box
    // 平均）。比 compute pipeline 简单，适合 specular prefilter 的 mip 0
    // 输入需求（不需要高质量过滤，反正 prefilter 自己会按粗糙度做卷积）。
    oneShotSubmitRHI(*rhiDevice, [&](rhi::RHICommandBuffer& rhiCmd) {
        auto envCubeTex = rhi::VkRHITexture::createNonOwning(vkDev, out.envCube.image(),
            rhi::toRhiFormat(VK_FORMAT_R16G16B16A16_SFLOAT), kEnvCubeSize, kEnvCubeSize, kEnvCubeMips);

        int32_t s = (int32_t)kEnvCubeSize;
        for (uint32_t lvl = 1; lvl < kEnvCubeMips; ++lvl) {
            // src mip lvl-1 旧 layout 按 lvl 分支：mip 0 来自 compute（GENERAL），
            // 后续 mip 来自上一轮 blit dst（TRANSFER_DST）。
            rhi::TextureLayout srcOldLayout = (lvl == 1)
                ? rhi::TextureLayout::General
                : rhi::TextureLayout::TransferDst;

            // src mip lvl-1: oldLayout -> TransferSrc
            {
                rhi::RHICommandBuffer::TextureBarrierRange br;
                br.baseMip = lvl - 1;
                br.mipCount = 1;
                br.layerCount = 6;
                rhiCmd.textureBarrier(*envCubeTex, srcOldLayout, rhi::TextureLayout::TransferSrc, br);
            }
            // dst mip lvl: General -> TransferDst
            {
                rhi::RHICommandBuffer::TextureBarrierRange br;
                br.baseMip = lvl;
                br.mipCount = 1;
                br.layerCount = 6;
                rhiCmd.textureBarrier(*envCubeTex, rhi::TextureLayout::General, rhi::TextureLayout::TransferDst, br);
            }

            uint32_t dstSide = std::max(s / 2, 1);
            {
                rhi::TextureBlitRegion r;
                r.srcMipLevel = lvl - 1;
                r.dstMipLevel = lvl;
                r.srcExtentWidth = (uint32_t)s;
                r.srcExtentHeight = (uint32_t)s;
                r.dstExtentWidth = (uint32_t)dstSide;
                r.dstExtentHeight = (uint32_t)dstSide;
                r.layerCount = 6;
                r.linearFilter = true;
                rhiCmd.blitTexture(*envCubeTex, *envCubeTex, r);
            }

            s = (int32_t)dstSide;
        }

        // mip 链生成完毕。把所有 mip 一次性转回 SHADER_READ_ONLY_OPTIMAL：
        //   - mip 0..N-2：之前是 TRANSFER_SRC（被前一轮当 src 用过）
        //   - mip N-1   ：之前是 TRANSFER_DST（最后一轮的 dst）
        // 后续 prefilter 阶段把 envCube 整体当 SampledCube 取样。
        // mip 0..N-2: TransferSrc -> ShaderReadOnly
        {
            rhi::RHICommandBuffer::TextureBarrierRange br;
            br.baseMip = 0;
            br.mipCount = kEnvCubeMips - 1;
            br.layerCount = 6;
            rhiCmd.textureBarrier(*envCubeTex, rhi::TextureLayout::TransferSrc, rhi::TextureLayout::ShaderReadOnly, br);
        }
        // mip N-1: TransferDst -> ShaderReadOnly
        {
            rhi::RHICommandBuffer::TextureBarrierRange br;
            br.baseMip = kEnvCubeMips - 1;
            br.mipCount = 1;
            br.layerCount = 6;
            rhiCmd.textureBarrier(*envCubeTex, rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly, br);
        }
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
        auto envCubeView = rhi::VkRHITextureView::createNonOwning(vkDev, cvi);

        // RHI: diffuseCube per-mip array view
        VkImageViewCreateInfo diffMip0CI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        diffMip0CI.image = out.diffuseCube.image();
        diffMip0CI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        diffMip0CI.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        diffMip0CI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
        auto diffMip0 = rhi::VkRHITextureView::createNonOwning(vkDev, diffMip0CI);

        // RHI: 描述符写入
        descSet->write({
            {0, rhi::DescriptorType::SampledImage, envCubeView.get()},
            {1, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, out.rhiLinear.get()},
            {2, rhi::DescriptorType::StorageImage, diffMip0.get()},
        });

        // RHI: 提交 compute dispatch
        oneShotSubmitRHI(*rhiDevice, [&](rhi::RHICommandBuffer& cmd) {
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
        auto envCubeView = rhi::VkRHITextureView::createNonOwning(vkDev, cvi);

        // RHI: 为每个 mip 创建视图 + 描述符集（预先分配，避免在 submit 内分配）
        std::vector<std::unique_ptr<rhi::RHITextureView>> mipViews(kSpecularMips);
        std::vector<std::unique_ptr<rhi::RHIDescriptorSet>> descSets(kSpecularMips);
        for (uint32_t lvl = 0; lvl < kSpecularMips; ++lvl) {
            VkImageViewCreateInfo mipCI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            mipCI.image = out.specularCube.image();
            mipCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            mipCI.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            mipCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, lvl, 1, 0, 6};
            mipViews[lvl] = rhi::VkRHITextureView::createNonOwning(vkDev, mipCI);

            descSets[lvl] = rhiDevice->createDescriptorSet(*dsl);
            descSets[lvl]->write({
                {0, rhi::DescriptorType::SampledImage, envCubeView.get()},
                {1, rhi::DescriptorType::Sampler, nullptr, nullptr, 0, 0, out.rhiLinear.get()},
                {2, rhi::DescriptorType::StorageImage, mipViews[lvl].get()},
            });
        }

        // RHI: 提交所有 mip 的 dispatch
        oneShotSubmitRHI(*rhiDevice, [&](rhi::RHICommandBuffer& cmd) {
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
        oneShotSubmitRHI(*rhiDevice, [&](rhi::RHICommandBuffer& cmd) {
            cmd.bindPipelineState(*pso);
            cmd.bindDescriptorSet(0, *descSet);
            struct PC { uint32_t size, p0, p1, p2; } pc{kBrdfLutSize, 0, 0, 0};
            cmd.pushConstants(rhi::ShaderStage::Compute, &pc, sizeof(pc));
            uint32_t g = (kBrdfLutSize + 7) / 8;
            cmd.dispatch(g, g, 1);
        });
    }

    // ===== 阶段 6：收尾 layout 转换 GENERAL → SHADER_READ_ONLY =====
    // RHI 批量屏障
    oneShotSubmitRHI(*rhiDevice, [&](rhi::RHICommandBuffer& cmd) {
        auto diffTex = rhi::VkRHITexture::createNonOwning(vkDev, out.diffuseCube.image(),
            rhi::toRhiFormat(VK_FORMAT_R16G16B16A16_SFLOAT), kDiffuseSize, kDiffuseSize, 1);
        auto specTex = rhi::VkRHITexture::createNonOwning(vkDev, out.specularCube.image(),
            rhi::toRhiFormat(VK_FORMAT_R16G16B16A16_SFLOAT), kSpecularSize, kSpecularSize, kSpecularMips);
        auto brdfTex = rhi::VkRHITexture::createNonOwning(vkDev, out.brdfLut.image(),
            rhi::toRhiFormat(VK_FORMAT_R16G16_SFLOAT), kBrdfLutSize, kBrdfLutSize, 1);

        rhi::RHICommandBuffer::TextureBarrierDesc bs2[3];
        bs2[0] = mkBar(diffTex.get(), rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly,
                       rhi::PipelineStage::ComputeShader, rhi::BufferAccess::StorageWrite,
                       rhi::PipelineStage::FragmentShader, rhi::BufferAccess::MemoryRead,
                       0, 1,             0, 6);
        bs2[1] = mkBar(specTex.get(), rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly,
                       rhi::PipelineStage::ComputeShader, rhi::BufferAccess::StorageWrite,
                       rhi::PipelineStage::FragmentShader, rhi::BufferAccess::MemoryRead,
                       0, kSpecularMips, 0, 6);
        bs2[2] = mkBar(brdfTex.get(), rhi::TextureLayout::General, rhi::TextureLayout::ShaderReadOnly,
                       rhi::PipelineStage::ComputeShader, rhi::BufferAccess::StorageWrite,
                       rhi::PipelineStage::FragmentShader, rhi::BufferAccess::MemoryRead,
                       0, 1,             0, 1);
        cmd.textureBarriers(3, bs2);
    });
}

}  // namespace somegi
