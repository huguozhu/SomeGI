#include "scene_gpu.h"
#include "core/device.h"
#include "core/buffer.h"
#include "upload.h"
#include "rhi/base/device.h"
#include "rhi/base/command_buffer.h"
#include "rhi/base/sampler.h"
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_buffer.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_command.h"
#include <cstring>

namespace somegi {

static Buffer makeStaging(Device& d, const void* data, size_t size) {
    Buffer b(d, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(b.mapped(), data, size);
    return b;
}

static void uploadBufferImpl(rhi::RHIDevice& rhiDevice, Device& d,
                             const void* data, size_t size, VkBufferUsageFlags usage,
                             Buffer& out) {
    out = Buffer(d, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer staging = makeStaging(d, data, size);

    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiDevice);
    auto srcBuf = rhi::VkRHIBuffer::createNonOwning(vkDev, staging.handle(), staging.size());
    auto dstBuf = rhi::VkRHIBuffer::createNonOwning(vkDev, out.handle(), out.size());

    oneShotSubmitRHI(rhiDevice, [&](rhi::RHICommandBuffer& cmd) {
        cmd.copyBuffer(*srcBuf, *dstBuf, size);
    });
}

static void uploadImageImpl(rhi::RHIDevice& rhiDevice, Device& d, const TextureCpu& cpu, Image& out, bool useMipmaps = true) {
    if (cpu.width <= 0 || cpu.height <= 0) {
        TextureCpu fallback;
        fallback.width = 1; fallback.height = 1; fallback.channels = 4;
        fallback.rgba = {255, 0, 255, 255}; fallback.isSrgb = cpu.isSrgb;
        uploadImageImpl(rhiDevice, d, fallback, out, useMipmaps);
        return;
    }
    uint32_t mipLevels = 1;
    if (useMipmaps) {
        uint32_t maxDim = cpu.width > cpu.height ? cpu.width : cpu.height;
        while (maxDim > 1) { maxDim >>= 1; ++mipLevels; }
    }

    VkFormat fmt = cpu.isSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    ImageDesc id{};
    id.format = fmt;
    id.extent = {(uint32_t)cpu.width, (uint32_t)cpu.height, 1};
    id.mipLevels = mipLevels;
    id.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (useMipmaps) id.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    out = Image(d, id);

    Buffer staging = makeStaging(d, cpu.rgba.data(), cpu.rgba.size());

    oneShotSubmitRHI(rhiDevice, [&](rhi::RHICommandBuffer& cmd) {
        auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiDevice);

        auto stagingBuf = rhi::VkRHIBuffer::createNonOwning(vkDev, staging.handle(), staging.size());
        auto dstTex = rhi::VkRHITexture::createNonOwning(vkDev, out.image(),
            rhi::toRhiFormat(fmt), (uint32_t)cpu.width, (uint32_t)cpu.height, mipLevels);
        cmd.textureBarrier(*dstTex, rhi::TextureLayout::Undefined, rhi::TextureLayout::TransferDst);

        {
            rhi::BufferTextureCopyRegion r;
            r.bufferOffset = 0;
            r.extentWidth = (uint32_t)cpu.width;
            r.extentHeight = (uint32_t)cpu.height;
            cmd.copyBufferToTexture(*stagingBuf, *dstTex, r);
        }

        if (useMipmaps) {
            int32_t mipW = (int32_t)cpu.width;
            int32_t mipH = (int32_t)cpu.height;

            for (uint32_t i = 1; i < mipLevels; ++i) {
                // src mip i-1: TransferDst -> TransferSrc
                {
                    rhi::RHICommandBuffer::TextureBarrierRange br;
                    br.baseMip = i - 1;
                    br.mipCount = 1;
                    cmd.textureBarrier(*dstTex, rhi::TextureLayout::TransferDst, rhi::TextureLayout::TransferSrc, br);
                }

                int32_t nextW = mipW > 1 ? mipW / 2 : 1;
                int32_t nextH = mipH > 1 ? mipH / 2 : 1;

                {
                    rhi::TextureBlitRegion r;
                    r.srcMipLevel = i - 1;
                    r.dstMipLevel = i;
                    r.srcExtentWidth = (uint32_t)mipW;
                    r.srcExtentHeight = (uint32_t)mipH;
                    r.dstExtentWidth = (uint32_t)nextW;
                    r.dstExtentHeight = (uint32_t)nextH;
                    r.linearFilter = true;
                    cmd.blitTexture(*dstTex, *dstTex, r);
                }

                // src mip i-1: TransferSrc -> ShaderReadOnly
                {
                    rhi::RHICommandBuffer::TextureBarrierRange br;
                    br.baseMip = i - 1;
                    br.mipCount = 1;
                    cmd.textureBarrier(*dstTex, rhi::TextureLayout::TransferSrc, rhi::TextureLayout::ShaderReadOnly, br);
                }

                mipW = nextW;
                mipH = nextH;
            }

            // 最后一级 mip: TransferDst -> ShaderReadOnly
            {
                rhi::RHICommandBuffer::TextureBarrierRange br;
                br.baseMip = mipLevels - 1;
                br.mipCount = 1;
                cmd.textureBarrier(*dstTex, rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly, br);
            }
        } else {
            cmd.textureBarrier(*dstTex, rhi::TextureLayout::TransferDst, rhi::TextureLayout::ShaderReadOnly);
        }
    });
}

static Image makeSolid1x1(rhi::RHIDevice& rhiDevice, Device& d,
                           uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool srgb) {
    TextureCpu c; c.width=1; c.height=1; c.channels=4;
    c.rgba = {r, g, b, a}; c.isSrgb = srgb;
    Image img;
    uploadImageImpl(rhiDevice, d, c, img);
    return img;
}

void uploadScene(Device& d, VkCommandPool pool, const SceneCpu& cpu, SceneGpu& out, bool useMipmaps,
                 rhi::RHIDevice* rhiDevice) {
    VkBufferUsageFlags asInput = 0;
    if (d.features().accelStruct)
        asInput = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    if (!cpu.vertices.empty())
        uploadBufferImpl(*rhiDevice, d, cpu.vertices.data(), cpu.vertices.size()*sizeof(Vertex),
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | asInput,
                         out.vertexBuffer);
    if (!cpu.indices.empty())
        uploadBufferImpl(*rhiDevice, d, cpu.indices.data(), cpu.indices.size()*sizeof(uint32_t),
                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | asInput,
                         out.indexBuffer);

    std::vector<MaterialGpu> mats;
    mats.reserve(cpu.materials.empty() ? 1 : cpu.materials.size());
    for (auto& m : cpu.materials) {
        MaterialGpu g{};
        g.baseColorFactor = m.baseColorFactor;
        g.emissiveFactor = m.emissiveFactor;
        g.metallicFactor = m.metallicFactor;
        g.roughnessFactor = m.roughnessFactor;
        g.normalScale = m.normalScale;
        g.occlusionStrength = m.occlusionStrength;
        g.alphaCutoff = m.alphaCutoff;
        g.baseColorTex = m.baseColorTex;
        g.mrTex = m.mrTex;
        g.normalTex = m.normalTex;
        g.occlusionTex = m.occlusionTex;
        g.emissiveTex = m.emissiveTex;
        g.alphaMode = m.alphaMode;
        g.doubleSided = m.doubleSided;
        mats.push_back(g);
    }
    if (mats.empty()) {
        MaterialGpu g{};
        g.baseColorFactor = glm::vec4(1.0f);
        g.metallicFactor = 0.0f;
        g.roughnessFactor = 0.8f;
        g.baseColorTex = -1; g.mrTex = -1; g.normalTex = -1; g.occlusionTex = -1; g.emissiveTex = -1;
        mats.push_back(g);
    }
    uploadBufferImpl(*rhiDevice, d, mats.data(), mats.size()*sizeof(MaterialGpu),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, out.materialBuffer);

    out.images.resize(cpu.textures.size());
    for (size_t i = 0; i < cpu.textures.size(); ++i) {
        uploadImageImpl(*rhiDevice, d, cpu.textures[i], out.images[i], useMipmaps);
    }
    out.whiteTex  = makeSolid1x1(*rhiDevice, d, 255, 255, 255, 255, true);
    out.normalTex = makeSolid1x1(*rhiDevice, d, 128, 128, 255, 255, false);

    // RHI sampler 创建
    {
        rhi::SamplerDesc sd;
        sd.magFilter = rhi::Filter::Linear;
        sd.minFilter = rhi::Filter::Linear;
        sd.mipmapMode = rhi::SamplerMipmapMode::Linear;
        sd.addressU = rhi::SamplerAddressMode::Repeat;
        sd.addressV = rhi::SamplerAddressMode::Repeat;
        sd.addressW = rhi::SamplerAddressMode::Repeat;
        sd.maxLod = 0.0f;  // VK_LOD_CLAMP_NONE
        out.m_rhiLinearSampler = rhiDevice->createSampler(sd);
        out.linearSampler = static_cast<VkSampler>(out.m_rhiLinearSampler->nativeHandle());
    }

    // 如果提供了 RHI 设备，填充 RHI 缓冲包装
    if (rhiDevice) {
        out.populateRHIWrappers(*rhiDevice);
    }
}

void destroySceneSamplers(Device& d, SceneGpu& gpu) {
    gpu.m_rhiLinearSampler.reset();  // RHI sampler 自动销毁
    gpu.linearSampler = VK_NULL_HANDLE;
}

void SceneGpu::populateRHIWrappers(rhi::RHIDevice& rhiDevice) {
    auto& vkDev = static_cast<rhi::VkRHIDevice&>(rhiDevice);
    m_rhiVertexBuffer = rhi::VkRHIBuffer::createNonOwning(vkDev, vertexBuffer.handle(), vertexBuffer.size());
    m_rhiIndexBuffer  = rhi::VkRHIBuffer::createNonOwning(vkDev, indexBuffer.handle(), indexBuffer.size());
    m_rhiMaterialBuffer = rhi::VkRHIBuffer::createNonOwning(vkDev, materialBuffer.handle(), materialBuffer.size());
}

}
