#include "scene_gpu.h"
#include "core/device.h"
#include "core/buffer.h"
#include "upload.h"
#include <cstring>

namespace somegi {

static Buffer makeStaging(Device& d, const void* data, size_t size) {
    Buffer b(d, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(b.mapped(), data, size);
    return b;
}

static void uploadBufferImpl(Device& d, VkCommandPool pool,
                             const void* data, size_t size, VkBufferUsageFlags usage,
                             Buffer& out) {
    out = Buffer(d, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer staging = makeStaging(d, data, size);
    oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
        VkBufferCopy c{0, 0, size};
        vkCmdCopyBuffer(cmd, staging.handle(), out.handle(), 1, &c);
    });
}

static void transitionImg(VkCommandBuffer cmd, VkImage img,
                          VkImageLayout oldL, VkImageLayout newL,
                          VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAcc,
                          VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAcc,
                          uint32_t baseMipLevel = 0, uint32_t mipLevels = 1) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask=srcStage; b.srcAccessMask=srcAcc; b.dstStageMask=dstStage; b.dstAccessMask=dstAcc;
    b.oldLayout=oldL; b.newLayout=newL; b.image=img;
    b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT, baseMipLevel, mipLevels, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount=1; di.pImageMemoryBarriers=&b;
    vkCmdPipelineBarrier2(cmd, &di);
}

static void uploadImageImpl(Device& d, VkCommandPool pool, const TextureCpu& cpu, Image& out) {
    if (cpu.width <= 0 || cpu.height <= 0) {
        TextureCpu fallback;
        fallback.width = 1; fallback.height = 1; fallback.channels = 4;
        fallback.rgba = {255, 0, 255, 255}; fallback.isSrgb = cpu.isSrgb;
        uploadImageImpl(d, pool, fallback, out);
        return;
    }
    uint32_t maxDim = cpu.width > cpu.height ? cpu.width : cpu.height;
    uint32_t mipLevels = 1;
    while (maxDim > 1) { maxDim >>= 1; ++mipLevels; }

    VkFormat fmt = cpu.isSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    ImageDesc id{};
    id.format = fmt;
    id.extent = {(uint32_t)cpu.width, (uint32_t)cpu.height, 1};
    id.mipLevels = mipLevels;
    id.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    out = Image(d, id);

    Buffer staging = makeStaging(d, cpu.rgba.data(), cpu.rgba.size());

    oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
        transitionImg(cmd, out.image(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            0, mipLevels);

        VkBufferImageCopy c{};
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = id.extent;
        vkCmdCopyBufferToImage(cmd, staging.handle(), out.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);

        int32_t mipW = (int32_t)cpu.width;
        int32_t mipH = (int32_t)cpu.height;

        for (uint32_t i = 1; i < mipLevels; ++i) {
            VkImageMemoryBarrier2 srcBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            srcBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            srcBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            srcBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            srcBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            srcBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            srcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            srcBarrier.image = out.image();
            srcBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1};

            VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &srcBarrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);

            int32_t nextW = mipW > 1 ? mipW / 2 : 1;
            int32_t nextH = mipH > 1 ? mipH / 2 : 1;

            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mipW, mipH, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {nextW, nextH, 1};

            vkCmdBlitImage(cmd, out.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          out.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          1, &blit, VK_FILTER_LINEAR);

            VkImageMemoryBarrier2 dstBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            dstBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            dstBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            dstBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            dstBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            dstBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            dstBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            dstBarrier.image = out.image();
            dstBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1};

            depInfo.pImageMemoryBarriers = &dstBarrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);

            mipW = nextW;
            mipH = nextH;
        }

        transitionImg(cmd, out.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            mipLevels - 1, 1);
    });
}

static Image makeSolid1x1(Device& d, VkCommandPool pool, uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool srgb) {
    TextureCpu c; c.width=1; c.height=1; c.channels=4;
    c.rgba = {r, g, b, a}; c.isSrgb = srgb;
    Image img;
    uploadImageImpl(d, pool, c, img);
    return img;
}

void uploadScene(Device& d, VkCommandPool pool, const SceneCpu& cpu, SceneGpu& out) {
    VkBufferUsageFlags asInput = 0;
    if (d.features().accelStruct)
        asInput = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    if (!cpu.vertices.empty())
        uploadBufferImpl(d, pool, cpu.vertices.data(), cpu.vertices.size()*sizeof(Vertex),
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | asInput,
                         out.vertexBuffer);
    if (!cpu.indices.empty())
        uploadBufferImpl(d, pool, cpu.indices.data(), cpu.indices.size()*sizeof(uint32_t),
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
    uploadBufferImpl(d, pool, mats.data(), mats.size()*sizeof(MaterialGpu),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, out.materialBuffer);

    out.images.resize(cpu.textures.size());
    for (size_t i = 0; i < cpu.textures.size(); ++i) {
        uploadImageImpl(d, pool, cpu.textures[i], out.images[i]);
    }
    out.whiteTex  = makeSolid1x1(d, pool, 255, 255, 255, 255, true);
    out.normalTex = makeSolid1x1(d, pool, 128, 128, 255, 255, false);

    VkSamplerCreateInfo s{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    s.magFilter = VK_FILTER_LINEAR; s.minFilter = VK_FILTER_LINEAR;
    s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    s.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.maxLod = VK_LOD_CLAMP_NONE;
    s.anisotropyEnable = VK_FALSE;
    VK_CHECK(vkCreateSampler(d.device(), &s, nullptr, &out.linearSampler));
}

void destroySceneSamplers(Device& d, SceneGpu& gpu) {
    if (gpu.linearSampler) vkDestroySampler(d.device(), gpu.linearSampler, nullptr);
    gpu.linearSampler = VK_NULL_HANDLE;
}

}
