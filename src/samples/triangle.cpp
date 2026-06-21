// Vulkan RHI minimal sample — draw a triangle through pure rhi:: abstractions.
#include "rhi/vulkan/vk_device.h"
#include "rhi/vulkan/vk_swapchain.h"
#include "rhi/vulkan/vk_shader.h"
#include "rhi/vulkan/vk_command.h"
#include "rhi/vulkan/vk_pso.h"
#include "rhi/vulkan/vk_descriptor.h"
#include "rhi/vulkan/vk_texture.h"
#include "rhi/vulkan/vk_fence.h"
#include "core/window.h"
#include <cstdio>
#include <fstream>
#include <vector>

using namespace somegi;

static std::vector<uint32_t> loadSpv(const char* name) {
    std::ifstream f(name, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("open: ") + name);
    auto size = static_cast<size_t>(f.tellg());
    std::vector<uint32_t> data(size / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

int main() {
    WindowDesc wd; wd.title = "RHI Triangle"; wd.width = 800; wd.height = 600;
    Window window(wd);
    rhi::VkRHIDevice rhiDevice(&window, false);  // validation off: 验证层 bug 在第三次 draw 时 NULL deref (offset 0xD0)

    auto pSwapchain = rhiDevice.createSwapchain(nullptr, wd.width, wd.height);
    auto& swapchain = static_cast<rhi::VkRHISwapchain&>(*pSwapchain);

    auto vertSpv = loadSpv("triangle.vert.spv");
    auto fragSpv = loadSpv("triangle.frag.spv");
    rhi::ShaderDesc sd;
    sd.stage = rhi::ShaderStage::Vertex; sd.entryPoint = "main";
    auto vs = rhiDevice.createShader(sd, vertSpv.data(), vertSpv.size() * 4);
    sd.stage = rhi::ShaderStage::Fragment;
    auto fs = rhiDevice.createShader(sd, fragSpv.data(), fragSpv.size() * 4);

    rhi::DescSetLayoutDesc ld; ld.debugName = "Triangle";
    auto setLayout = rhiDevice.createDescriptorSetLayout(ld); // 需要至少一个 layout（即使为空），避免 pSetLayouts=nullptr
    auto emptySet = rhiDevice.createDescriptorSet(*setLayout);

    rhi::GraphicsPSODesc pd; pd.debugName = "Triangle";
    pd.vertexShader = vs.get(); pd.fragmentShader = fs.get();
    pd.topology = rhi::PrimitiveTopology::TriangleList;
    pd.rasterization = {rhi::FillMode::Solid, rhi::CullMode::None, false};
    pd.renderTargets.colorFormats = {pSwapchain->format()};
    pd.descriptorSetLayouts = {setLayout.get()};
    auto pso = rhiDevice.createGraphicsPSO(pd);

    auto cmdPool = rhiDevice.createCommandPool();
    auto* rawCmd0 = cmdPool->allocateRaw();
    auto* rawCmd1 = cmdPool->allocateRaw();
    auto& cmd0 = static_cast<rhi::VkRHICommandBuffer&>(*rawCmd0);
    auto& cmd1 = static_cast<rhi::VkRHICommandBuffer&>(*rawCmd1);

    uint32_t frameIdx = 0;
    while (!window.shouldClose()) {
        window.pollEvents();

        auto frame = pSwapchain->acquireNextFrame();
        if (frame.needsResize) { pSwapchain->recreate(); continue; }

        auto& cmd = (frame.frameInFlight == 0) ? cmd0 : cmd1;
        cmd.reset();
        cmd.begin();

        auto swTex = rhi::VkRHITexture::createNonOwning(rhiDevice,
            swapchain.vkImage(frame.imageIndex),
            pSwapchain->format(), frame.width, frame.height, 1);

        cmd.textureBarrier(*swTex,
            rhi::TextureLayout::Undefined, rhi::TextureLayout::ColorAttachment);

        rhi::RenderingAttachmentInfo color{};
        color.view = frame.view.get();
        color.loadOp = rhi::AttachmentLoadOp::Clear;
        color.storeOp = rhi::AttachmentStoreOp::Store;
        color.clearColor[0] = 0.1f; color.clearColor[2] = 0.3f;
        cmd.beginRendering(&color, 1, nullptr, frame.width, frame.height);
        cmd.bindPipelineState(*pso);
        cmd.bindDescriptorSet(0, *emptySet);
        cmd.setViewport(0, 0, (float)frame.width, (float)frame.height);
        cmd.setScissor(0, 0, frame.width, frame.height);
        cmd.draw(3, 0, 0);
        cmd.endRendering();

        cmd.textureBarrier(*swTex,
            rhi::TextureLayout::ColorAttachment, rhi::TextureLayout::Present);

        cmd.end();

        auto sigFence = rhi::VkRHIFence::createNonOwning(rhiDevice,
            *static_cast<VkFence*>(frame.inFlightFence));
        rhi::SubmitDesc sd;
        sd.commandBuffer = &cmd;
        sd.waitSemaphore = frame.imageAvailable.get();
        sd.signalSemaphore = frame.renderFinished.get();
        sd.signalFence = sigFence.get();
        rhiDevice.submit(sd);
        pSwapchain->present(frame);

        sigFence->wait();
        ++frameIdx;
    }

    rhiDevice.waitIdle();
    return 0;
}
