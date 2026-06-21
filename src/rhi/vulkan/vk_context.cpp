// rhi/vulkan/vk_context.cpp
#include "vk_context.h"
#include "vk_command.h"
#include "../base/fence.h"    // RHISemaphore
#include <core/vk_common.h>

namespace somegi {
namespace rhi {

VkContext::VkContext(VkRHIDevice& device, uint32_t framesInFlight)
    : m_device(device), m_framesInFlight(framesInFlight) {
    createResources();
}

VkContext::~VkContext() {
    destroyResources();
}

void VkContext::createResources() {
    // ── 命令池 ──
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = m_device.queueFamily();
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(m_device.vkDevice(), &pci, nullptr, &m_pool));

    // ── 命令缓冲区 + fence ──
    m_frames.resize(m_framesInFlight);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = m_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    for (uint32_t i = 0; i < m_framesInFlight; ++i) {
        VK_CHECK(vkAllocateCommandBuffers(m_device.vkDevice(), &ai, &m_frames[i].cmd));

        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(m_device.vkDevice(), &fci, nullptr, &m_frames[i].fence));
    }
}

void VkContext::destroyResources() {
    if (m_pool) {
        vkDestroyCommandPool(m_device.vkDevice(), m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
    }
    // fence 随 pool 销毁而释放；cmd buffer 同样
    for (auto& f : m_frames) {
        if (f.fence) vkDestroyFence(m_device.vkDevice(), f.fence, nullptr);
    }
    m_frames.clear();
}

RHICommandBuffer& VkContext::beginFrame(uint32_t frameIndex) {
    auto& fr = m_frames[frameIndex % m_framesInFlight];

    // 等待上一轮此 slot 的 GPU 工作完成
    if (!fr.fenceSignaled) {
        VK_CHECK(vkWaitForFences(m_device.vkDevice(), 1, &fr.fence, VK_TRUE, UINT64_MAX));
    }
    VK_CHECK(vkResetFences(m_device.vkDevice(), 1, &fr.fence));
    fr.fenceSignaled = false;

    // 重置 + 开始命令缓冲区
    VK_CHECK(vkResetCommandBuffer(fr.cmd, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(fr.cmd, &bi));

    // 返回 RHI 包装（非拥有型）
    auto* rhiCmd = new VkRHICommandBuffer(m_device, fr.cmd);
    // 注意：调用方需自行管理生命周期；通常用临时对象
    m_tempCmd.reset(rhiCmd);
    return *m_tempCmd;
}

void VkContext::endFrame(uint32_t frameIndex,
                              const RHISemaphore* waitSemaphore,
                              const RHISemaphore* signalSemaphore) {
    auto& fr = m_frames[frameIndex % m_framesInFlight];

    // 结束录制
    vkEndCommandBuffer(fr.cmd);

    // 提交
    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = fr.cmd;

    VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};

    uint32_t waitCount = 0;
    const VkSemaphoreSubmitInfo* pWait = nullptr;
    if (waitSemaphore) {
        waitInfo.semaphore = (VkSemaphore)(uintptr_t)waitSemaphore->nativeHandle();
        waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        waitCount = 1;
        pWait = &waitInfo;
    }

    uint32_t signalCount = 0;
    const VkSemaphoreSubmitInfo* pSignal = nullptr;
    if (signalSemaphore) {
        signalInfo.semaphore = (VkSemaphore)(uintptr_t)signalSemaphore->nativeHandle();
        signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
        signalCount = 1;
        pSignal = &signalInfo;
    }

    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.waitSemaphoreInfoCount = waitCount;
    si.pWaitSemaphoreInfos = pWait;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &csi;
    si.signalSemaphoreInfoCount = signalCount;
    si.pSignalSemaphoreInfos = pSignal;

    VK_CHECK(vkQueueSubmit2(m_device.vkQueue(), 1, &si, fr.fence));
}

RHICommandBuffer& VkContext::commandBuffer(uint32_t frameIndex) {
    auto& fr = m_frames[frameIndex % m_framesInFlight];
    m_tempCmd.reset(new VkRHICommandBuffer(m_device, fr.cmd));
    return *m_tempCmd;
}

void VkContext::waitIdle() {
    vkDeviceWaitIdle(m_device.vkDevice());
}

VkCommandBuffer VkContext::vkCommandBuffer(uint32_t frameIndex) const {
    return m_frames[frameIndex % m_framesInFlight].cmd;
}

void VkContext::endFrame(uint32_t frameIndex, VkSemaphore waitSem, VkSemaphore signalSem) {
    auto& fr = m_frames[frameIndex % m_framesInFlight];

    vkEndCommandBuffer(fr.cmd);

    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = fr.cmd;

    VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    uint32_t waitCount = 0;
    const VkSemaphoreSubmitInfo* pWait = nullptr;
    if (waitSem != VK_NULL_HANDLE) {
        waitInfo.semaphore = waitSem;
        waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        waitCount = 1;
        pWait = &waitInfo;
    }

    VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    uint32_t signalCount = 0;
    const VkSemaphoreSubmitInfo* pSignal = nullptr;
    if (signalSem != VK_NULL_HANDLE) {
        signalInfo.semaphore = signalSem;
        signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
        signalCount = 1;
        pSignal = &signalInfo;
    }

    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.waitSemaphoreInfoCount = waitCount;
    si.pWaitSemaphoreInfos = pWait;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &csi;
    si.signalSemaphoreInfoCount = signalCount;
    si.pSignalSemaphoreInfos = pSignal;

    VK_CHECK(vkQueueSubmit2(m_device.vkQueue(), 1, &si, fr.fence));
}

} // namespace rhi
} // namespace somegi
