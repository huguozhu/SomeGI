#include "nanite_streaming.h"
#include <algorithm>
#include <cstring>

namespace somegi::nanite {

NaniteStreamer::NaniteStreamer() = default;

NaniteStreamer::~NaniteStreamer() = default;

void NaniteStreamer::init(uint32_t maxPages) {
    m_pages.resize(maxPages);
    for (uint32_t i = 0; i < maxPages; ++i) {
        m_pages[i].state = 0; // free
        m_freeSlots.push_back(i);
    }
    m_gpuHeapSize = maxPages * kPageSize;
}

void NaniteStreamer::registerPage(uint32_t pageId, uint32_t clusterOffset, uint32_t clusterCount) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_freeSlots.empty()) {
        // Need to evict first
        return;
    }

    uint32_t slot = m_freeSlots.back();
    m_freeSlots.pop_back();

    PageEntry& entry = m_pages[slot];
    entry.pageId = pageId;
    entry.clusterOffset = clusterOffset;
    entry.clusterCount = clusterCount;
    entry.gpuOffset = slot * kPageSize;
    entry.state = 1; // loading
    entry.lastAccessFrame = 0;
    m_loadingCount++;
}

void NaniteStreamer::requestPages(const std::vector<uint32_t>& pageIds, uint32_t frameIndex) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto pageId : pageIds) {
        for (auto& p : m_pages) {
            if (p.pageId == pageId && p.state == 2) { // resident
                p.lastAccessFrame = frameIndex;
                break;
            }
        }
    }
}

void NaniteStreamer::update(uint32_t frameIndex) {
    // Check for completed async loads
    // In a real implementation, this would check a fence or flag
    // For now, just mark all loading pages as resident
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& p : m_pages) {
        if (p.state == 1) { // loading → resident
            p.state = 2;
            p.lastAccessFrame = frameIndex;
            m_loadingCount--;
            m_residentCount++;
        }
    }
}

void NaniteStreamer::evictIfNeeded(uint32_t frameIndex, uint32_t maxResidentPages) {
    std::lock_guard<std::mutex> lock(m_mutex);

    while (m_residentCount > maxResidentPages) {
        // Find LRU resident page
        uint32_t lruIdx = UINT32_MAX;
        uint32_t lruFrame = UINT32_MAX;

        for (uint32_t i = 0; i < m_pages.size(); ++i) {
            if (m_pages[i].state == 2 && m_pages[i].lastAccessFrame < lruFrame) {
                lruFrame = m_pages[i].lastAccessFrame;
                lruIdx = i;
            }
        }

        if (lruIdx == UINT32_MAX) break;

        m_pages[lruIdx].state = 0; // free
        m_freeSlots.push_back(lruIdx);
        m_residentCount--;
    }
}

bool NaniteStreamer::uploadPage(uint32_t pageId, const void* data, size_t size) {
    std::lock_guard<std::mutex> lock(m_mutex);

    for (auto& p : m_pages) {
        if (p.pageId == pageId && p.state == 1) { // loading
            // Mark as resident immediately (synchronous upload)
            p.state = 2;
            m_loadingCount--;
            m_residentCount++;
            return true;
        }
    }
    return false;
}

std::vector<NaniteStreamer::UploadCmd> NaniteStreamer::flushUploads() {
    // Return list of pending upload commands
    // In real impl, this would generate vkCmdCopyBuffer commands
    return {};
}

} // namespace somegi::nanite
