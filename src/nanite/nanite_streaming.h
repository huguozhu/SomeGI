#pragma once
#include "nanite_types.h"
#include "core/buffer.h"
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>

namespace somegi::nanite {

// ---- Page table --------------------------------------------------------
static constexpr uint32_t kPageSize = 65536; // 64 KB per page
static constexpr uint32_t kMaxPages   = 4096; // 256 MB total budget

struct PageEntry {
    uint32_t pageId;           // which cluster range this page covers
    uint32_t clusterOffset;    // first cluster in this page
    uint32_t clusterCount;     // number of clusters in this page
    uint32_t gpuOffset;        // byte offset into the GPU cluster buffer
    uint32_t state;            // 0=free, 1=loading, 2=resident, 3=evicting
    uint32_t lastAccessFrame;  // for LRU eviction
    uint32_t _pad0, _pad1;
};

// ---- Streaming system ---------------------------------------------------
class NaniteStreamer {
public:
    using LoadCallback = std::function<bool(uint32_t pageId, const void*& data, size_t& size)>;

    NaniteStreamer();
    ~NaniteStreamer();

    // Initialize with a page table buffer (GPU-visible)
    void init(uint32_t maxPages = kMaxPages);

    // Register a page for streaming
    void registerPage(uint32_t pageId, uint32_t clusterOffset, uint32_t clusterCount);

    // Mark pages as needed (called after GPU cull pass)
    void requestPages(const std::vector<uint32_t>& pageIds, uint32_t frameIndex);

    // Process pending loads (call each frame on CPU)
    void update(uint32_t frameIndex);

    // Evict least-recently-used pages to stay under budget
    void evictIfNeeded(uint32_t frameIndex, uint32_t maxResidentPages);

    // Access state
    uint32_t residentPages() const { return m_residentCount; }
    uint32_t loadingPages()  const { return m_loadingCount; }
    const std::vector<PageEntry>& pages() const { return m_pages; }

    // Upload completed page data to GPU
    bool uploadPage(uint32_t pageId, const void* data, size_t size);

    // Get upload commands that need to be submitted
    struct UploadCmd {
        VkBuffer src;
        VkBuffer dst;
        VkDeviceSize size;
        VkDeviceSize dstOffset;
    };
    std::vector<UploadCmd> flushUploads();

private:
    std::vector<PageEntry> m_pages;
    std::vector<uint32_t>  m_freeSlots;
    std::atomic<uint32_t>  m_residentCount{0};
    std::atomic<uint32_t>  m_loadingCount{0};
    std::mutex m_mutex;

    // GPU buffer that holds all cluster data
    struct GpuAlloc {
        uint32_t offset; // byte offset in the big buffer
        uint32_t size;
    };
    std::vector<GpuAlloc> m_gpuAllocs;
    uint32_t m_gpuHeapSize = 0;
};

} // namespace somegi::nanite
