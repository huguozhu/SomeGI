#include "lpv_grid.h"
#include "core/device.h"

namespace somegi {

namespace {
Image makeLpvImage(Device& d, uint32_t resolution) {
    ImageDesc desc{};
    desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;   // 4 SH 系数 / cell
    desc.extent = {resolution, resolution, resolution};
    desc.type   = VK_IMAGE_TYPE_3D;
    // STORAGE：inject / propagate 写；SAMPLED：lighting trilinear 读；
    // TRANSFER_DST：每帧 vkCmdClearColorImage 抹掉残留。
    desc.usage  = VK_IMAGE_USAGE_STORAGE_BIT
                | VK_IMAGE_USAGE_SAMPLED_BIT
                | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return Image(d, desc);
}
}

void LpvGrid::create(Device& d, uint32_t resolution) {
    lpvR = makeLpvImage(d, resolution);
    lpvG = makeLpvImage(d, resolution);
    lpvB = makeLpvImage(d, resolution);
}

void LpvGrid::destroy() {
    lpvR.reset();
    lpvG.reset();
    lpvB.reset();
}

void LpvResources::create(Device& d, uint32_t resolution) {
    m_resolution = resolution;
    m_grids[0].create(d, resolution);
    m_grids[1].create(d, resolution);
    m_gv = makeLpvImage(d, resolution);   // 与 LPV 同形 RGBA16F 32³
    m_curIdx = 0;
}

void LpvResources::destroy() {
    m_grids[0].destroy();
    m_grids[1].destroy();
    m_gv.reset();
    m_resolution = 0;
    m_curIdx = 0;
}

}
