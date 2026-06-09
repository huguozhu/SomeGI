// Hi-Z depth pyramid builder.
// Level 0 reads full-res depth, each subsequent level reduces 2x2→max depth.

[[vk::binding(0, 0)]] Texture2D<float> gDepthIn;

struct HiZMipOut { RWTexture2D<float> dst; };
[[vk::binding(1, 0)]] RWTexture2D<float> gHiZMip1;  // half-res
[[vk::binding(2, 0)]] RWTexture2D<float> gHiZMip2;  // quarter-res
[[vk::binding(3, 0)]] RWTexture2D<float> gHiZMip3;  // eighth-res
[[vk::binding(4, 0)]] RWTexture2D<float> gHiZMip4;  // 16th-res

struct PC { uint2 srcSize; };
[[vk::push_constant]] PC gPC;

[numthreads(16, 16, 1)] void cs_main(uint3 tid : SV_DispatchThreadID) {
    uint2 coord = tid.xy;
    // Level 0→1: full → half
    if (coord.x < gPC.srcSize.x / 2 && coord.y < gPC.srcSize.y / 2) {
        float d00 = gDepthIn[uint2(coord.x*2,   coord.y*2)];
        float d10 = gDepthIn[uint2(coord.x*2+1, coord.y*2)];
        float d01 = gDepthIn[uint2(coord.x*2,   coord.y*2+1)];
        float d11 = gDepthIn[uint2(coord.x*2+1, coord.y*2+1)];
        float m = max(max(d00, d10), max(d01, d11));
        gHiZMip1[coord] = m;
    }
    // Level 1→2
    if (coord.x < gPC.srcSize.x / 4 && coord.y < gPC.srcSize.y / 4) {
        float d00 = gHiZMip1[uint2(coord.x*2,   coord.y*2)];
        float d10 = gHiZMip1[uint2(coord.x*2+1, coord.y*2)];
        float d01 = gHiZMip1[uint2(coord.x*2,   coord.y*2+1)];
        float d11 = gHiZMip1[uint2(coord.x*2+1, coord.y*2+1)];
        gHiZMip2[coord] = max(max(d00, d10), max(d01, d11));
    }
    // Level 2→3
    if (coord.x < gPC.srcSize.x / 8 && coord.y < gPC.srcSize.y / 8) {
        float d00 = gHiZMip2[uint2(coord.x*2,   coord.y*2)];
        float d10 = gHiZMip2[uint2(coord.x*2+1, coord.y*2)];
        float d01 = gHiZMip2[uint2(coord.x*2,   coord.y*2+1)];
        float d11 = gHiZMip2[uint2(coord.x*2+1, coord.y*2+1)];
        gHiZMip3[coord] = max(max(d00, d10), max(d01, d11));
    }
    // Level 3→4
    if (coord.x < gPC.srcSize.x / 16 && coord.y < gPC.srcSize.y / 16) {
        float d00 = gHiZMip3[uint2(coord.x*2,   coord.y*2)];
        float d10 = gHiZMip3[uint2(coord.x*2+1, coord.y*2)];
        float d01 = gHiZMip3[uint2(coord.x*2,   coord.y*2+1)];
        float d11 = gHiZMip3[uint2(coord.x*2+1, coord.y*2+1)];
        gHiZMip4[coord] = max(max(d00, d10), max(d01, d11));
    }
}
