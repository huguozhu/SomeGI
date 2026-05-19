#pragma once
#include "core/image.h"
#include "core/buffer.h"
#include "scene/env_loader.h"
#include <vector>

// IBL（Image-Based Lighting）烘焙器。
//
// 输入：CPU 上的 HDR equirectangular（球面经纬投影）贴图。
// 输出：四张 GPU image，给 IBLTechnique 在运行时采样：
//   1) envCube      —— equirect → 六面 cubemap（含 mip 链，供 specular
//                      prefilter 多 LOD 取样）。
//   2) diffuseCube  —— 漫反射辐照度立方贴图：对 envCube 做 cosine-weighted
//                      半球积分（兰伯特卷积），运行时直接 SampleLevel 0 即得。
//   3) specularCube —— GGX prefilter 镜面立方贴图：每个 mip 对应一个
//                      roughness（mip 0 = 0，mip N-1 = 1），运行时按粗糙度
//                      选 mip。是 split-sum 近似的第一项。
//   4) brdfLut      —— 二维 split-sum BRDF 积分表（NoV × roughness → A,B 系数），
//                      split-sum 近似的第二项；与材质无关，只烘一次。
//
// 整个 bake 过程纯 compute，分多阶段 dispatch；所有中间 layout 转换都在
// 内部一次性 oneShotSubmit 完成。bake 结束后四张 image 都已转到
// SHADER_READ_ONLY_OPTIMAL，调用方可以直接挂到描述符上。
//
// 数据流（stage 顺序）：
//   equirect (HDR) →[equi_to_cube]→ envCube mip0
//   envCube mip0 →[blit chain]→ envCube mip 1..N
//   envCube →[prefilter_diffuse]→ diffuseCube
//   envCube →[prefilter_specular per-mip]→ specularCube mip 0..N
//                                       →[brdf_lut]→ brdfLut
//
// 所有 compute kernel 的 spv 由 build/shaders/gi/ibl/*.spv 提供（见
// shaders/gi/ibl/）。

namespace somegi {

class Device;

// IBL 烘焙的 GPU 资源集合。归调用方所有，IBLTechnique 借用而不拥有。
struct IblResources {
    Image envCube;        // 512² × 6 面, 6 mip, R16G16B16A16_SFLOAT —— 给 prefilter 阶段当源采样
    Image diffuseCube;    // 32²  × 6 面, 1 mip, R16G16B16A16_SFLOAT —— 漫反射辐照度
    Image specularCube;   // 256² × 6 面, 6 mip, R16G16B16A16_SFLOAT —— 镜面 prefilter（mip = roughness）
    Image brdfLut;        // 256² × 1, 1 mip, R16G16_SFLOAT —— split-sum 第二项的二维表
    VkSampler linear = VK_NULL_HANDLE;   // 共享的 LINEAR + CLAMP_TO_EDGE sampler
    uint32_t specularMipCount = 6;       // 与 specularCube.mipLevels() 一致；shader 端按 roughness 选 mip 用

    void destroy(Device& d);
};

class IblBaker {
public:
    // 输入：env（CPU 端的 HDR equirect 像素 + 尺寸）；
    // 输出：out 填好 4 张 image + sampler，layout 全部已转到 SHADER_READ_ONLY。
    // 失败抛 std::runtime_error。
    void bake(Device& d, VkCommandPool pool, const EnvCpu& env, IblResources& out);
};

}
