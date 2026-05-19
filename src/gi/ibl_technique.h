#pragma once
#include "gi_technique.h"
#include "ibl_baker.h"
#include "core/buffer.h"

// IBLTechnique —— IBL（Image-Based Lighting）的运行时封装。
//
// 与 IblBaker 的关系：IblBaker 一次性烘焙四张 IBL 资源（envCube /
// diffuseCube / specularCube / brdfLut）；那些资源由 App 长期持有
// （App::m_envIbl，伴随整个进程生命周期）。IBLTechnique 通过 onAttach
// 时传入的 GIContext::iblBaked 借用它们，自己只负责：
//   1) 创建 set=1 的 descriptor set layout 和 descriptor set
//      （binding 0..3 = 三个 cube 贴图 + sampler，binding 4 = 自己的
//      params UBO 携带 intensity）。
//   2) 在 drawUI 里画"IBL intensity"滑条，slider 改变时把新值写入
//      paramsUbo（host-coherent）。
//   3) onDetach 释放 layout / pool / paramsUbo（不释放 IblResources，
//      它是借用的）。
//
// 多 GI 切换时（None ↔ IBL），App 销毁旧 m_giTech、构造新的；
// IblResources 不变，所以切换很轻量。

namespace somegi {

class IBLTechnique : public IGITechnique {
public:
    IBLTechnique() = default;
    ~IBLTechnique() override;

    const char* name() const override { return "IBL"; }
    const char* shaderVariant() const override { return "ibl"; }

    void onAttach(const GIContext& ctx) override;
    void onDetach() override;

    VkDescriptorSetLayout descriptorSetLayout() const override { return m_dsl; }
    VkDescriptorSet descriptorSet() const override { return m_set; }

    void drawUI() override;

    // 给 App / lighting 端用：specular mip count 通过 FrameUBO.counts.y
    // 注入 shader（evalIBLSpecular 按 roughness×(mips-1) 选 mip）。
    uint32_t specularMipCount() const { return m_res ? m_res->specularMipCount : 0; }

private:
    void buildSet();      // 把所有 descriptor 写入 m_set
    void writeParams();   // memcpy 当前 m_intensity 到 m_paramsUbo（host-coherent）

    Device* m_device = nullptr;
    const IblResources* m_res = nullptr;   // 借用，归 App 所有

    VkDescriptorSetLayout m_dsl = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    // set=1 binding 4 的 UBO，结构体为 { float intensity; float pad×3 }
    // （std140 16B 对齐）。slider 改值时通过 mapped 内存即时写入。
    Buffer m_paramsUbo;

    // ImGui 滑条状态。intensity ∈ [0, 4]：0=完全关闭 IBL（剩直接光），
    // 1=正常，>1=放大 IBL 用于调试。
    float m_intensity = 1.0f;
};

}
