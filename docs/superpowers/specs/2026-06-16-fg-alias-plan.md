# FrameGraph 资源别名复用 — 分析结论

日期：2026-06-16
状态：已分析，暂缓
分支：frame_graph

---

## 结论

别名分析算法已修复工作正常，但当前 pass 结构**无别名复用机会**。

## 原因

所有 7 个候选资源（ssao/ssr/ssgi/rtGI/restir/rsmGI/lumenGI）都在 Lighting pass 被读取，
它们的生命周期全部重叠：

```
ssao:     write=12  read=14
ssr:      write=13  read=14
ssgi:     write=6   read=14
rtGI:     write=7   read=14
restir:   write=8   read=14
lumenGI:  write=9   read=14
rsmGI:    write=10  read=14
          ↑ 全部在 [12,14] 区间重叠 ↑
```

`computeAliasing` 正确判断出零 aliasing 机会。编译器输出无 alias group 是**正确行为**。

## 决策

不为了启用 aliasing 而拆分 Lighting pass（用 GPU 性能换显存是本末倒置）。
别名分析功能就绪，未来引入真正短寿命的临时纹理（denoiser intermediate、depth pyramid 等）时自动生效。

## 后续

当引入 `createTexture()` 托管资源时，实施方案 A（回写 RenderTargets），详见下方。

---

## 方案 A：回写 RenderTargets（后续需要时实施）

1. `FGResources` 增加 `getImage(FGHandle)` 方法
2. `FrameGraph::populateViewCache()` 为托管 texture 缓存 `VkImage`
3. `execute()` 后 swap 托管 texture 的 `Image` 与 `RenderTargets` 中的对应成员
4. `setupFrameGraph()` 将帧内临时资源从 `importTexture()` 改为 `createTexture()`
5. record 函数无需改动（继续通过 `rt.xxx.image()` 访问）

## 方案 B：Record 函数接收 FGResources（长期方向）

1. 逐个 pass 修改 record 函数签名：`record(cmd, rt)` → `record(cmd, resources)`
2. Pass 内部通过 `resources.getTextureView(handle)` 获取资源视图
3. 删除方案 A 的 RenderTargets 回写逻辑
