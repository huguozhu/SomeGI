// Nanite Phase 3 — per-cluster frustum culling

import shared_types;

// Vulkan indirect draw command (not available as built-in in Slang)
struct DrawCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int  vertexOffset;
    uint firstInstance;
};

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] StructuredBuffer<ClusterHeader> gClusters;
[[vk::binding(2, 0)]] RWStructuredBuffer<DrawCommand> gDrawArgs;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> gDrawCount;

// ClusterHeader layout (matches nanite_types.h — 48 bytes)
struct ClusterHeader {
    float4 boundingSphere;
    uint   triangleOffset;
    uint   triangleCount;
    uint   vertexOffset;
    uint   vertexCount;
    uint   lodParentIndex;
    uint   lodError;
    uint   pad0;
};

struct CullPC {
    uint clusterCount;
    uint pad0, pad1, pad2;
};
[[vk::push_constant]] CullPC gPC;

bool isVisible(ClusterHeader c, float4x4 viewProj) {
    float4 cs = mul(viewProj, float4(c.boundingSphere.xyz, 1.0));
    float r = c.boundingSphere.w;
    float w = cs.w;

    if (cs.z + r < 0.0)  return false;
    if (cs.z - r > w)    return false;
    if (cs.x + r < -w)   return false;
    if (cs.x - r >  w)   return false;
    if (cs.y + r < -w)   return false;
    if (cs.y - r >  w)   return false;

    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint ci = dt.x;
    if (ci >= gPC.clusterCount) return;

    ClusterHeader c = gClusters[ci];

    if (isVisible(c, gFrame.viewProj)) {
        uint slot;
        InterlockedAdd(gDrawCount[0], 1, slot);

        DrawCommand cmd;
        cmd.indexCount    = c.triangleCount * 3u;
        cmd.instanceCount = 1;
        cmd.firstIndex    = c.triangleOffset;
        cmd.vertexOffset  = (int)c.vertexOffset;
        cmd.firstInstance  = 0;
        gDrawArgs[slot] = cmd;
    }
}
