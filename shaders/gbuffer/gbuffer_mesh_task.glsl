#version 460
#extension GL_EXT_mesh_shader : require

// ── Task Shader ───────────────────────────────────────────────
// 与 mesh+frag 共享 set=0 bindings 0-11

struct DrawData {
    mat4 modelMatrix;
    uint materialIndex, firstIndex, indexCount, vertexOffset;
    vec3 aabbMin; uint _pad0; vec3 aabbMax; uint _pad1;
};

struct TaskPayload { uint drawIndex; };
taskPayloadSharedEXT TaskPayload pld;

layout(set=0, binding=0) readonly buffer DrawBuf { DrawData draws[]; } gDrawData;
layout(set=0, binding=1, std140) uniform CullUbo {
    mat4 viewProj;
    vec4 frustum[6];
    vec2 screenSize;
    uint drawCount;
    uint hizMaxMip;
} gCull;

layout(local_size_x=64) in;

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= gCull.drawCount) return;
    // Phase 1：直通所有 draw，不做 cull（先验证 task+mesh+frag 管线）
    pld.drawIndex = tid;
    EmitMeshTasksEXT(1u, 1u, 1u);
}
