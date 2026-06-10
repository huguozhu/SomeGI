#version 460
#extension GL_EXT_mesh_shader : require

// ── Task Shader：frustum culling + 大 draw 拆分 ─────────────────
struct DrawData {
    mat4 modelMatrix; uint materialIndex; int firstIndex; uint indexCount; int vertexOffset;
    float aabbMinX, aabbMinY, aabbMinZ; uint _pad0; float aabbMaxX, aabbMaxY, aabbMaxZ; uint _pad1;
};
layout(set=0, binding=0) readonly buffer DrawBuf { DrawData draws[]; } gDrawData;

struct CullUbo { mat4 viewProj; vec4 frustum[6]; vec2 screenSize; uint drawCount; uint hizMaxMip; };
layout(set=0, binding=1, std140) uniform CullBuf { CullUbo cull; };

#define MAX_TRIS 85
struct TaskPayload { uint drawIndex; uint triOffset; };
taskPayloadSharedEXT TaskPayload pld;

layout(local_size_x=64) in;

bool aabbInFrustum(vec3 mn, vec3 mx, vec4 f[6]) {
    for (int i = 0; i < 6; i++) {
        vec3 n = f[i].xyz; float d = f[i].w;
        vec3 p = vec3(n.x > 0 ? mn.x : mx.x, n.y > 0 ? mn.y : mx.y, n.z > 0 ? mn.z : mx.z);
        if (dot(n, p) + d < 0) return false;
    }
    return true;
}

void main() {
    // 诊断：仅 thread 0 发射 1 个 mesh task (draw 0, offset 0)
    if (gl_GlobalInvocationID.x != 0u) return;
    pld.drawIndex = 0u;
    pld.triOffset = 0u;
    EmitMeshTasksEXT(1u, 1u, 1u);
}
