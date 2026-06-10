#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : require

// ── Binding 0: DrawData SSBO ──────────────────────────────────
// 必须与 C++ DrawEntry (112 bytes) 精确对齐
struct DrawData {
    mat4 modelMatrix;     // offset 0,  64 bytes
    uint materialIndex;   // offset 64, 4 bytes
    int  firstIndex;      // offset 68, 4 bytes
    uint indexCount;      // offset 72, 4 bytes
    int  vertexOffset;    // offset 76, 4 bytes
    float aabbMinX, aabbMinY, aabbMinZ;  // offset 80, 12 bytes
    uint _pad0;           // offset 92, 4 bytes
    float aabbMaxX, aabbMaxY, aabbMaxZ;  // offset 96, 12 bytes
    uint _pad1;           // offset 108, 4 bytes
    // Total: 112 bytes
};
layout(set=0, binding=0) readonly buffer DrawBuf { DrawData draws[]; } gDrawData;

// ── Binding 6: FrameUniforms UBO ──────────────────────────────
layout(set=0, binding=6, std140) uniform FrameUBO {
    mat4 view, proj, viewProj, invViewProj, prevViewProj;
    vec4 cameraPos, sunDir, sunColor_intensity, ambient;
    ivec4 counts, lpvCounts; vec4 lpvGridMinCell;
    ivec4 vxgiCounts; vec4 vxgiGridMinCell;
    ivec4 prtCounts; vec4 prtGridMinCell;
    vec4 prtLightSH_R, prtLightSH_G, prtLightSH_B;
    vec4 prtLightSH9_R0, prtLightSH9_R1, prtLightSH9_G0, prtLightSH9_G1, prtLightSH9_B0, prtLightSH9_B1;
    vec4 prtLightSH16_R0, prtLightSH16_R1, prtLightSH16_G0, prtLightSH16_G1, prtLightSH16_B0, prtLightSH16_B1;
    ivec4 ddgiCounts; vec4 ddgiOrigin, ddgiSpacing; ivec4 ddgiOctaSizes, lumenCounts;
} gFrame;

// ── Binding 7-8: Vertex/Index SSBO ────────────────────────────
layout(set=0, binding=7) readonly buffer VertBuf { float verts[]; };
layout(set=0, binding=8) readonly buffer IdxBuf  { uint  indices[]; };

#define MAX_VERTS 256
#define MAX_TRIS  85
#define VERTEX_FLOATS 12   // sizeof(Vertex)=48 bytes = 12 floats packed

// ── Task Payload（Task Shader 设置 drawIndex + triOffset） ──
struct TaskPayload { uint drawIndex; uint triOffset; };
taskPayloadSharedEXT TaskPayload pld;

// ── Per-vertex outputs (passed to fragment shader) ─────────────
layout(location=0) out vec3 outWorldPos[];
layout(location=1) out vec3 outNormal[];
layout(location=2) out vec4 outTangent[];
layout(location=3) out vec2 outUV[];
layout(location=4) flat out int outMatIndex[];

layout(local_size_x=64) in;
layout(triangles) out;
layout(max_vertices=MAX_VERTS, max_primitives=MAX_TRIS) out;

void main() {
    uint drawIndex = pld.drawIndex;
    uint triOffset = pld.triOffset;
    DrawData dd = gDrawData.draws[drawIndex];
    uint totalTris = dd.indexCount / 3u;
    uint triCount = min(totalTris - triOffset, MAX_TRIS);
    uint vtxCount = min(triCount * 3u, MAX_VERTS);
    mat3 n3 = mat3(dd.modelMatrix);

    uint actualVerts = vtxCount;
    uint actualPrims = triCount;
    if (gl_LocalInvocationID.x == 0u) SetMeshOutputsEXT(actualVerts, actualPrims);

    // 64 线程 × 多轮循环，覆盖最多 MAX_TRIS 个三角形
    for (uint block = 0u; block < MAX_TRIS; block += 64u) {
        uint i = block + gl_LocalInvocationID.x;
        if (i >= MAX_TRIS) break;
        if (i < triCount) {
            uint baseIdx = dd.firstIndex + (triOffset + i) * 3u;
            uint i0 = indices[baseIdx + 0u];
            uint i1 = indices[baseIdx + 1u];
            uint i2 = indices[baseIdx + 2u];

            for (uint v = 0u; v < 3u; v++) {
                uint vi = (v == 0u) ? i0 : ((v == 1u) ? i1 : i2);
                uint vo = (dd.vertexOffset + vi) * VERTEX_FLOATS;
                vec3 pos = vec3(verts[vo+0u], verts[vo+1u], verts[vo+2u]);
                vec3 nrm = vec3(verts[vo+3u], verts[vo+4u], verts[vo+5u]);
                vec4 tan = vec4(verts[vo+6u], verts[vo+7u], verts[vo+8u], verts[vo+9u]);
                vec2 uv  = vec2(verts[vo+10u], verts[vo+11u]);
                vec4 wp = dd.modelMatrix * vec4(pos, 1.0);
                uint oi = i * 3u + v;
                gl_MeshVerticesEXT[oi].gl_Position = gFrame.viewProj * wp;
                outWorldPos[oi]  = wp.xyz;
                outNormal[oi]    = normalize(n3 * nrm);
                outTangent[oi]   = vec4(normalize(n3 * tan.xyz), tan.w);
                outUV[oi]        = uv;
                outMatIndex[oi]  = int(dd.materialIndex);
            }
            gl_PrimitiveTriangleIndicesEXT[i] = uvec3(i*3u+0u, i*3u+1u, i*3u+2u);
        } else {
            gl_MeshVerticesEXT[i*3u+0u].gl_Position = vec4(0,0,0,0);
            gl_MeshVerticesEXT[i*3u+1u].gl_Position = vec4(0,0,0,0);
            gl_MeshVerticesEXT[i*3u+2u].gl_Position = vec4(0,0,0,0);
            outWorldPos[i*3u+0u] = vec3(0); outNormal[i*3u+0u] = vec3(0,0,1);
            outTangent[i*3u+0u]  = vec4(1,0,0,1); outUV[i*3u+0u] = vec2(0); outMatIndex[i*3u+0u] = 0;
            gl_PrimitiveTriangleIndicesEXT[i] = uvec3(i*3u+0u, i*3u+1u, i*3u+2u);
        }
    }
}
