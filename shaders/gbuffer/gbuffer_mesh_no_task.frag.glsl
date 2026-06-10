#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

// ── Per-vertex inputs (from mesh shader) ──────────────────────
layout(location=0) in vec3 inWorldPos;
layout(location=1) in vec3 inNormal;
layout(location=2) in vec4 inTangent;
layout(location=3) in vec2 inUV;
layout(location=4) flat in int inMatIndex;

// ── Fragment bindings ─────────────────────────────────────────
// SSBO std430 布局，需与 C++ MaterialGpu 精确对齐
struct MaterialGpu {
    vec4 baseColorFactor;     // offset 0
    vec3 emissiveFactor;      // offset 16
    float metallicFactor;     // offset 28
    float roughnessFactor;    // offset 32
    float normalScale;        // offset 36
    float occlusionStrength;  // offset 40
    float alphaCutoff;        // offset 44
    int baseColorTex;         // offset 48
    int mrTex;                // offset 52
    int normalTex;            // offset 56
    int occlusionTex;         // offset 60
    int emissiveTex;          // offset 64
    uint alphaMode;           // offset 68
    uint doubleSided;         // offset 72
    uint _pad0;               // offset 76
};
layout(scalar, set=0, binding=9) readonly buffer MatBuf { MaterialGpu materials[]; };
layout(set=0, binding=10) uniform sampler gLinear;
layout(set=0, binding=11) uniform texture2D gTextures[];

// ── MRT outputs ───────────────────────────────────────────────
layout(location=0) out vec4 outRT0;  // baseColor.rgb, metallic
layout(location=1) out vec4 outRT1;  // normal.xyz, roughness
layout(location=2) out vec4 outRT2;  // emissive.rgb, ao

vec4 sampleTex(int idx, vec2 uv) {
    if (idx < 0) return vec4(1,1,1,1);
    return texture(sampler2D(gTextures[idx], gLinear), uv);
}

void main() {
    MaterialGpu m = materials[inMatIndex];

    // ── Alpha test ──
    vec4 baseTex = sampleTex(m.baseColorTex, inUV);
    vec4 base = m.baseColorFactor * baseTex;
    if (m.alphaMode == 1u && base.a < m.alphaCutoff) discard;

    // ── Metallic / Roughness ──
    float metallic  = m.metallicFactor;
    float roughness = m.roughnessFactor;
    if (m.mrTex >= 0) {
        vec3 t = sampleTex(m.mrTex, inUV).rgb;
        roughness *= t.g;
        metallic  *= t.b;
    }
    metallic  = clamp(metallic, 0.0, 1.0);
    roughness = clamp(roughness, 0.0, 1.0);

    // ── Normal mapping ──
    vec3 N = normalize(inNormal);
    if (m.normalTex >= 0) {
        vec3 nm = sampleTex(m.normalTex, inUV).rgb * 2.0 - 1.0;
        nm.xy *= m.normalScale;
        vec3 T = normalize(inTangent.xyz);
        vec3 B = normalize(cross(N, T) * inTangent.w);
        N = normalize(T * nm.x + B * nm.y + N * nm.z);
    }

    // ── AO ──
    float ao = 1.0;
    if (m.occlusionTex >= 0) {
        float t = sampleTex(m.occlusionTex, inUV).r;
        ao = mix(1.0, t, m.occlusionStrength);
    }

    // ── Emissive ──
    vec3 emi = m.emissiveFactor;
    if (m.emissiveTex >= 0) emi *= sampleTex(m.emissiveTex, inUV).rgb;

    outRT0 = vec4(base.rgb, metallic);
    outRT1 = vec4(N, roughness);
    outRT2 = vec4(clamp(emi, 0.0, 1.0), ao);
}
