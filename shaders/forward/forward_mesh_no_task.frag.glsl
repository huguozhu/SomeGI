#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

layout(location=0) in vec3 inWorldPos;
layout(location=1) in vec3 inNormal;
layout(location=2) in vec4 inTangent;
layout(location=3) in vec2 inUV;
layout(location=4) flat in int inMatIndex;

// set=0 binding 6: FrameUniforms
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

struct MaterialGpu {
    vec4 baseColorFactor; vec3 emissiveFactor; float metallicFactor;
    float roughnessFactor, normalScale, occlusionStrength, alphaCutoff;
    int baseColorTex, mrTex, normalTex, occlusionTex, emissiveTex;
    uint alphaMode, doubleSided, _pad0;
};
layout(scalar, set=0, binding=9) readonly buffer MatBuf { MaterialGpu materials[]; };
layout(set=0, binding=10) uniform sampler gLinear;
layout(set=0, binding=11) uniform texture2D gTextures[];

// set=1: IBL
layout(set=1, binding=0) uniform textureCube gIblDiffuse;
layout(set=1, binding=1) uniform textureCube gIblSpecular;
layout(set=1, binding=2) uniform texture2D   gIblBrdfLut;
layout(set=1, binding=3) uniform sampler     gIblSampler;
layout(set=1, binding=4, std140) uniform IblParams { float intensity; } gIblParams;

layout(location=0) out vec4 outColor;

vec4 sampleTex(int idx, vec2 uv) {
    if (idx < 0) return vec4(1,1,1,1);
    return texture(sampler2D(gTextures[idx], gLinear), uv);
}

vec3 fresnelSchlickRoughness(float cosT, vec3 F0, float r) {
    return F0 + (max(vec3(1.0 - r), F0) - F0) * pow(clamp(1.0 - cosT, 0.0, 1.0), 5.0);
}

vec3 evalIBL(vec3 N, vec3 V, vec3 albedo, float metallic, float roughness, float specMips) {
    float NoV = clamp(dot(N, V), 0.0, 1.0);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlickRoughness(NoV, F0, roughness);
    vec3 kd = (1.0 - F) * (1.0 - metallic);
    vec3 irradiance = texture(samplerCube(gIblDiffuse, gIblSampler), N).rgb;
    vec3 diffuse = kd * irradiance * albedo * gIblParams.intensity;
    vec3 R = reflect(-V, N);
    float mip = roughness * (specMips - 1.0);
    vec3 prefiltered = textureLod(samplerCube(gIblSpecular, gIblSampler), R, mip).rgb;
    vec2 ab = texture(sampler2D(gIblBrdfLut, gIblSampler), vec2(NoV, roughness)).xy;
    vec3 specular = prefiltered * (F * ab.x + ab.y) * gIblParams.intensity;
    return diffuse + specular;
}

void main() {
    MaterialGpu m = materials[inMatIndex];

    vec4 baseTex = sampleTex(m.baseColorTex, inUV);
    vec4 base = m.baseColorFactor * baseTex;
    if (m.alphaMode == 1u && base.a < m.alphaCutoff) discard;

    float metallic = m.metallicFactor, roughness = m.roughnessFactor;
    if (m.mrTex >= 0) { vec3 t = sampleTex(m.mrTex, inUV).rgb; roughness *= t.g; metallic *= t.b; }
    metallic = clamp(metallic, 0.0, 1.0); roughness = clamp(roughness, 0.0, 1.0);

    vec3 N = normalize(inNormal);
    if (m.normalTex >= 0) {
        vec3 nm = sampleTex(m.normalTex, inUV).rgb * 2.0 - 1.0; nm.xy *= m.normalScale;
        vec3 T = normalize(inTangent.xyz); vec3 B = normalize(cross(N, T) * inTangent.w);
        N = normalize(T * nm.x + B * nm.y + N * nm.z);
    }

    vec3 V = normalize(gFrame.cameraPos.xyz - inWorldPos);
    vec3 L = normalize(-gFrame.sunDir.xyz);
    vec3 sun = vec3(0);
    {
        vec3 H = normalize(V + L);
        float NoV = clamp(dot(N, V), 0.0, 1.0), NoL = clamp(dot(N, L), 0.0, 1.0);
        float NoH = clamp(dot(N, H), 0.0, 1.0), VoH = clamp(dot(V, H), 0.0, 1.0);
        vec3 F0 = mix(vec3(0.04), base.rgb, metallic);
        vec3 F = fresnelSchlickRoughness(VoH, F0, roughness);
        float D = roughness * roughness / (3.14159 * pow(NoH * NoH * (roughness * roughness - 1.0) + 1.0, 2.0));
        float G = 0.25 / mix(NoV * (1.0 - roughness) + roughness, NoV, 1.0) *
                          mix(NoL * (1.0 - roughness) + roughness, NoL, 1.0);
        sun = ((1.0 - F) * (1.0 - metallic) * base.rgb / 3.14159 + F * D * G) * NoL;
        sun *= gFrame.sunColor_intensity.rgb * gFrame.sunColor_intensity.w;
    }

    float occ = 1.0;
    if (m.occlusionTex >= 0) { float t = sampleTex(m.occlusionTex, inUV).r; occ = mix(1.0, t, m.occlusionStrength); }

    vec3 indirect = evalIBL(N, V, base.rgb, metallic, roughness, float(gFrame.counts.y));
    indirect *= occ;

    vec3 emi = m.emissiveFactor;
    if (m.emissiveTex >= 0) emi *= sampleTex(m.emissiveTex, inUV).rgb;

    outColor = vec4(sun + indirect + emi, base.a);
}
