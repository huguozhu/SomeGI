// NDGI Probe Trace —— 对每个探针用 VK_ray_query 追踪光线，收集训练样本
//
// dispatch (probeCount * raysPerProbe / 64, 1, 1): 每 thread 一条光线
//
// 每样本写入 9 个 float: (worldPos.xyz, normal.xyz, radiance.rgb)
// 样本计数 atomicAdd

import shared_types;

// RT instance data（与 scene_rt_as.cpp 中的 GpuInstanceData 对齐）
struct RtInstanceData {
    int materialIndex;
    int vertexOffset;
    uint firstIndex;
    int _pad0;
};

// Vertex 结构（与 scene_gpu.h 中的 Vertex 对齐）
struct Vertex {
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uv0;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure gTLAS;
[[vk::binding(1, 0)]] StructuredBuffer<RtInstanceData> gInstances;
[[vk::binding(2, 0)]] StructuredBuffer<Vertex> gVertices;
[[vk::binding(3, 0)]] StructuredBuffer<uint> gIndices;
[[vk::binding(4, 0)]] StructuredBuffer<MaterialGpu> gMaterials;
[[vk::binding(5, 0)]] Texture2D gTextures[];
[[vk::binding(6, 0)]] SamplerState gLinearSampler;
[[vk::binding(7, 0)]] ConstantBuffer<FrameUniforms> gFrame;

[[vk::binding(8, 0)]] RWStructuredBuffer<float> gSampleBuf;      // 9 floats/sample
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> gSampleCount;     // atomic counter

struct ProbeTracePC {
    float3 ddgiOrigin;
    float  _pad0;
    float3 ddgiSpacing;
    float  _pad1;
    uint   probesX, probesY, probesZ;
    uint   raysPerProbe;
    float  randomRotation;
    uint   _pad2;
};
[[vk::push_constant]] ProbeTracePC gPC;

// Spherical Fibonacci
float3 sphericalFibonacci(uint i, uint N) {
    const float PHI = 1.6180339887498949;
    float phi = 6.2831853 * frac(float(i) * (PHI - 1.0));
    float cosTheta = 1.0 - (2.0 * float(i) + 1.0) / float(N);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

float3 rotateY(float3 v, float angle) {
    float c = cos(angle), s = sin(angle);
    return float3(c*v.x + s*v.z, v.y, -s*v.x + c*v.z);
}

float4 sampleTex(int idx, float2 uv) {
    if (idx < 0) return float4(1, 1, 1, 1);
    return gTextures[idx].SampleLevel(gLinearSampler, uv, 0);
}

// 简化的直接光照计算（在命中点）
float3 evalDirectLight(float3 P, float3 N, float3 albedo, float metallic, float roughness) {
    float3 V = normalize(gFrame.cameraPos.xyz - P);
    float3 L = normalize(-gFrame.sunDir.xyz);
    float NdotL = saturate(dot(N, L));
    if (NdotL <= 0.0) return float3(0);

    // GGX specular
    float3 H = normalize(V + L);
    float NdotH = saturate(dot(N, H));
    float NdotV = saturate(dot(N, V));
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (3.14159265 * denom * denom);

    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G = NdotV / (NdotV * (1.0 - k) + k) * NdotL / (NdotL * (1.0 - k) + k);

    float3 F0 = lerp(float3(0.04), albedo, metallic);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - NdotH, 5.0);
    float3 spec = D * G * F / max(4.0 * NdotV * NdotL, 0.001);

    float3 kd = (1.0 - F) * (1.0 - metallic);
    float3 diff = kd * albedo / 3.14159265;

    return (diff + spec) * gFrame.sunColor_intensity.rgb * gFrame.sunColor_intensity.w * NdotL;
}

// 从命中三角形插值顶点属性
float3 interpolateNormal(RtInstanceData inst, float2 bary, uint primId) {
    uint base = inst.firstIndex + primId * 3;
    uint i0 = gIndices[base], i1 = gIndices[base + 1], i2 = gIndices[base + 2];
    float3 n0 = gVertices[i0 + inst.vertexOffset].normal;
    float3 n1 = gVertices[i1 + inst.vertexOffset].normal;
    float3 n2 = gVertices[i2 + inst.vertexOffset].normal;
    float u = bary.x, v = bary.y;
    return normalize(n0 * (1.0 - u - v) + n1 * u + n2 * v);
}

float2 interpolateUV(RtInstanceData inst, float2 bary, uint primId) {
    uint base = inst.firstIndex + primId * 3;
    uint i0 = gIndices[base], i1 = gIndices[base + 1], i2 = gIndices[base + 2];
    float2 uv0 = gVertices[i0 + inst.vertexOffset].uv0;
    float2 uv1 = gVertices[i1 + inst.vertexOffset].uv0;
    float2 uv2 = gVertices[i2 + inst.vertexOffset].uv0;
    float u = bary.x, v = bary.y;
    return uv0 * (1.0 - u - v) + uv1 * u + uv2 * v;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint totalRays = gPC.probesX * gPC.probesY * gPC.probesZ * gPC.raysPerProbe;
    if (dt.x >= totalRays) return;

    uint probeIdx = dt.x / gPC.raysPerProbe;
    uint rayIdx   = dt.x % gPC.raysPerProbe;

    // 探针位置
    uint pz = probeIdx / (gPC.probesX * gPC.probesY);
    uint py = (probeIdx / gPC.probesX) % gPC.probesY;
    uint px = probeIdx % gPC.probesX;
    float3 probePos = gPC.ddgiOrigin
                    + float3(float(px), float(py), float(pz)) * gPC.ddgiSpacing;

    // 光线方向
    float3 dir = sphericalFibonacci(rayIdx, gPC.raysPerProbe);
    dir = rotateY(dir, gPC.randomRotation);

    // Ray query
    RayDesc ray;
    ray.Origin = probePos + dir * 0.01;  // 轻微偏移避免自交
    ray.Direction = dir;
    ray.TMin = 0.001;
    ray.TMax = 1e6;

    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(gTLAS, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    float3 radiance = float3(0);
    float3 worldPos = probePos;
    float3 normal = float3(0, 1, 0);

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        uint instId = q.CommittedInstanceID();
        RtInstanceData inst = gInstances[instId];
        uint primId = q.CommittedPrimitiveIndex();
        float2 bary = q.CommittedTriangleBarycentrics();
        float t = q.CommittedRayT();

        worldPos = probePos + dir * t;
        normal = interpolateNormal(inst, bary, primId);

        // 变换法线到世界空间 ... 简化：假设 identity transform
        // TODO: 正确做法是从 TLAS instance 中读取 transform matrix

        // 材质属性
        int matIdx = inst.materialIndex;
        MaterialGpu mat = gMaterials[matIdx];

        float2 uv = interpolateUV(inst, bary, primId);
        float4 baseTex = sampleTex(mat.baseColorTex, uv);
        float4 baseColor = mat.baseColorFactor * baseTex;

        float metallic = mat.metallicFactor;
        float roughness = mat.roughnessFactor;
        if (mat.mrTex >= 0) {
            float3 mr = gTextures[mat.mrTex].SampleLevel(gLinearSampler, uv, 0).rgb;
            roughness *= mr.g;
            metallic *= mr.b;
        }

        // 计算直接光照
        radiance = evalDirectLight(worldPos, normal, baseColor.rgb, metallic, roughness);

        // 添加 emissive
        float3 emi = mat.emissiveFactor;
        if (mat.emissiveTex >= 0)
            emi *= gTextures[mat.emissiveTex].SampleLevel(gLinearSampler, uv, 0).rgb;
        radiance += emi;

        // 添加环境光
        radiance += gFrame.ambient.rgb * baseColor.rgb * (1.0 - metallic);
    } else {
        // Miss: 天空颜色
        radiance = gFrame.ambient.rgb * 0.5;  // 简化天空
    }

    // 写入样本 buffer（atomic add 获取写入位置）
    uint idx;
    InterlockedAdd(gSampleCount[0], 1, idx);
    uint maxSamples = gPC.probesX * gPC.probesY * gPC.probesZ * gPC.raysPerProbe;
    if (idx < maxSamples) {
        uint base = idx * 9;
        gSampleBuf[base + 0] = worldPos.x;
        gSampleBuf[base + 1] = worldPos.y;
        gSampleBuf[base + 2] = worldPos.z;
        gSampleBuf[base + 3] = normal.x;
        gSampleBuf[base + 4] = normal.y;
        gSampleBuf[base + 5] = normal.z;
        gSampleBuf[base + 6] = radiance.r;
        gSampleBuf[base + 7] = radiance.g;
        gSampleBuf[base + 8] = radiance.b;
    }
}
