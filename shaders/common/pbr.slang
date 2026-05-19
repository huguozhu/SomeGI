static const float PI = 3.14159265359;

float3 fresnelSchlick(float cosT, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosT), 5.0);
}
float distributionGGX(float NoH, float a) {
    float a2 = a*a;
    float d = (NoH*NoH * (a2 - 1.0) + 1.0);
    return a2 / (PI * d * d + 1e-7);
}
float geometrySmith(float NoV, float NoL, float a) {
    float k = (a + 1.0); k = (k*k) / 8.0;
    float gv = NoV / (NoV * (1 - k) + k);
    float gl = NoL / (NoL * (1 - k) + k);
    return gv * gl;
}

float3 directBRDF(float3 N, float3 V, float3 L, float3 albedo, float metallic, float roughness) {
    float a = max(roughness*roughness, 0.002);
    float3 H = normalize(L + V);
    float NoV = saturate(dot(N, V));
    float NoL = saturate(dot(N, L));
    float NoH = saturate(dot(N, H));
    float VoH = saturate(dot(V, H));
    float3 F0 = lerp(float3(0.04), albedo, metallic);
    float3 F = fresnelSchlick(VoH, F0);
    float D = distributionGGX(NoH, a);
    float G = geometrySmith(NoV, NoL, a);
    float3 spec = (D * G * F) / max(4.0 * NoV * NoL, 1e-4);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    return (kd * albedo / PI + spec) * NoL;
}
