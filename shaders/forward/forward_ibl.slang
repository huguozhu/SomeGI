import shared_types;
import pbr;
import ibl_eval;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] StructuredBuffer<MaterialGpu> gMaterials;
[[vk::binding(2, 0)]] SamplerState gLinear;
[[vk::binding(3, 0)]] Texture2D gTextures[];

// NDGI MLP 权重 (bindings 4-9)
[[vk::binding(4, 0)]] StructuredBuffer<float4> gNdgiW1;
[[vk::binding(5, 0)]] StructuredBuffer<float4> gNdgiB1;
[[vk::binding(6, 0)]] StructuredBuffer<float4> gNdgiW2;
[[vk::binding(7, 0)]] StructuredBuffer<float4> gNdgiB2;
[[vk::binding(8, 0)]] StructuredBuffer<float4> gNdgiW3;
[[vk::binding(9, 0)]] StructuredBuffer<float4> gNdgiB3;

[[vk::binding(10, 0)]] StructuredBuffer<DrawData> gDrawData;

struct VsIn {
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float4 tangent : TANGENT;
    float2 uv0     : TEXCOORD0;
};

struct VsOut {
    float4 svPos    : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float4 tangent  : TEXCOORD2;
    float2 uv       : TEXCOORD3;
    nointerpolation int matIndex : TEXCOORD4;
};

[shader("vertex")]
VsOut vs_main(VsIn i, uint drawID : SV_InstanceID) {
    DrawData dd = gDrawData[drawID];
    VsOut o;
    float4 wp = mul(dd.modelMatrix, float4(i.pos, 1.0));
    o.worldPos = wp.xyz;
    o.svPos = mul(gFrame.viewProj, wp);
    float3x3 n3 = (float3x3)dd.modelMatrix;
    o.normal  = normalize(mul(n3, i.normal));
    o.tangent = float4(normalize(mul(n3, i.tangent.xyz)), i.tangent.w);
    o.uv = i.uv0;
    o.matIndex = int(dd.materialIndex);
    return o;
}

float4 sampleTex(int idx, float2 uv) {
    if (idx < 0) return float4(1, 1, 1, 1);
    return gTextures[idx].Sample(gLinear, uv);
}

[shader("pixel")]
float4 ps_main(VsOut i) : SV_Target {
    MaterialGpu m = gMaterials[i.matIndex];

    float4 baseTex = sampleTex(m.baseColorTex, i.uv);
    float4 base = m.baseColorFactor * baseTex;
    if (m.alphaMode == 1u && base.a < m.alphaCutoff) discard;

    float metallic = m.metallicFactor;
    float roughness = m.roughnessFactor;
    if (m.mrTex >= 0) {
        float3 t = gTextures[m.mrTex].Sample(gLinear, i.uv).rgb;
        roughness *= t.g;
        metallic  *= t.b;
    }
    metallic = saturate(metallic);
    roughness = saturate(roughness);

    float3 N = normalize(i.normal);
    if (m.normalTex >= 0) {
        float3 nm = gTextures[m.normalTex].Sample(gLinear, i.uv).rgb * 2.0 - 1.0;
        nm.xy *= m.normalScale;
        float3 T = normalize(i.tangent.xyz);
        float3 B = normalize(cross(N, T) * i.tangent.w);
        N = normalize(T*nm.x + B*nm.y + N*nm.z);
    }

    float3 V = normalize(gFrame.cameraPos.xyz - i.worldPos);
    float3 L = normalize(-gFrame.sunDir.xyz);
    float3 sun = directBRDF(N, V, L, base.rgb, metallic, roughness) *
                 gFrame.sunColor_intensity.rgb * gFrame.sunColor_intensity.w;

    float occ = 1.0;
    if (m.occlusionTex >= 0) {
        float t = gTextures[m.occlusionTex].Sample(gLinear, i.uv).r;
        occ = lerp(1.0, t, m.occlusionStrength);
    }

    // NDGI / IBL 路径选择（ddgiCounts.w: 0=none, 1=DDGI, 2=NDGI）
    float3 indirect = float3(0);
    if (gFrame.ddgiCounts.w == 2) {
        // NDGI: neural irradiance cache for diffuse
        float3 ndgiIrr = ndgiSampleIrradiance(i.worldPos, N);
        indirect = ndgiIrr * base.rgb * (1.0 - metallic) / 3.14159265;
    } else {
        // IBL：specularMipCount 经 gFrame.counts.y 注入
        indirect = evalIBL(N, V, base.rgb, metallic, roughness,
                           (float)gFrame.counts.y);
        indirect *= occ;
    }

    float3 emi = m.emissiveFactor;
    if (m.emissiveTex >= 0) emi *= gTextures[m.emissiveTex].Sample(gLinear, i.uv).rgb;

    return float4(sun + indirect + emi, base.a);
}

// NDGI MLP 推理（内联版，bindings 4-9，避免 import 路径的 binding 冲突）
float3 ndgiSampleIrradiance(float3 worldPos, float3 N) {
    float4 w1r0=gNdgiW1[0],w1r1=gNdgiW1[1],w1r2=gNdgiW1[2],w1r3=gNdgiW1[3];
    float4 w1r4=gNdgiW1[4],w1r5=gNdgiW1[5],w1r6=gNdgiW1[6],w1r7=gNdgiW1[7];
    float4 w1r8=gNdgiW1[8],w1r9=gNdgiW1[9],w1r10=gNdgiW1[10],w1r11=gNdgiW1[11];
    float4 w1r12=gNdgiW1[12],w1r13=gNdgiW1[13],w1r14=gNdgiW1[14],w1r15=gNdgiW1[15];
    float4 w1r16=gNdgiW1[16],w1r17=gNdgiW1[17],w1r18=gNdgiW1[18],w1r19=gNdgiW1[19];
    float4 w1r20=gNdgiW1[20],w1r21=gNdgiW1[21],w1r22=gNdgiW1[22],w1r23=gNdgiW1[23];
    float4 h1[4];
    h1[0]=gNdgiB1[0]+worldPos.x*w1r0+worldPos.y*w1r4+worldPos.z*w1r8+N.x*w1r12+N.y*w1r16+N.z*w1r20;
    h1[1]=gNdgiB1[1]+worldPos.x*w1r1+worldPos.y*w1r5+worldPos.z*w1r9+N.x*w1r13+N.y*w1r17+N.z*w1r21;
    h1[2]=gNdgiB1[2]+worldPos.x*w1r2+worldPos.y*w1r6+worldPos.z*w1r10+N.x*w1r14+N.y*w1r18+N.z*w1r22;
    h1[3]=gNdgiB1[3]+worldPos.x*w1r3+worldPos.y*w1r7+worldPos.z*w1r11+N.x*w1r15+N.y*w1r19+N.z*w1r23;
    for(int i=0;i<4;i++)h1[i]=max(h1[i],0.0)+0.01*min(h1[i],0.0);
    float h1f[16]={h1[0].x,h1[0].y,h1[0].z,h1[0].w,h1[1].x,h1[1].y,h1[1].z,h1[1].w,h1[2].x,h1[2].y,h1[2].z,h1[2].w,h1[3].x,h1[3].y,h1[3].z,h1[3].w};
    float4 h2[4]={gNdgiB2[0],gNdgiB2[1],gNdgiB2[2],gNdgiB2[3]};
    for(int j=0;j<16;j++){float v=h1f[j];int b=j*4;h2[0]+=v*gNdgiW2[b+0];h2[1]+=v*gNdgiW2[b+1];h2[2]+=v*gNdgiW2[b+2];h2[3]+=v*gNdgiW2[b+3];}
    for(int i=0;i<4;i++)h2[i]=max(h2[i],0.0)+0.01*min(h2[i],0.0);
    float h2f[16]={h2[0].x,h2[0].y,h2[0].z,h2[0].w,h2[1].x,h2[1].y,h2[1].z,h2[1].w,h2[2].x,h2[2].y,h2[2].z,h2[2].w,h2[3].x,h2[3].y,h2[3].z,h2[3].w};
    float r=gNdgiB3[0].x,g=gNdgiB3[0].y,b=gNdgiB3[0].z;
    for(int j=0;j<16;j++){float v=h2f[j];int s=(j*3)/4,c=(j*3)%4;
        float w0=(c==0?gNdgiW3[s].x:c==1?gNdgiW3[s].y:c==2?gNdgiW3[s].z:gNdgiW3[s].w);
        float w1=(c==0||c==3?(c==0?gNdgiW3[s].y:gNdgiW3[s+1].x):(c==1?gNdgiW3[s].z:gNdgiW3[s+1].y));
        float w2=(c==3?gNdgiW3[s+1].x:c==2?gNdgiW3[s+1].w:gNdgiW3[s+1].z);
        if(c==0){r+=v*gNdgiW3[s].x;g+=v*gNdgiW3[s].y;b+=v*gNdgiW3[s].z;}
        else if(c==1){r+=v*gNdgiW3[s].w;g+=v*gNdgiW3[s+1].x;b+=v*gNdgiW3[s+1].y;}
        else if(c==2){r+=v*gNdgiW3[s+1].z;g+=v*gNdgiW3[s+1].w;b+=v*gNdgiW3[s+2].x;}
        else{r+=v*gNdgiW3[s+1].w;g+=v*gNdgiW3[s+2].x;b+=v*gNdgiW3[s+2].y;}}
    return max(float3(r,g,b),0.0);
}
