// Forward Mesh Shader（仅 Mesh + Fragment，无 Task Shader）

import shared_types;
import pbr;
import ibl_eval;

[[vk::binding(0, 0)]] StructuredBuffer<DrawData> gDrawData;
[[vk::binding(6, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(7, 0)]] StructuredBuffer<float> gVertexData;
[[vk::binding(8, 0)]] StructuredBuffer<uint> gIndexData;
[[vk::binding(9, 0)]] StructuredBuffer<MaterialGpu> gMaterials;
[[vk::binding(10,0)]] SamplerState gLinear;
[[vk::binding(11,0)]] Texture2D gTextures[];

#define VERTEX_FLOATS 14
#define MAX_VERTS 192
#define MAX_TRIS  64

struct VsOut {
    float4 svPos : SV_Position; float3 worldPos : TEXCOORD0; float3 normal : TEXCOORD1;
    float4 tangent : TEXCOORD2; float2 uv : TEXCOORD3; nointerpolation int matIndex : TEXCOORD4;
};

[shader("mesh")] [outputtopology("triangle")] [numthreads(64,1,1)]
void ms_main(uint32_t drawIndex : SV_GroupID, uint3 tid : SV_GroupThreadID,
    out vertices VsOut v[MAX_VERTS], out indices uint3 t[MAX_TRIS], out primitives float4 p[MAX_TRIS]) {
    DrawData dd = gDrawData[drawIndex];
    uint triCount = dd.indexCount/3; float3x3 n3=(float3x3)dd.modelMatrix;
    for (uint i=tid.x;i<MAX_TRIS;i+=64){if(i<triCount){
        uint3 idx; idx.x=gIndexData[dd.firstIndex+i*3+0];idx.y=gIndexData[dd.firstIndex+i*3+1];idx.z=gIndexData[dd.firstIndex+i*3+2];
        for(uint vv=0;vv<3;++vv){uint vi=(vv==0)?idx.x:((vv==1)?idx.y:idx.z);uint vo=(dd.vertexOffset+vi)*VERTEX_FLOATS;
            float3 pos=float3(gVertexData[vo+0],gVertexData[vo+1],gVertexData[vo+2]);
            float3 nrm=float3(gVertexData[vo+3],gVertexData[vo+4],gVertexData[vo+5]);
            float4 tan=float4(gVertexData[vo+6],gVertexData[vo+7],gVertexData[vo+8],gVertexData[vo+9]);
            float2 uv=float2(gVertexData[vo+10],gVertexData[vo+11]);
            float4 wp=mul(dd.modelMatrix,float4(pos,1.0));uint oi=i*3+vv;
            v[oi].svPos=mul(gFrame.viewProj,wp);v[oi].worldPos=wp.xyz;
            v[oi].normal=normalize(mul(n3,nrm));v[oi].tangent=float4(normalize(mul(n3,tan.xyz)),tan.w);
            v[oi].uv=uv;v[oi].matIndex=int(dd.materialIndex);}
        t[i]=uint3(i*3+0,i*3+1,i*3+2);}else{v[i*3+0].svPos=float4(0,0,0,0);v[i*3+1].svPos=float4(0,0,0,0);v[i*3+2].svPos=float4(0,0,0,0);t[i]=uint3(i*3+0,i*3+1,i*3+2);}}
}

float4 sampleTex(int idx,float2 uv){if(idx<0)return float4(1,1,1,1);return gTextures[idx].Sample(gLinear,uv);}
[shader("pixel")] float4 ps_main(VsOut i):SV_Target{
    MaterialGpu m=gMaterials[i.matIndex];float4 baseTex=sampleTex(m.baseColorTex,i.uv);float4 base=m.baseColorFactor*baseTex;
    if(m.alphaMode==1u&&base.a<m.alphaCutoff)discard;float metallic=m.metallicFactor,roughness=m.roughnessFactor;
    if(m.mrTex>=0){float3 t=gTextures[m.mrTex].Sample(gLinear,i.uv).rgb;roughness*=t.g;metallic*=t.b;}
    metallic=saturate(metallic);roughness=saturate(roughness);float3 N=normalize(i.normal);
    if(m.normalTex>=0){float3 nm=gTextures[m.normalTex].Sample(gLinear,i.uv).rgb*2.0-1.0;nm.xy*=m.normalScale;
        float3 T=normalize(i.tangent.xyz);float3 B=normalize(cross(N,T)*i.tangent.w);N=normalize(T*nm.x+B*nm.y+N*nm.z);}
    float3 V=normalize(gFrame.cameraPos.xyz-i.worldPos);float3 L=normalize(-gFrame.sunDir.xyz);
    float3 sun=directBRDF(N,V,L,base.rgb,metallic,roughness)*gFrame.sunColor_intensity.rgb*gFrame.sunColor_intensity.w;
    float occ=1.0;if(m.occlusionTex>=0){float t=gTextures[m.occlusionTex].Sample(gLinear,i.uv).r;occ=lerp(1.0,t,m.occlusionStrength);}
    float3 indirect=evalIBL(N,V,base.rgb,metallic,roughness,(float)gFrame.counts.y);indirect*=occ;
    float3 emi=m.emissiveFactor;if(m.emissiveTex>=0)emi*=gTextures[m.emissiveTex].Sample(gLinear,i.uv).rgb;
    return float4(sun+indirect+emi,base.a);
}
