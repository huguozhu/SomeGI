import shared_types;
struct CullUniforms { float4 frustum[6]; uint drawCount; };
[[vk::binding(0,0)]] StructuredBuffer<DrawData> gIn;
[[vk::binding(1,0)]] ConstantBuffer<CullUniforms> gCull;
[[vk::binding(2,0)]] RWStructuredBuffer<IndirectDrawCommand> gOut;
[[vk::binding(3,0)]] RWStructuredBuffer<uint> gCount;

bool aabbInFrustum(float3 mn,float3 mx,float4 f[6]){
    for(int i=0;i<6;++i){float3 n=f[i].xyz;float d=f[i].w;
        float3 p=float3(n.x>0?mn.x:mx.x,n.y>0?mn.y:mx.y,n.z>0?mn.z:mx.z);
        if(dot(n,p)+d<0)return false;}
    return true;
}
[numthreads(256,1,1)] void cs_main(uint3 tid:SV_DispatchThreadID){
    if(tid.x>=gCull.drawCount)return;
    DrawData dd=gIn[tid.x];
    if(!aabbInFrustum(dd.aabbMin,dd.aabbMax,gCull.frustum))return;
    uint idx;InterlockedAdd(gCount[0],1,idx);
    IndirectDrawCommand cmd;
    cmd.indexCount=dd.indexCount;cmd.instanceCount=1;cmd.firstIndex=dd.firstIndex;cmd.vertexOffset=dd.vertexOffset;cmd.firstInstance=tid.x;
    gOut[idx]=cmd;
}
