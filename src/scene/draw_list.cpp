#include "scene/draw_list.h"
#include <algorithm>
#include <limits>
namespace somegi {
static std::pair<glm::vec3,glm::vec3> wAABB(const glm::vec3& mn,const glm::vec3& mx,const glm::mat4& w){
    glm::vec3 a{1e30f},b{-1e30f};
    for(int i=0;i<8;++i){glm::vec3 c((i&1)?mx.x:mn.x,(i&2)?mx.y:mn.y,(i&4)?mx.z:mn.z);
    glm::vec3 v=glm::vec3(w*glm::vec4(c,1));a=glm::min(a,v);b=glm::max(b,v);}
    return{a,b};
}
void buildDrawList(const SceneCpu& cpu,std::vector<DrawEntry>& out){
    out.clear();size_t n=0;
    for(auto& nd:cpu.nodes)if(nd.meshIndex>=0)n+=cpu.meshes[nd.meshIndex].primitives.size();
    out.reserve(n);
    for(auto& nd:cpu.nodes){if(nd.meshIndex<0)continue;const Mesh& M=cpu.meshes[nd.meshIndex];
        for(auto& p:M.primitives){DrawEntry e{};e.worldTransform=nd.worldTransform;
            e.materialIndex=p.materialIndex>=0?p.materialIndex:0;e.firstIndex=p.firstIndex;
            e.indexCount=p.indexCount;e.vertexOffset=p.vertexOffset;
            auto[mn,mx]=wAABB(M.localAabbMin,M.localAabbMax,nd.worldTransform);
            e.aabbMin=mn;e.aabbMax=mx;out.push_back(e);}}
}}
