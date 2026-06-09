#include "renderer/culling/frustum_cull_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>
#include <cstring>
#include <cmath>
namespace somegi {
struct CullUbo { glm::vec4 f[6]; uint32_t n; uint32_t _p[3]; };
void extractFrustumPlanes(const glm::mat4& vp, glm::vec4 f[6]){
    glm::vec4 r1=vp[0],r2=vp[1],r3=vp[2],r4=vp[3];
    f[0]=r4+r1;f[1]=r4-r1;f[2]=r4+r2;f[3]=r4-r2;f[4]=r4+r3;f[5]=r4-r3;
    for(int i=0;i<6;++i){float L=glm::length(glm::vec3(f[i]));if(L>1e-10f)f[i]/=L;}
}
void FrustumCullPass::init(Device& d, uint32_t){
    m_device=&d;
    std::array<VkDescriptorSetLayoutBinding,4> b{};
    b[0]={0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT};
    b[1]={1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT};
    b[2]={2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT};
    b[3]={3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,(uint32_t)b.size(),b.data()};
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(),&li,nullptr,&m_dsl));
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&m_dsl,0,nullptr};
    VK_CHECK(vkCreatePipelineLayout(d.device(),&pli,nullptr,&m_pl));
    ShaderModule sh(d,shaderDir()/"culling"/"frustum_cull.spv");
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=sh.handle();cp.stage.pName="cs_main";
    cp.layout=m_pl;
    VK_CHECK(vkCreateComputePipelines(d.device(),VK_NULL_HANDLE,1,&cp,nullptr,&m_pipe));
    std::array<VkDescriptorPoolSize,2> ps;
    ps[0].type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;ps[0].descriptorCount=3*kFramesInFlight;
    ps[1].type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;ps[1].descriptorCount=kFramesInFlight;
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = kFramesInFlight; pci.poolSizeCount = 2; pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(),&pci,nullptr,&m_pool));
    VkDescriptorSetLayout ls[kFramesInFlight]={m_dsl,m_dsl};
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,0,m_pool,kFramesInFlight,ls};
    VK_CHECK(vkAllocateDescriptorSets(d.device(),&ai,m_sets));
    m_ubo=Buffer(d,sizeof(CullUbo),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}
void FrustumCullPass::destroy(){
    if(!m_device)return;auto dev=m_device->device();
    if(m_pipe)vkDestroyPipeline(dev,m_pipe,nullptr);
    if(m_pl)vkDestroyPipelineLayout(dev,m_pl,nullptr);
    if(m_pool)vkDestroyDescriptorPool(dev,m_pool,nullptr);
    if(m_dsl)vkDestroyDescriptorSetLayout(dev,m_dsl,nullptr);
    m_pipe=VK_NULL_HANDLE;m_pl=VK_NULL_HANDLE;m_pool=VK_NULL_HANDLE;m_dsl=VK_NULL_HANDLE;
    m_ubo.reset();m_device=nullptr;
}
void FrustumCullPass::record(VkCommandBuffer cmd,VkBuffer drawBuf,uint32_t drawCount,VkBuffer indirectOut,VkBuffer countOut,const glm::mat4& vp,uint32_t fi){
    if(drawCount==0)return;
    vkCmdFillBuffer(cmd,countOut,0,sizeof(uint32_t),0);
    VkBufferMemoryBarrier2 fb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    fb.srcStageMask=VK_PIPELINE_STAGE_2_CLEAR_BIT;fb.srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT;
    fb.dstStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;fb.dstAccessMask=VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    fb.buffer=countOut;fb.size=VK_WHOLE_SIZE;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};di.bufferMemoryBarrierCount=1;di.pBufferMemoryBarriers=&fb;
    vkCmdPipelineBarrier2(cmd,&di);
    CullUbo u;extractFrustumPlanes(vp,u.f);u.n=drawCount;
    std::memcpy(m_ubo.mapped(),&u,sizeof(u));
    VkDescriptorSet ds=m_sets[fi];
    VkDescriptorBufferInfo ddi{drawBuf,0,VK_WHOLE_SIZE},ubi{m_ubo.handle(),0,VK_WHOLE_SIZE},idi{indirectOut,0,VK_WHOLE_SIZE},cdi{countOut,0,VK_WHOLE_SIZE};
    std::array<VkWriteDescriptorSet,4> w{};
    w[0].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[0].dstSet=ds;w[0].dstBinding=0;w[0].descriptorCount=1;w[0].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[0].pBufferInfo=&ddi;
    w[1].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[1].dstSet=ds;w[1].dstBinding=1;w[1].descriptorCount=1;w[1].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;w[1].pBufferInfo=&ubi;
    w[2].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[2].dstSet=ds;w[2].dstBinding=2;w[2].descriptorCount=1;w[2].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[2].pBufferInfo=&idi;
    w[3].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[3].dstSet=ds;w[3].dstBinding=3;w[3].descriptorCount=1;w[3].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[3].pBufferInfo=&cdi;
    vkUpdateDescriptorSets(m_device->device(),(uint32_t)w.size(),w.data(),0,nullptr);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,m_pipe);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,m_pl,0,1,&ds,0,nullptr);
    vkCmdDispatch(cmd,(drawCount+255)/256,1,1);
}
}
