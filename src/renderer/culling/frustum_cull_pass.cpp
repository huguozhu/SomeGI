#include "renderer/culling/frustum_cull_pass.h"
#include "core/device.h"
#include "core/shader.h"
#include <array>
#include <cstring>
#include <cmath>

namespace somegi {

struct CullUbo {
    glm::mat4 viewProj;
    glm::vec4 frustum[6];
    glm::vec2 screenSize;
    uint32_t  drawCount;
    uint32_t  hizMaxMip;
};

void extractFrustumPlanes(const glm::mat4& vp, glm::vec4 f[6]) {
    glm::vec4 r1=vp[0],r2=vp[1],r3=vp[2],r4=vp[3];
    f[0]=r4+r1;f[1]=r4-r1;f[2]=r4+r2;f[3]=r4-r2;f[4]=r4+r3;f[5]=r4-r3;
    for(int i=0;i<6;++i){float L=glm::length(glm::vec3(f[i]));if(L>1e-10f)f[i]/=L;}
}

void FrustumCullPass::init(Device& d, uint32_t maxDraws) {
    m_device=&d; m_maxDraws=maxDraws;
    // 8 bindings: 0-3 culling, 4-7 Hi-Z mips
    std::array<VkDescriptorSetLayoutBinding,8> b{};
    b[0]={0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT};
    b[1]={1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT};
    b[2]={2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT};
    b[3]={3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT};
    for(uint32_t i=4;i<8;++i)
        b[i]={i,VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,0,0,8,b.data()};
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(),&li,nullptr,&m_dsl));
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,0,0,1,&m_dsl,0,nullptr};
    VK_CHECK(vkCreatePipelineLayout(d.device(),&pli,nullptr,&m_pl));
    ShaderModule sh(d,shaderDir()/"culling"/"frustum_cull.spv");
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=sh.handle();cp.stage.pName="cs_main";
    cp.layout=m_pl;
    VK_CHECK(vkCreateComputePipelines(d.device(),VK_NULL_HANDLE,1,&cp,nullptr,&m_pipe));
    VkDescriptorPoolSize ps[3]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3*kFramesInFlight},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,kFramesInFlight},{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,4*kFramesInFlight}};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets=kFramesInFlight;pci.poolSizeCount=3;pci.pPoolSizes=ps;
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
    m_ubo.reset();m_device=nullptr;
}

static void writeDescriptors(Device* d, VkDescriptorSet ds, VkBuffer drawBuf, VkBuffer uboBuf, VkBuffer indOut, VkBuffer cntOut, VkImageView hiz[4]) {
    VkDescriptorBufferInfo ddi{drawBuf,0,VK_WHOLE_SIZE},ubi{uboBuf,0,VK_WHOLE_SIZE},idi{indOut,0,VK_WHOLE_SIZE},cdi{cntOut,0,VK_WHOLE_SIZE};
    std::array<VkWriteDescriptorSet,8> w{};
    w[0].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[0].dstSet=ds;w[0].dstBinding=0;w[0].descriptorCount=1;w[0].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[0].pBufferInfo=&ddi;
    w[1].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[1].dstSet=ds;w[1].dstBinding=1;w[1].descriptorCount=1;w[1].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;w[1].pBufferInfo=&ubi;
    w[2].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[2].dstSet=ds;w[2].dstBinding=2;w[2].descriptorCount=1;w[2].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[2].pBufferInfo=&idi;
    w[3].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[3].dstSet=ds;w[3].dstBinding=3;w[3].descriptorCount=1;w[3].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[3].pBufferInfo=&cdi;
    for(int i=0;i<4;++i){
        VkDescriptorImageInfo ii{};ii.imageView=hiz[i];ii.imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        w[4+i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w[4+i].dstSet=ds;w[4+i].dstBinding=4+i;w[4+i].descriptorCount=1;w[4+i].descriptorType=VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;w[4+i].pImageInfo=&ii;
    }
    vkUpdateDescriptorSets(d->device(),8,w.data(),0,nullptr);
}

void FrustumCullPass::record(VkCommandBuffer cmd, VkBuffer drawBuf, uint32_t drawCount, VkBuffer indirectOut, VkBuffer countOut, const glm::mat4& vp, VkExtent2D ss, uint32_t fi) {
    VkImageView nullViews[4]={};
    record(cmd,drawBuf,drawCount,indirectOut,countOut,vp,ss,fi,nullViews[0],nullViews[1],nullViews[2],nullViews[3]);
}

void FrustumCullPass::record(VkCommandBuffer cmd, VkBuffer drawBuf, uint32_t drawCount, VkBuffer indirectOut, VkBuffer countOut, const glm::mat4& vp, VkExtent2D ss, uint32_t fi, VkImageView hizMip1, VkImageView hizMip2, VkImageView hizMip3, VkImageView hizMip4) {
    if(drawCount==0)return;
    // Zero count buffer
    vkCmdFillBuffer(cmd,countOut,0,sizeof(uint32_t),0);
    VkBufferMemoryBarrier2 fb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    fb.srcStageMask=VK_PIPELINE_STAGE_2_CLEAR_BIT;fb.srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT;
    fb.dstStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;fb.dstAccessMask=VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    fb.buffer=countOut;fb.size=VK_WHOLE_SIZE;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};di.bufferMemoryBarrierCount=1;di.pBufferMemoryBarriers=&fb;
    vkCmdPipelineBarrier2(cmd,&di);
    // Update UBO
    CullUbo u{};u.viewProj=vp;extractFrustumPlanes(vp,u.frustum);
    u.screenSize=glm::vec2(ss.width,ss.height);u.drawCount=drawCount;
    u.hizMaxMip=(hizMip1!=VK_NULL_HANDLE)?4u:0u;
    std::memcpy(m_ubo.mapped(),&u,sizeof(u));
    // Write descriptors
    VkImageView hiz[4]={hizMip1,hizMip2,hizMip3,hizMip4};
    writeDescriptors(m_device,m_sets[fi],drawBuf,m_ubo.handle(),indirectOut,countOut,hiz);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,m_pipe);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,m_pl,0,1,&m_sets[fi],0,nullptr);
    vkCmdDispatch(cmd,(drawCount+255)/256,1,1);
}

}
