#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "gltf_loader.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cfloat>

namespace somegi {

static glm::mat4 toMat4(const cgltf_node* n) {
    if (n->has_matrix) return glm::make_mat4(n->matrix);
    glm::mat4 t = glm::translate(glm::mat4(1.0f),
                                 n->has_translation ? glm::vec3(n->translation[0], n->translation[1], n->translation[2])
                                                    : glm::vec3(0));
    glm::mat4 r(1.0f);
    if (n->has_rotation) {
        glm::quat q(n->rotation[3], n->rotation[0], n->rotation[1], n->rotation[2]);
        r = glm::mat4_cast(q);
    }
    glm::mat4 s = glm::scale(glm::mat4(1.0f),
                             n->has_scale ? glm::vec3(n->scale[0], n->scale[1], n->scale[2])
                                          : glm::vec3(1));
    return t * r * s;
}

static glm::mat4 worldOf(const cgltf_node* n) {
    glm::mat4 m = toMat4(n);
    while (n->parent) { n = n->parent; m = toMat4(n) * m; }
    return m;
}

static bool readImageRGBA(const cgltf_image* img, const std::filesystem::path& gltfDir,
                          TextureCpu& out, bool srgb) {
    out.isSrgb = srgb;
    if (img->buffer_view) {
        const auto* bv = img->buffer_view;
        const uint8_t* src = (const uint8_t*)bv->buffer->data + bv->offset;
        int w, h, c;
        stbi_uc* pix = stbi_load_from_memory(src, (int)bv->size, &w, &h, &c, 4);
        if (!pix) return false;
        out.width = w; out.height = h; out.channels = 4;
        out.rgba.assign(pix, pix + size_t(w)*h*4);
        stbi_image_free(pix);
        return true;
    } else if (img->uri) {
        auto p = gltfDir / img->uri;
        int w, h, c;
        stbi_uc* pix = stbi_load(p.string().c_str(), &w, &h, &c, 4);
        if (!pix) return false;
        out.width = w; out.height = h; out.channels = 4;
        out.rgba.assign(pix, pix + size_t(w)*h*4);
        stbi_image_free(pix);
        return true;
    }
    return false;
}

template <typename T>
static const T* accessorPtr(const cgltf_accessor* acc, size_t i) {
    auto* bv = acc->buffer_view;
    auto* buf = (const uint8_t*)bv->buffer->data + bv->offset + acc->offset;
    size_t stride = acc->stride ? acc->stride : cgltf_calc_size(acc->type, acc->component_type);
    return (const T*)(buf + i * stride);
}

bool loadGltf(const std::filesystem::path& path, SceneCpu& s, std::string& err) {
    cgltf_options opt{};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&opt, path.string().c_str(), &data) != cgltf_result_success) {
        err = "cgltf_parse_file failed"; return false;
    }
    if (cgltf_load_buffers(&opt, data, path.string().c_str()) != cgltf_result_success) {
        cgltf_free(data); err = "cgltf_load_buffers failed"; return false;
    }

    auto dir = path.parent_path();

    s.textures.resize(data->images_count);
    for (size_t i = 0; i < data->images_count; ++i) {
        if (!readImageRGBA(&data->images[i], dir, s.textures[i], false)) {
            // 留空（width=0），后续上传跳过 / 用默认
        }
    }

    auto texFromView = [&](const cgltf_texture_view& v, bool srgb) -> int {
        if (!v.texture || !v.texture->image) return -1;
        size_t idx = (size_t)(v.texture->image - data->images);
        if (srgb) s.textures[idx].isSrgb = true;
        return (int)idx;
    };

    s.materials.resize(data->materials_count);
    for (size_t i = 0; i < data->materials_count; ++i) {
        const auto& m = data->materials[i];
        MaterialDesc d{};
        if (m.has_pbr_metallic_roughness) {
            const auto& pmr = m.pbr_metallic_roughness;
            d.baseColorFactor = glm::vec4(pmr.base_color_factor[0], pmr.base_color_factor[1],
                                          pmr.base_color_factor[2], pmr.base_color_factor[3]);
            d.metallicFactor = pmr.metallic_factor;
            d.roughnessFactor = pmr.roughness_factor;
            d.baseColorTex = texFromView(pmr.base_color_texture, true);
            d.mrTex = texFromView(pmr.metallic_roughness_texture, false);
        }
        d.normalTex = texFromView(m.normal_texture, false);
        d.normalScale = m.normal_texture.scale;
        d.occlusionTex = texFromView(m.occlusion_texture, false);
        d.occlusionStrength = m.occlusion_texture.scale;
        d.emissiveTex = texFromView(m.emissive_texture, true);
        d.emissiveFactor = glm::vec3(m.emissive_factor[0], m.emissive_factor[1], m.emissive_factor[2]);
        d.alphaCutoff = m.alpha_cutoff;
        d.alphaMode = (m.alpha_mode == cgltf_alpha_mode_mask) ? 1u :
                      (m.alpha_mode == cgltf_alpha_mode_blend) ? 2u : 0u;
        d.doubleSided = m.double_sided ? 1u : 0u;
        s.materials[i] = d;
    }

    s.meshes.resize(data->meshes_count);
    for (size_t mi = 0; mi < data->meshes_count; ++mi) {
        auto& mesh = data->meshes[mi];
        Mesh& M = s.meshes[mi];
        glm::vec3 meshMn(FLT_MAX), meshMx(-FLT_MAX);
        M.primitives.resize(mesh.primitives_count);
        for (size_t pi = 0; pi < mesh.primitives_count; ++pi) {
            const auto& p = mesh.primitives[pi];
            Primitive prim{};
            prim.vertexOffset = (int32_t)s.vertices.size();
            prim.firstIndex = (uint32_t)s.indices.size();
            prim.materialIndex = p.material ? (int)(p.material - data->materials) : -1;

            const cgltf_accessor *posA = nullptr, *nrmA = nullptr, *tanA = nullptr, *uvA = nullptr;
            for (size_t a = 0; a < p.attributes_count; ++a) {
                const auto& at = p.attributes[a];
                if (at.type == cgltf_attribute_type_position) posA = at.data;
                else if (at.type == cgltf_attribute_type_normal) nrmA = at.data;
                else if (at.type == cgltf_attribute_type_tangent) tanA = at.data;
                else if (at.type == cgltf_attribute_type_texcoord && at.index == 0) uvA = at.data;
            }
            if (!posA) continue;
            size_t vc = posA->count;
            for (size_t v = 0; v < vc; ++v) {
                Vertex vx{};
                const float* P = accessorPtr<float>(posA, v);
                vx.position = glm::vec3(P[0], P[1], P[2]);
                if (nrmA) {
                    const float* N = accessorPtr<float>(nrmA, v);
                    vx.normal = glm::vec3(N[0], N[1], N[2]);
                } else vx.normal = glm::vec3(0, 1, 0);
                if (tanA) {
                    const float* T = accessorPtr<float>(tanA, v);
                    vx.tangent = glm::vec4(T[0], T[1], T[2], T[3]);
                } else vx.tangent = glm::vec4(1, 0, 0, 1);
                if (uvA) {
                    const float* U = accessorPtr<float>(uvA, v);
                    vx.uv0 = glm::vec2(U[0], U[1]);
                } else vx.uv0 = glm::vec2(0);
                s.vertices.push_back(vx);
                meshMn = glm::min(meshMn, vx.position);
                meshMx = glm::max(meshMx, vx.position);
            }

            if (p.indices) {
                size_t ic = p.indices->count;
                prim.indexCount = (uint32_t)ic;
                for (size_t i = 0; i < ic; ++i) {
                    s.indices.push_back((uint32_t)cgltf_accessor_read_index(p.indices, i));
                }
            } else {
                prim.indexCount = (uint32_t)vc;
                for (uint32_t i = 0; i < vc; ++i) s.indices.push_back(i);
            }
            M.primitives[pi] = prim;
        }
        M.localAabbMin = meshMn;
        M.localAabbMax = meshMx;
    }

    s.nodes.resize(data->nodes_count);
    for (size_t i = 0; i < data->nodes_count; ++i) {
        Node N{};
        N.worldTransform = worldOf(&data->nodes[i]);
        N.meshIndex = data->nodes[i].mesh ? (int)(data->nodes[i].mesh - data->meshes) : -1;
        s.nodes[i] = N;
    }

    // World-space AABB: 把每个 node 引用的 mesh 的 8 角点做变换并合并。
    glm::vec3 wMn(FLT_MAX), wMx(-FLT_MAX);
    bool any = false;
    for (auto& n : s.nodes) {
        if (n.meshIndex < 0) continue;
        auto& M = s.meshes[n.meshIndex];
        if (M.localAabbMin.x > M.localAabbMax.x) continue;
        for (int c = 0; c < 8; ++c) {
            glm::vec4 corner(
                (c & 1) ? M.localAabbMax.x : M.localAabbMin.x,
                (c & 2) ? M.localAabbMax.y : M.localAabbMin.y,
                (c & 4) ? M.localAabbMax.z : M.localAabbMin.z,
                1.0f);
            glm::vec3 w = glm::vec3(n.worldTransform * corner);
            wMn = glm::min(wMn, w);
            wMx = glm::max(wMx, w);
            any = true;
        }
    }
    if (any) { s.aabbMin = wMn; s.aabbMax = wMx; }
    else     { s.aabbMin = glm::vec3(-1); s.aabbMax = glm::vec3(1); }

    cgltf_free(data);
    return true;
}

}
