# M1 — glTF 加载 + 前向直接光 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** 用 cgltf 加载 glTF v2 模型，按 metallic-roughness PBR 做前向直接光渲染，依次跑通 cube.gltf 和 Sponza.gltf。

**Architecture:**
- 前向 + 单方向光 + 简单环境光（纯灰色）。
- HDR offscreen R16G16B16A16_SFLOAT 渲染目标 → tonemap compute → swapchain。
- 顶点：position, normal, tangent, uv0（24+24 byte, packed）。glTF 缺 tangent 时 fragment 用 derivatives 近似。
- 材质：pbrMetallicRoughness 5 通道 (baseColor, mrPacked, normal, occlusion, emissive)；缺贴图用 1×1 默认值。
- Shader：slang，CMake 自动编译为 spv。

**Tech Stack:** cgltf v1.14, stb_image, slang, glm.

**Decision: GBuffer 推迟到 M4** —— 直接光 + IBL（M3）前向就够。

---

## 文件结构（M1 完成后）

```
src/
├── core/
│   ├── allocator.{h,cpp}         # 简易 VkAllocateMemory 包装
│   ├── buffer.{h,cpp}            # RAII Buffer
│   ├── image.{h,cpp}             # RAII Image + view
│   ├── shader.{h,cpp}            # 加载 spv
│   └── (M0 已有的)
├── scene/
│   ├── CMakeLists.txt
│   ├── gltf_loader.{h,cpp}       # cgltf 解析 → SceneCpu
│   ├── scene.h                   # SceneCpu / SceneGpu / Mesh / Primitive / MaterialDesc
│   ├── scene_gpu.{h,cpp}         # 上传几何/贴图到 GPU
│   ├── camera.{h,cpp}            # FlyCamera
│   └── upload.{h,cpp}            # one-shot transfer cmd
├── renderer/
│   ├── CMakeLists.txt
│   ├── forward_pass.{h,cpp}      # PBR pipeline + draw
│   ├── tonemap_pass.{h,cpp}      # compute, hdr → ldr swap
│   └── render_targets.{h,cpp}    # hdr color + depth
└── app/  (沿用 M0)

shaders/
├── common/
│   ├── pbr.slang
│   └── shared_types.slang
├── forward/
│   └── forward.slang             # vs + ps
└── tonemap/
    └── tonemap.slang             # compute
```

---

## Task 1: Shader 编译 CMake 规则

**Files:**
- Create: `cmake/CompileSlang.cmake`
- Modify: `CMakeLists.txt`（include + 调用）

- [ ] **Step 1: cmake/CompileSlang.cmake**

```cmake
# 用法：compile_slang_target(<target> SOURCES a.slang b.slang ENTRY vs_main ps_main)
# 简化做法：每个 .slang 文件含一个 entry，按 stage 后缀决定。
# 我们用：每个 .slang 可能含多个 entry，用户列出来即可。
function(compile_slang TARGET_NAME)
    set(options)
    set(oneValueArgs OUTPUT_DIR)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(SLANG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT SLANG_OUTPUT_DIR)
        set(SLANG_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
    endif()
    file(MAKE_DIRECTORY ${SLANG_OUTPUT_DIR})

    set(_outputs)
    foreach(SRC ${SLANG_SOURCES})
        get_filename_component(_name ${SRC} NAME_WE)
        get_filename_component(_dir ${SRC} DIRECTORY)
        # 把 shaders/forward/forward.slang -> forward/forward.spv
        file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/shaders ${SRC})
        get_filename_component(_rel_dir ${_rel} DIRECTORY)
        set(_outdir "${SLANG_OUTPUT_DIR}/${_rel_dir}")
        file(MAKE_DIRECTORY ${_outdir})
        set(_out "${_outdir}/${_name}.spv")
        # 命令：编译为 spirv，所有 entry，自动生成
        add_custom_command(
            OUTPUT ${_out}
            COMMAND ${SLANGC_EXE}
                ${SRC}
                -target spirv
                -profile spirv_1_5
                -o ${_out}
                -fvk-use-entrypoint-name
                -emit-spirv-directly
            DEPENDS ${SRC}
            COMMENT "slangc ${_rel}"
            VERBATIM
        )
        list(APPEND _outputs ${_out})
    endforeach()
    add_custom_target(${TARGET_NAME} DEPENDS ${_outputs})
endfunction()
```

- [ ] **Step 2: 顶层 CMakeLists.txt 引入并定义 shaders target**

在 `CMakeLists.txt` 末尾（`add_subdirectory(src/app)` 前）插入：

```cmake
include(cmake/CompileSlang.cmake)

file(GLOB_RECURSE SHADER_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/shaders/*.slang
)
if(SHADER_SOURCES AND SLANGC_EXE)
    compile_slang(somegi_shaders SOURCES ${SHADER_SOURCES})
    # 构建顺序：app 依赖 shaders
    set(SOMEGI_HAS_SHADERS ON)
endif()
```

并在 `src/app/CMakeLists.txt` 末尾加：

```cmake
if(TARGET somegi_shaders)
    add_dependencies(SomeGI somegi_shaders)
endif()
```

- [ ] **Step 3: 写一个 dummy slang 文件验证编译规则**

`shaders/_smoke/smoke.slang`：

```hlsl
[shader("vertex")]
float4 vs_main(float3 pos : POSITION) : SV_Position {
    return float4(pos, 1.0);
}

[shader("pixel")]
float4 ps_main() : SV_Target {
    return float4(1, 0, 0, 1);
}
```

- [ ] **Step 4: 重新 configure & 构建**

```powershell
cmake -S . -B build
cmake --build build --config Debug --target somegi_shaders
```

Expected: 在 `build/shaders/_smoke/smoke.spv` 生成文件，且 `cmake --build build --config Debug` 不再有错误。

- [ ] **Step 5: 提交**

```powershell
git add cmake/CompileSlang.cmake CMakeLists.txt src/app/CMakeLists.txt shaders/
git commit -m "build: add slang->spv shader compilation rule (M1)"
```

---

## Task 2: core/Allocator + Buffer + Image — RAII GPU 资源

**Files:**
- Create: `src/core/allocator.h`、`allocator.cpp`
- Create: `src/core/buffer.h`、`buffer.cpp`
- Create: `src/core/image.h`、`image.cpp`
- Modify: `src/core/CMakeLists.txt`（加 sources）

简易内存分配（不接 VMA），每个 Buffer/Image 独立 vkAllocateMemory；M1 够用，M4 之前换 VMA。

- [ ] **Step 1: allocator.h**

```cpp
#pragma once
#include "vk_common.h"

namespace somegi {

class Device;

uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props);

}
```

- [ ] **Step 2: allocator.cpp**

```cpp
#include "allocator.h"
#include "device.h"

namespace somegi {

uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("findMemoryType failed");
}

}
```

- [ ] **Step 3: buffer.h**

```cpp
#pragma once
#include "vk_common.h"

namespace somegi {

class Device;

class Buffer {
public:
    Buffer() = default;
    Buffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage,
           VkMemoryPropertyFlags memProps);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept { swap(o); }
    Buffer& operator=(Buffer&& o) noexcept { swap(o); return *this; }

    void swap(Buffer& o) noexcept;
    void reset();

    VkBuffer handle() const { return m_buffer; }
    VkDeviceMemory memory() const { return m_memory; }
    VkDeviceSize size() const { return m_size; }
    VkDeviceAddress deviceAddress() const { return m_address; }
    void* mapped() const { return m_mapped; }

private:
    Device* m_device = nullptr;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkDeviceSize m_size = 0;
    VkDeviceAddress m_address = 0;
    void* m_mapped = nullptr;
};

}
```

- [ ] **Step 4: buffer.cpp**

```cpp
#include "buffer.h"
#include "device.h"
#include "allocator.h"

namespace somegi {

Buffer::Buffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags memProps) : m_device(&d), m_size(size) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(d.device(), &ci, nullptr, &m_buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(d.device(), m_buffer, &req);

    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flags;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(d.physicalDevice(), req.memoryTypeBits, memProps);
    VK_CHECK(vkAllocateMemory(d.device(), &ai, nullptr, &m_memory));
    VK_CHECK(vkBindBufferMemory(d.device(), m_buffer, m_memory, 0));

    if (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        VK_CHECK(vkMapMemory(d.device(), m_memory, 0, size, 0, &m_mapped));
    }
    VkBufferDeviceAddressInfo bdai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bdai.buffer = m_buffer;
    m_address = vkGetBufferDeviceAddress(d.device(), &bdai);
}

Buffer::~Buffer() { reset(); }

void Buffer::swap(Buffer& o) noexcept {
    std::swap(m_device, o.m_device);
    std::swap(m_buffer, o.m_buffer);
    std::swap(m_memory, o.m_memory);
    std::swap(m_size, o.m_size);
    std::swap(m_address, o.m_address);
    std::swap(m_mapped, o.m_mapped);
}

void Buffer::reset() {
    if (m_device) {
        if (m_mapped) vkUnmapMemory(m_device->device(), m_memory);
        if (m_buffer) vkDestroyBuffer(m_device->device(), m_buffer, nullptr);
        if (m_memory) vkFreeMemory(m_device->device(), m_memory, nullptr);
    }
    m_device = nullptr;
    m_buffer = VK_NULL_HANDLE;
    m_memory = VK_NULL_HANDLE;
    m_size = 0; m_address = 0; m_mapped = nullptr;
}

}
```

- [ ] **Step 5: image.h**

```cpp
#pragma once
#include "vk_common.h"

namespace somegi {
class Device;

struct ImageDesc {
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkExtent3D extent = {1, 1, 1};
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkImageType type = VK_IMAGE_TYPE_2D;
    VkImageUsageFlags usage = 0;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageCreateFlags flags = 0;
};

class Image {
public:
    Image() = default;
    Image(Device& d, const ImageDesc& desc);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& o) noexcept { swap(o); }
    Image& operator=(Image&& o) noexcept { swap(o); return *this; }

    void swap(Image& o) noexcept;
    void reset();

    VkImage image() const { return m_image; }
    VkImageView view() const { return m_view; }
    VkFormat format() const { return m_desc.format; }
    VkExtent3D extent() const { return m_desc.extent; }
    const ImageDesc& desc() const { return m_desc; }

private:
    Device* m_device = nullptr;
    VkImage m_image = VK_NULL_HANDLE;
    VkImageView m_view = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    ImageDesc m_desc{};
};

}
```

- [ ] **Step 6: image.cpp**

```cpp
#include "image.h"
#include "device.h"
#include "allocator.h"

namespace somegi {

Image::Image(Device& d, const ImageDesc& desc) : m_device(&d), m_desc(desc) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.flags = desc.flags;
    ci.imageType = desc.type;
    ci.format = desc.format;
    ci.extent = desc.extent;
    ci.mipLevels = desc.mipLevels;
    ci.arrayLayers = desc.arrayLayers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = desc.usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(d.device(), &ci, nullptr, &m_image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(d.device(), m_image, &req);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(d.physicalDevice(), req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(d.device(), &ai, nullptr, &m_memory));
    VK_CHECK(vkBindImageMemory(d.device(), m_image, m_memory, 0));

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = m_image;
    vi.viewType = (desc.arrayLayers == 6 && (desc.flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT))
                  ? VK_IMAGE_VIEW_TYPE_CUBE
                  : (desc.type == VK_IMAGE_TYPE_2D ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_3D);
    vi.format = desc.format;
    vi.subresourceRange = {desc.aspect, 0, desc.mipLevels, 0, desc.arrayLayers};
    VK_CHECK(vkCreateImageView(d.device(), &vi, nullptr, &m_view));
}

Image::~Image() { reset(); }

void Image::swap(Image& o) noexcept {
    std::swap(m_device, o.m_device);
    std::swap(m_image, o.m_image);
    std::swap(m_view, o.m_view);
    std::swap(m_memory, o.m_memory);
    std::swap(m_desc, o.m_desc);
}

void Image::reset() {
    if (m_device) {
        if (m_view)   vkDestroyImageView(m_device->device(), m_view, nullptr);
        if (m_image)  vkDestroyImage(m_device->device(), m_image, nullptr);
        if (m_memory) vkFreeMemory(m_device->device(), m_memory, nullptr);
    }
    m_device = nullptr;
    m_image = VK_NULL_HANDLE; m_view = VK_NULL_HANDLE; m_memory = VK_NULL_HANDLE;
    m_desc = {};
}

}
```

- [ ] **Step 7: 更新 src/core/CMakeLists.txt**

```cmake
add_library(somegi_core STATIC
    vk_common.h
    window.h
    window.cpp
    device.h
    device.cpp
    swapchain.h
    swapchain.cpp
    allocator.h
    allocator.cpp
    buffer.h
    buffer.cpp
    image.h
    image.cpp
    shader.h
    shader.cpp
)

target_include_directories(somegi_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_core
    PUBLIC
        Vulkan::Vulkan
        glfw
        glm::glm
        vk-bootstrap::vk-bootstrap
)
target_compile_definitions(somegi_core PUBLIC
    GLFW_INCLUDE_VULKAN
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_RADIANS
)
```

---

## Task 3: core/Shader — 加载 spv，反射为 VkShaderModule

**Files:**
- Create: `src/core/shader.h`、`shader.cpp`

- [ ] **Step 1: shader.h**

```cpp
#pragma once
#include "vk_common.h"
#include <filesystem>
#include <vector>
#include <string>

namespace somegi {

class Device;

class ShaderModule {
public:
    ShaderModule() = default;
    ShaderModule(Device& d, const std::filesystem::path& spvPath);
    ~ShaderModule();

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ShaderModule(ShaderModule&&) noexcept;
    ShaderModule& operator=(ShaderModule&&) noexcept;

    VkShaderModule handle() const { return m_module; }

private:
    Device* m_device = nullptr;
    VkShaderModule m_module = VK_NULL_HANDLE;
};

std::filesystem::path shaderDir();

}
```

- [ ] **Step 2: shader.cpp**

```cpp
#include "shader.h"
#include "device.h"
#include <fstream>

namespace somegi {

ShaderModule::ShaderModule(Device& d, const std::filesystem::path& spvPath) : m_device(&d) {
    std::ifstream f(spvPath, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("shader open failed: " + spvPath.string());
    size_t size = f.tellg();
    if (size % 4) throw std::runtime_error("shader spv size not multiple of 4");
    std::vector<uint32_t> data(size / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), size);

    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = size;
    ci.pCode = data.data();
    VK_CHECK(vkCreateShaderModule(d.device(), &ci, nullptr, &m_module));
}

ShaderModule::~ShaderModule() {
    if (m_device && m_module) vkDestroyShaderModule(m_device->device(), m_module, nullptr);
}

ShaderModule::ShaderModule(ShaderModule&& o) noexcept
    : m_device(o.m_device), m_module(o.m_module) {
    o.m_device = nullptr; o.m_module = VK_NULL_HANDLE;
}
ShaderModule& ShaderModule::operator=(ShaderModule&& o) noexcept {
    if (this != &o) {
        if (m_device && m_module) vkDestroyShaderModule(m_device->device(), m_module, nullptr);
        m_device = o.m_device; m_module = o.m_module;
        o.m_device = nullptr; o.m_module = VK_NULL_HANDLE;
    }
    return *this;
}

std::filesystem::path shaderDir() {
    // SOMEGI_SHADER_DIR 由 CMake 传入；落在 build/shaders。
    return std::filesystem::path(SOMEGI_SHADER_DIR);
}

}
```

- [ ] **Step 3: 在 CMake 里把 build/shaders 路径作为 define 传给 app**

`src/app/CMakeLists.txt` 末尾加：

```cmake
target_compile_definitions(SomeGI PRIVATE
    SOMEGI_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
    SOMEGI_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
)
target_compile_definitions(somegi_core PRIVATE
    SOMEGI_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
    SOMEGI_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
)
```

(把 SOMEGI_SHADER_DIR 给 core 是为了 shader.cpp 编译时能 #ifdef 出来。)

- [ ] **Step 4: 提交**

```powershell
git add src/core/
git commit -m "core: buffer/image/shader RAII helpers"
```

构建一次确认编译通过：

```powershell
cmake -S . -B build
cmake --build build --config Debug
```

---

## Task 4: scene/Scene 类型 + Camera

**Files:**
- Create: `src/scene/CMakeLists.txt`
- Create: `src/scene/scene.h`
- Create: `src/scene/camera.h`、`camera.cpp`
- Modify: 顶层 `CMakeLists.txt`（add_subdirectory(src/scene)，src/app 链接 somegi_scene）

- [ ] **Step 1: src/scene/scene.h**

```cpp
#pragma once
#include "core/buffer.h"
#include "core/image.h"
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>

namespace somegi {

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;   // xyz=tangent, w=bitangent sign
    glm::vec2 uv0;
};

struct MaterialDesc {
    glm::vec4 baseColorFactor{1.0f};
    glm::vec3 emissiveFactor{0.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float normalScale = 1.0f;
    float occlusionStrength = 1.0f;
    float alphaCutoff = 0.5f;
    int baseColorTex = -1;     // index into Scene::textures, -1 = use white default
    int mrTex = -1;
    int normalTex = -1;
    int occlusionTex = -1;
    int emissiveTex = -1;
    uint32_t alphaMode = 0;    // 0 OPAQUE, 1 MASK, 2 BLEND
    uint32_t doubleSided = 0;
};

struct Primitive {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t  vertexOffset = 0;
    int32_t  materialIndex = -1;
};

struct Mesh {
    std::vector<Primitive> primitives;
};

struct Node {
    glm::mat4 worldTransform{1.0f};
    int meshIndex = -1;
};

struct TextureCpu {
    int width = 0, height = 0, channels = 4;
    std::vector<uint8_t> rgba;   // always RGBA8
    bool isSrgb = false;
};

struct SceneCpu {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Mesh> meshes;
    std::vector<Node> nodes;
    std::vector<MaterialDesc> materials;
    std::vector<TextureCpu> textures;
    glm::vec3 aabbMin{0}, aabbMax{0};
};

struct SceneGpu {
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer materialBuffer;          // MaterialGpu[] (SSBO)
    std::vector<Image> images;      // 与 SceneCpu::textures 对应
    Image whiteTex;                  // 1x1 白
    Image normalTex;                 // 1x1 (128,128,255,255)
    VkSampler linearSampler = VK_NULL_HANDLE;
    VkSampler linearSamplerSrgb = VK_NULL_HANDLE;
};

}
```

- [ ] **Step 2: src/scene/camera.h**

```cpp
#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace somegi {

class Camera {
public:
    glm::vec3 position{0, 1, 3};
    float yaw = -90.0f;     // degrees, looking -Z
    float pitch = 0.0f;
    float fovDeg = 60.0f;
    float nearZ = 0.05f;
    float farZ = 200.0f;

    glm::vec3 forward() const;
    glm::vec3 right() const;
    glm::vec3 up() const;

    glm::mat4 view() const;
    glm::mat4 proj(float aspect) const;
};

class FlyController {
public:
    void update(Camera& cam, float dtSec, struct GLFWwindow* window);
private:
    bool m_dragging = false;
    double m_lastX = 0, m_lastY = 0;
};

}
```

- [ ] **Step 3: src/scene/camera.cpp**

```cpp
#include "camera.h"
#include <GLFW/glfw3.h>
#include <glm/gtc/constants.hpp>

namespace somegi {

glm::vec3 Camera::forward() const {
    float y = glm::radians(yaw), p = glm::radians(pitch);
    return glm::normalize(glm::vec3(cos(y)*cos(p), sin(p), sin(y)*cos(p)));
}
glm::vec3 Camera::right() const {
    return glm::normalize(glm::cross(forward(), glm::vec3(0,1,0)));
}
glm::vec3 Camera::up() const {
    return glm::normalize(glm::cross(right(), forward()));
}
glm::mat4 Camera::view() const {
    return glm::lookAt(position, position + forward(), glm::vec3(0,1,0));
}
glm::mat4 Camera::proj(float aspect) const {
    auto p = glm::perspective(glm::radians(fovDeg), aspect, nearZ, farZ);
    p[1][1] *= -1.0f; // Vulkan Y flip
    return p;
}

void FlyController::update(Camera& cam, float dt, GLFWwindow* w) {
    int rmb = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT);
    double mx, my; glfwGetCursorPos(w, &mx, &my);
    if (rmb == GLFW_PRESS) {
        if (!m_dragging) { m_dragging = true; m_lastX = mx; m_lastY = my; }
        float dx = float(mx - m_lastX), dy = float(my - m_lastY);
        m_lastX = mx; m_lastY = my;
        cam.yaw   += dx * 0.15f;
        cam.pitch -= dy * 0.15f;
        cam.pitch = glm::clamp(cam.pitch, -89.0f, 89.0f);
    } else {
        m_dragging = false;
    }

    float speed = 3.0f;
    if (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed = 12.0f;
    glm::vec3 d{0};
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) d += cam.forward();
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) d -= cam.forward();
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) d -= cam.right();
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) d += cam.right();
    if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) d -= glm::vec3(0,1,0);
    if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) d += glm::vec3(0,1,0);
    if (glm::length(d) > 0) cam.position += glm::normalize(d) * speed * dt;
}

}
```

- [ ] **Step 4: src/scene/CMakeLists.txt（M1 末尾再补 gltf_loader/scene_gpu/upload）**

```cmake
add_library(somegi_scene STATIC
    scene.h
    camera.h
    camera.cpp
    upload.h
    upload.cpp
    gltf_loader.h
    gltf_loader.cpp
    scene_gpu.h
    scene_gpu.cpp
)

target_link_libraries(somegi_scene
    PUBLIC somegi_core
    PRIVATE cgltf stb_headers
)
```

- [ ] **Step 5: 顶层 CMakeLists.txt**

`add_subdirectory(src/core)` 后加：

```cmake
add_subdirectory(src/scene)
```

`src/app/CMakeLists.txt` 中：

```cmake
target_link_libraries(SomeGI PRIVATE somegi_core somegi_scene)
```

---

## Task 5: scene/Upload — 一次性 transfer 命令封装

**Files:**
- Create: `src/scene/upload.h`、`upload.cpp`

- [ ] **Step 1: upload.h**

```cpp
#pragma once
#include "core/vk_common.h"
#include <functional>

namespace somegi {
class Device;

void oneShotSubmit(Device& d, VkCommandPool pool, std::function<void(VkCommandBuffer)> body);

}
```

- [ ] **Step 2: upload.cpp**

```cpp
#include "upload.h"
#include "core/device.h"

namespace somegi {

void oneShotSubmit(Device& d, VkCommandPool pool, std::function<void(VkCommandBuffer)> body) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(d.device(), &ai, &cmd));

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    body(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = cmd;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1; si.pCommandBufferInfos = &csi;

    VK_CHECK(vkQueueSubmit2(d.graphicsQueue(), 1, &si, VK_NULL_HANDLE));
    vkQueueWaitIdle(d.graphicsQueue());
    vkFreeCommandBuffers(d.device(), pool, 1, &cmd);
}

}
```

---

## Task 6: scene/GltfLoader — cgltf 解析

**Files:**
- Create: `src/scene/gltf_loader.h`、`gltf_loader.cpp`

- [ ] **Step 1: gltf_loader.h**

```cpp
#pragma once
#include "scene.h"
#include <filesystem>

namespace somegi {

bool loadGltf(const std::filesystem::path& path, SceneCpu& outScene, std::string& outErr);

}
```

- [ ] **Step 2: gltf_loader.cpp**

```cpp
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "gltf_loader.h"
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <cstring>

namespace somegi {

static glm::mat4 toMat4(const cgltf_node* n) {
    if (n->has_matrix) return glm::make_mat4(n->matrix);
    glm::mat4 t = glm::translate(glm::mat4(1.0f),
                                 n->has_translation ? glm::vec3(n->translation[0],n->translation[1],n->translation[2]) : glm::vec3(0));
    glm::mat4 r(1.0f);
    if (n->has_rotation) {
        glm::quat q(n->rotation[3], n->rotation[0], n->rotation[1], n->rotation[2]);
        r = glm::mat4_cast(q);
    }
    glm::mat4 s = glm::scale(glm::mat4(1.0f),
                             n->has_scale ? glm::vec3(n->scale[0],n->scale[1],n->scale[2]) : glm::vec3(1));
    return t * r * s;
}

static glm::mat4 worldOf(const cgltf_node* n) {
    glm::mat4 m = toMat4(n);
    while (n->parent) { n = n->parent; m = toMat4(n) * m; }
    return m;
}

static int readImageRGBA(const cgltf_image* img, const std::filesystem::path& gltfDir,
                         const cgltf_data* data, TextureCpu& out, bool srgb) {
    out.isSrgb = srgb;
    if (img->buffer_view) {
        const auto* bv = img->buffer_view;
        const uint8_t* src = (const uint8_t*)bv->buffer->data + bv->offset;
        int w, h, c;
        stbi_uc* pix = stbi_load_from_memory(src, (int)bv->size, &w, &h, &c, 4);
        if (!pix) return -1;
        out.width = w; out.height = h; out.channels = 4;
        out.rgba.assign(pix, pix + w*h*4);
        stbi_image_free(pix);
        return 0;
    } else if (img->uri) {
        auto p = gltfDir / img->uri;
        int w, h, c;
        stbi_uc* pix = stbi_load(p.string().c_str(), &w, &h, &c, 4);
        if (!pix) return -1;
        out.width = w; out.height = h; out.channels = 4;
        out.rgba.assign(pix, pix + w*h*4);
        stbi_image_free(pix);
        return 0;
    }
    return -1;
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

    // Textures: 先把 image 转 RGBA8
    s.textures.resize(data->images_count);
    for (size_t i = 0; i < data->images_count; ++i) {
        // 默认 linear；后面材质引用时若是 base/emissive 改 srgb
        readImageRGBA(&data->images[i], dir, data, s.textures[i], /*srgb*/false);
    }

    auto texIndexFromTextureView = [&](const cgltf_texture_view& v, bool srgb) -> int {
        if (!v.texture || !v.texture->image) return -1;
        size_t idx = (size_t)(v.texture->image - data->images);
        if (srgb) s.textures[idx].isSrgb = true;
        return (int)idx;
    };

    // Materials
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
            d.baseColorTex = texIndexFromTextureView(pmr.base_color_texture, /*srgb*/true);
            d.mrTex = texIndexFromTextureView(pmr.metallic_roughness_texture, false);
        }
        d.normalTex = texIndexFromTextureView(m.normal_texture, false);
        d.normalScale = m.normal_texture.scale;
        d.occlusionTex = texIndexFromTextureView(m.occlusion_texture, false);
        d.occlusionStrength = m.occlusion_texture.scale;
        d.emissiveTex = texIndexFromTextureView(m.emissive_texture, true);
        d.emissiveFactor = glm::vec3(m.emissive_factor[0], m.emissive_factor[1], m.emissive_factor[2]);
        d.alphaCutoff = m.alpha_cutoff;
        d.alphaMode = (m.alpha_mode == cgltf_alpha_mode_mask) ? 1u :
                      (m.alpha_mode == cgltf_alpha_mode_blend) ? 2u : 0u;
        d.doubleSided = m.double_sided ? 1u : 0u;
        s.materials[i] = d;
    }

    // Meshes / Primitives — flatten 顶点 / 索引
    s.meshes.resize(data->meshes_count);
    glm::vec3 mn(FLT_MAX), mx(-FLT_MAX);
    for (size_t mi = 0; mi < data->meshes_count; ++mi) {
        auto& mesh = data->meshes[mi];
        Mesh& M = s.meshes[mi];
        M.primitives.resize(mesh.primitives_count);
        for (size_t pi = 0; pi < mesh.primitives_count; ++pi) {
            const auto& p = mesh.primitives[pi];
            Primitive prim{};
            prim.vertexOffset = (int32_t)s.vertices.size();
            prim.firstIndex = (uint32_t)s.indices.size();
            prim.materialIndex = p.material ? (int)(p.material - data->materials) : -1;

            const cgltf_accessor* posA = nullptr, *nrmA = nullptr, *tanA = nullptr, *uvA = nullptr;
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
                } else {
                    vx.normal = glm::vec3(0, 1, 0);
                }
                if (tanA) {
                    const float* T = accessorPtr<float>(tanA, v);
                    vx.tangent = glm::vec4(T[0], T[1], T[2], T[3]);
                } else {
                    vx.tangent = glm::vec4(1, 0, 0, 1);
                }
                if (uvA) {
                    const float* U = accessorPtr<float>(uvA, v);
                    vx.uv0 = glm::vec2(U[0], U[1]);
                } else {
                    vx.uv0 = glm::vec2(0);
                }
                s.vertices.push_back(vx);
                mn = glm::min(mn, vx.position);
                mx = glm::max(mx, vx.position);
            }

            // Indices
            if (p.indices) {
                size_t ic = p.indices->count;
                prim.indexCount = (uint32_t)ic;
                for (size_t i = 0; i < ic; ++i) {
                    uint32_t idx = (uint32_t)cgltf_accessor_read_index(p.indices, i);
                    s.indices.push_back(idx);
                }
            } else {
                prim.indexCount = (uint32_t)vc;
                for (uint32_t i = 0; i < vc; ++i) s.indices.push_back(i);
            }
            M.primitives[pi] = prim;
        }
    }
    s.aabbMin = mn; s.aabbMax = mx;

    // Nodes (扁平存世界变换)
    s.nodes.resize(data->nodes_count);
    for (size_t i = 0; i < data->nodes_count; ++i) {
        Node N{};
        N.worldTransform = worldOf(&data->nodes[i]);
        N.meshIndex = data->nodes[i].mesh ? (int)(data->nodes[i].mesh - data->meshes) : -1;
        s.nodes[i] = N;
    }

    cgltf_free(data);
    return true;
}

}
```

---

## Task 7: scene/SceneGpu — 上传到 GPU

**Files:**
- Create: `src/scene/scene_gpu.h`、`scene_gpu.cpp`

- [ ] **Step 1: scene_gpu.h**

```cpp
#pragma once
#include "scene.h"

namespace somegi {

class Device;

struct MaterialGpu {
    glm::vec4 baseColorFactor;
    glm::vec3 emissiveFactor;
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;
    float alphaCutoff;
    int baseColorTex, mrTex, normalTex, occlusionTex, emissiveTex;
    uint32_t alphaMode;
    uint32_t doubleSided;
    uint32_t _pad0;
};

void uploadScene(Device& d, VkCommandPool pool, const SceneCpu& cpu, SceneGpu& out);
void destroySceneSamplers(Device& d, SceneGpu& gpu);

}
```

- [ ] **Step 2: scene_gpu.cpp**

```cpp
#include "scene_gpu.h"
#include "core/device.h"
#include "core/buffer.h"
#include "upload.h"
#include <cstring>

namespace somegi {

static Buffer makeStaging(Device& d, const void* data, size_t size) {
    Buffer b(d, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(b.mapped(), data, size);
    return b;
}

static void uploadBuffer(Device& d, VkCommandPool pool,
                         const void* data, size_t size, VkBufferUsageFlags usage,
                         Buffer& out) {
    out = Buffer(d, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buffer staging = makeStaging(d, data, size);
    oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
        VkBufferCopy c{0, 0, size};
        vkCmdCopyBuffer(cmd, staging.handle(), out.handle(), 1, &c);
    });
}

static void transitionImg(VkCommandBuffer cmd, VkImage img,
                          VkImageLayout oldL, VkImageLayout newL,
                          VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAcc,
                          VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAcc,
                          uint32_t mipLevels = 1) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask=srcStage; b.srcAccessMask=srcAcc; b.dstStageMask=dstStage; b.dstAccessMask=dstAcc;
    b.oldLayout=oldL; b.newLayout=newL; b.image=img;
    b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount=1; di.pImageMemoryBarriers=&b;
    vkCmdPipelineBarrier2(cmd, &di);
}

static void uploadImage(Device& d, VkCommandPool pool, const TextureCpu& cpu, Image& out) {
    VkFormat fmt = cpu.isSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    ImageDesc id{};
    id.format = fmt;
    id.extent = {(uint32_t)cpu.width, (uint32_t)cpu.height, 1};
    id.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    out = Image(d, id);

    Buffer staging = makeStaging(d, cpu.rgba.data(), cpu.rgba.size());

    oneShotSubmit(d, pool, [&](VkCommandBuffer cmd) {
        transitionImg(cmd, out.image(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkBufferImageCopy c{};
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = id.extent;
        vkCmdCopyBufferToImage(cmd, staging.handle(), out.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        transitionImg(cmd, out.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    });
}

static Image makeSolid1x1(Device& d, VkCommandPool pool, glm::u8vec4 color, bool srgb) {
    TextureCpu c; c.width=1; c.height=1; c.channels=4; c.rgba={color.r,color.g,color.b,color.a}; c.isSrgb=srgb;
    Image img;
    uploadImage(d, pool, c, img);
    return img;
}

void uploadScene(Device& d, VkCommandPool pool, const SceneCpu& cpu, SceneGpu& out) {
    if (!cpu.vertices.empty())
        uploadBuffer(d, pool, cpu.vertices.data(), cpu.vertices.size()*sizeof(Vertex),
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     out.vertexBuffer);
    if (!cpu.indices.empty())
        uploadBuffer(d, pool, cpu.indices.data(), cpu.indices.size()*sizeof(uint32_t),
                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT, out.indexBuffer);

    // Material SSBO
    std::vector<MaterialGpu> mats;
    mats.reserve(cpu.materials.size());
    for (auto& m : cpu.materials) {
        MaterialGpu g{};
        g.baseColorFactor = m.baseColorFactor;
        g.emissiveFactor = m.emissiveFactor;
        g.metallicFactor = m.metallicFactor;
        g.roughnessFactor = m.roughnessFactor;
        g.normalScale = m.normalScale;
        g.occlusionStrength = m.occlusionStrength;
        g.alphaCutoff = m.alphaCutoff;
        g.baseColorTex = m.baseColorTex;
        g.mrTex = m.mrTex;
        g.normalTex = m.normalTex;
        g.occlusionTex = m.occlusionTex;
        g.emissiveTex = m.emissiveTex;
        g.alphaMode = m.alphaMode;
        g.doubleSided = m.doubleSided;
        mats.push_back(g);
    }
    if (mats.empty()) mats.push_back({});  // 总有 1 个
    uploadBuffer(d, pool, mats.data(), mats.size()*sizeof(MaterialGpu),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, out.materialBuffer);

    // Images
    out.images.resize(cpu.textures.size());
    for (size_t i = 0; i < cpu.textures.size(); ++i) {
        uploadImage(d, pool, cpu.textures[i], out.images[i]);
    }
    out.whiteTex  = makeSolid1x1(d, pool, {255,255,255,255}, /*srgb*/true);
    out.normalTex = makeSolid1x1(d, pool, {128,128,255,255}, /*srgb*/false);

    VkSamplerCreateInfo s{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    s.magFilter = VK_FILTER_LINEAR; s.minFilter = VK_FILTER_LINEAR;
    s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    s.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.maxLod = VK_LOD_CLAMP_NONE;
    s.anisotropyEnable = VK_FALSE;
    VK_CHECK(vkCreateSampler(d.device(), &s, nullptr, &out.linearSampler));
    out.linearSamplerSrgb = out.linearSampler;  // 简化：format 决定 srgb，sampler 不必区分
}

void destroySceneSamplers(Device& d, SceneGpu& gpu) {
    if (gpu.linearSampler) vkDestroySampler(d.device(), gpu.linearSampler, nullptr);
    gpu.linearSampler = VK_NULL_HANDLE;
}

}
```

---

## Task 8: shaders — 公共定义 + forward.slang

**Files:**
- Delete: `shaders/_smoke/smoke.slang`（M1 Task 1 的占位）
- Create: `shaders/common/shared_types.slang`
- Create: `shaders/common/pbr.slang`
- Create: `shaders/forward/forward.slang`
- Create: `shaders/tonemap/tonemap.slang`

- [ ] **Step 1: shared_types.slang**

```hlsl
// shaders/common/shared_types.slang

struct FrameUniforms {
    float4x4 view;
    float4x4 proj;
    float4x4 viewProj;
    float3   cameraPos;
    float    _p0;
    float3   sunDir;
    float    _p1;
    float3   sunColor;
    float    sunIntensity;
    float3   ambient;
    float    _p2;
    int      materialCount;
    int      _p3, _p4, _p5;
};

struct PushConsts {
    float4x4 model;
    int materialIndex;
    int _p0, _p1, _p2;
};

struct MaterialGpu {
    float4 baseColorFactor;
    float3 emissiveFactor;
    float  metallicFactor;
    float  roughnessFactor;
    float  normalScale;
    float  occlusionStrength;
    float  alphaCutoff;
    int    baseColorTex, mrTex, normalTex, occlusionTex, emissiveTex;
    uint   alphaMode;
    uint   doubleSided;
    uint   _pad0;
};
```

- [ ] **Step 2: pbr.slang**

```hlsl
// shaders/common/pbr.slang
import shared_types;

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
```

- [ ] **Step 3: forward.slang**

```hlsl
// shaders/forward/forward.slang
import shared_types;
import pbr;

[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> gFrame;
[[vk::binding(1, 0)]] StructuredBuffer<MaterialGpu> gMaterials;
[[vk::binding(2, 0)]] SamplerState gLinear;
// 贴图数组 (bindless-lite)，descriptorIndexing 启用即可
[[vk::binding(3, 0)]] Texture2D gTextures[];

[[vk::push_constant]] PushConsts gPC;

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
VsOut vs_main(VsIn i) {
    VsOut o;
    float4 wp = mul(gPC.model, float4(i.pos, 1.0));
    o.worldPos = wp.xyz;
    o.svPos = mul(gFrame.viewProj, wp);
    // 法线/切线变换：用 model 的 3x3 (假设 uniform scale，否则要 inverse-transpose)
    float3x3 n3 = (float3x3)gPC.model;
    o.normal  = normalize(mul(n3, i.normal));
    o.tangent = float4(normalize(mul(n3, i.tangent.xyz)), i.tangent.w);
    o.uv = i.uv0;
    o.matIndex = gPC.materialIndex;
    return o;
}

float3 sampleAlbedo(int idx, float2 uv) {
    if (idx < 0) return float3(1);
    return gTextures[idx].Sample(gLinear, uv).rgb;
}
float4 sampleAlbedoA(int idx, float2 uv) {
    if (idx < 0) return float4(1);
    return gTextures[idx].Sample(gLinear, uv);
}
float sampleR(int idx, float2 uv, float def) {
    if (idx < 0) return def;
    return gTextures[idx].Sample(gLinear, uv).r;
}
float3 sampleNormal(int idx, float2 uv) {
    if (idx < 0) return float3(0.5, 0.5, 1.0);
    return gTextures[idx].Sample(gLinear, uv).rgb;
}

[shader("pixel")]
float4 ps_main(VsOut i) : SV_Target {
    MaterialGpu m = gMaterials[i.matIndex];

    float4 baseTex = sampleAlbedoA(m.baseColorTex, i.uv);
    float4 base = m.baseColorFactor * baseTex;
    if (m.alphaMode == 1u && base.a < m.alphaCutoff) discard;

    float3 mr = float3(0, m.roughnessFactor, m.metallicFactor);
    if (m.mrTex >= 0) {
        float3 t = gTextures[m.mrTex].Sample(gLinear, i.uv).rgb;
        // glTF: R=occlusion (sometimes packed), G=roughness, B=metallic
        mr.y *= t.g; mr.z *= t.b;
    }
    float metallic = saturate(mr.z);
    float roughness = saturate(mr.y);

    // Normal mapping
    float3 N = normalize(i.normal);
    if (m.normalTex >= 0) {
        float3 nm = gTextures[m.normalTex].Sample(gLinear, i.uv).rgb * 2.0 - 1.0;
        nm.xy *= m.normalScale;
        float3 T = normalize(i.tangent.xyz);
        float3 B = normalize(cross(N, T) * i.tangent.w);
        N = normalize(T*nm.x + B*nm.y + N*nm.z);
    }

    float3 V = normalize(gFrame.cameraPos - i.worldPos);
    float3 L = normalize(-gFrame.sunDir);
    float3 sun = directBRDF(N, V, L, base.rgb, metallic, roughness) *
                 gFrame.sunColor * gFrame.sunIntensity;

    float occ = sampleR(m.occlusionTex, i.uv, 1.0);
    occ = lerp(1.0, occ, m.occlusionStrength);
    float3 amb = gFrame.ambient * base.rgb * occ * (1.0 - metallic);

    float3 emi = m.emissiveFactor;
    if (m.emissiveTex >= 0) emi *= gTextures[m.emissiveTex].Sample(gLinear, i.uv).rgb;

    return float4(sun + amb + emi, base.a);
}
```

- [ ] **Step 4: tonemap.slang**

```hlsl
// shaders/tonemap/tonemap.slang
[[vk::binding(0, 0)]] Texture2D gHdr;
[[vk::binding(1, 0)]] SamplerState gLinear;
[[vk::binding(2, 0)]] RWTexture2D<float4> gLdr;

float3 acesApprox(float3 c) {
    c *= 0.6;
    float a = 2.51, b = 0.03, c2 = 2.43, d = 0.59, e = 0.14;
    return saturate((c*(a*c+b)) / (c*(c2*c+d)+e));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 dt : SV_DispatchThreadID) {
    uint2 dim;
    gLdr.GetDimensions(dim.x, dim.y);
    if (any(dt.xy >= dim)) return;
    float2 uv = (float2(dt.xy) + 0.5) / float2(dim);
    float3 hdr = gHdr.SampleLevel(gLinear, uv, 0).rgb;
    float3 ldr = acesApprox(hdr);
    // approx linear→srgb
    ldr = pow(saturate(ldr), 1.0/2.2);
    gLdr[dt.xy] = float4(ldr, 1.0);
}
```

- [ ] **Step 5: 删除 smoke.slang，重新构建**

```powershell
Remove-Item shaders/_smoke/smoke.slang
cmake --build build --config Debug --target somegi_shaders
```

Expected：`build/shaders/forward/forward.spv` 与 `build/shaders/tonemap/tonemap.spv` 生成。

---

## Task 9: renderer/RenderTargets — HDR + depth + tonemap dst

**Files:**
- Create: `src/renderer/CMakeLists.txt`
- Create: `src/renderer/render_targets.h`、`render_targets.cpp`
- Modify: 顶层 CMakeLists（add_subdirectory + 链接）

- [ ] **Step 1: render_targets.h**

```cpp
#pragma once
#include "core/image.h"

namespace somegi {

class Device;

struct RenderTargets {
    Image hdrColor;     // R16G16B16A16_SFLOAT
    Image depth;        // D32_SFLOAT
    Image ldrTonemap;   // 与 swapchain 同尺寸的中间 LDR（B8G8R8A8_UNORM, STORAGE+TRANSFER_SRC）

    void create(Device& d, VkExtent2D ext);
    void destroy();

    VkExtent2D extent{};
};

}
```

- [ ] **Step 2: render_targets.cpp**

```cpp
#include "render_targets.h"
#include "core/device.h"

namespace somegi {

void RenderTargets::create(Device& d, VkExtent2D ext) {
    extent = ext;

    ImageDesc hdr{};
    hdr.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    hdr.extent = {ext.width, ext.height, 1};
    hdr.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    hdrColor = Image(d, hdr);

    ImageDesc dep{};
    dep.format = VK_FORMAT_D32_SFLOAT;
    dep.extent = {ext.width, ext.height, 1};
    dep.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    dep.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depth = Image(d, dep);

    ImageDesc ldr{};
    ldr.format = VK_FORMAT_B8G8R8A8_UNORM;
    ldr.extent = {ext.width, ext.height, 1};
    ldr.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ldrTonemap = Image(d, ldr);
}

void RenderTargets::destroy() {
    hdrColor.reset();
    depth.reset();
    ldrTonemap.reset();
}

}
```

---

## Task 10: renderer/ForwardPass — pipeline + 描述符 + draw

**Files:**
- Create: `src/renderer/forward_pass.h`、`forward_pass.cpp`

简化设计：
- Set 0：FrameUBO + materialSSBO + sampler + 贴图数组（descriptorIndexing/runtimeDescriptorArray）
- Push Constant：model + materialIndex
- 顶点输入：pos vec3, normal vec3, tangent vec4, uv vec2
- 渲染目标：HDR color (R16G16B16A16_SFLOAT) + Depth (D32_SFLOAT)，dynamic rendering

代码骨架：

```cpp
#pragma once
#include "core/buffer.h"
#include "core/shader.h"
#include "scene/scene.h"
#include "render_targets.h"
#include <glm/glm.hpp>

namespace somegi {
class Device;

struct FrameUBO {
    glm::mat4 view, proj, viewProj;
    glm::vec4 cameraPos;          // xyz + pad
    glm::vec4 sunDir;             // xyz + pad
    glm::vec4 sunColor_intensity; // xyz=color, w=intensity
    glm::vec4 ambient;
    glm::ivec4 counts;            // x=materialCount
};

class ForwardPass {
public:
    void init(Device& d, VkFormat colorFmt, VkFormat depthFmt, uint32_t maxTextures);
    void destroy();

    void bindScene(Device& d, const SceneGpu& gpu, uint32_t materialCount, uint32_t textureCount);
    void updateFrame(const FrameUBO& ubo);

    void record(VkCommandBuffer cmd, const RenderTargets& rt,
                const SceneCpu& cpu, const SceneGpu& gpu);

private:
    Device* m_device = nullptr;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    Buffer m_frameUbo;            // host-visible
    uint32_t m_maxTextures = 0;
};

}
```

实现内容（核心，删减自我注释）：

`forward_pass.cpp`：

```cpp
#include "forward_pass.h"
#include "core/device.h"
#include "scene/scene_gpu.h"
#include <array>
#include <cstring>

namespace somegi {

void ForwardPass::init(Device& d, VkFormat colorFmt, VkFormat depthFmt, uint32_t maxTextures) {
    m_device = &d; m_maxTextures = maxTextures;

    // Descriptor set layout
    std::array<VkDescriptorSetLayoutBinding, 4> b{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLER,         1, VK_SHADER_STAGE_FRAGMENT_BIT};
    b[3] = {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,   maxTextures, VK_SHADER_STAGE_FRAGMENT_BIT};

    std::array<VkDescriptorBindingFlags, 4> bf{0,0,0,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT};
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bfci.bindingCount = (uint32_t)bf.size(); bfci.pBindingFlags = bf.data();

    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.pNext = &bfci;
    li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.device(), &li, nullptr, &m_setLayout));

    // Pipeline layout（push constant: 64 + 16 = 80 bytes）
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pc.size = 64 + 16;

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(d.device(), &plci, nullptr, &m_pipelineLayout));

    // Shaders
    auto sd = shaderDir();
    ShaderModule vs(d, sd / "forward" / "forward.spv");
    ShaderModule fs(d, sd / "forward" / "forward.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs.handle(); stages[0].pName = "vs_main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs.handle(); stages[1].pName = "ps_main";

    // Vertex input
    VkVertexInputBindingDescription vib{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription, 4> via{};
    via[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(Vertex,position)};
    via[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(Vertex,normal)};
    via[2] = {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex,tangent)};
    via[3] = {3, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(Vertex,uv0)};

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = (uint32_t)via.size(); vi.pVertexAttributeDescriptions = via.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;  // reverse-Z 暂不用：用 LE 即可
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &ba;

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyni{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyni.dynamicStateCount = 2; dyni.pDynamicStates = dyn;

    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1; rci.pColorAttachmentFormats = &colorFmt;
    rci.depthAttachmentFormat = depthFmt;

    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.pNext = &rci;
    gpci.stageCount = 2; gpci.pStages = stages;
    gpci.pVertexInputState = &vi; gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp; gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms; gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb; gpci.pDynamicState = &dyni;
    gpci.layout = m_pipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(d.device(), VK_NULL_HANDLE, 1, &gpci, nullptr, &m_pipeline));

    // Descriptor pool & set
    std::array<VkDescriptorPoolSize, 4> ps{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, maxTextures},
    }};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = (uint32_t)ps.size(); pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(d.device(), &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_setLayout;
    VK_CHECK(vkAllocateDescriptorSets(d.device(), &dai, &m_set));

    // FrameUBO
    m_frameUbo = Buffer(d, sizeof(FrameUBO),
                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void ForwardPass::destroy() {
    auto dev = m_device->device();
    if (m_pool) vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_pipeline) vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    if (m_setLayout) vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
    m_frameUbo.reset();
}

void ForwardPass::bindScene(Device& d, const SceneGpu& gpu, uint32_t materialCount, uint32_t textureCount) {
    // Update set: UBO + materials SSBO + sampler + texture array
    VkDescriptorBufferInfo uboInfo{m_frameUbo.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo matInfo{gpu.materialBuffer.handle(), 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = gpu.linearSampler;

    std::vector<VkDescriptorImageInfo> imgs;
    imgs.reserve(m_maxTextures);
    for (uint32_t i = 0; i < m_maxTextures; ++i) {
        VkDescriptorImageInfo ii{};
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        if (i < textureCount) ii.imageView = gpu.images[i].view();
        else                  ii.imageView = gpu.whiteTex.view();
        imgs.push_back(ii);
    }

    std::array<VkWriteDescriptorSet, 4> w{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = m_set; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &uboInfo;
    w[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[1].dstSet = m_set; w[1].dstBinding = 1; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &matInfo;
    w[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[2].dstSet = m_set; w[2].dstBinding = 2; w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w[2].pImageInfo = &samplerInfo;
    w[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[3].dstSet = m_set; w[3].dstBinding = 3; w[3].descriptorCount = m_maxTextures;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w[3].pImageInfo = imgs.data();

    vkUpdateDescriptorSets(d.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void ForwardPass::updateFrame(const FrameUBO& ubo) {
    std::memcpy(m_frameUbo.mapped(), &ubo, sizeof(FrameUBO));
}

void ForwardPass::record(VkCommandBuffer cmd, const RenderTargets& rt,
                         const SceneCpu& cpu, const SceneGpu& gpu) {
    VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = rt.hdrColor.view();
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{0.02f, 0.02f, 0.04f, 1.0f}};

    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageView = rt.depth.view();
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0,0}, rt.extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1; ri.pColorAttachments = &color;
    ri.pDepthAttachment = &depth;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{0,0, (float)rt.extent.width, (float)rt.extent.height, 0, 1};
    VkRect2D sc{{0,0}, rt.extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipelineLayout, 0, 1, &m_set, 0, nullptr);
    VkDeviceSize zero = 0;
    VkBuffer vb = gpu.vertexBuffer.handle();
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &zero);
    vkCmdBindIndexBuffer(cmd, gpu.indexBuffer.handle(), 0, VK_INDEX_TYPE_UINT32);

    // 遍历 nodes -> meshes -> primitives
    struct PC {
        glm::mat4 model;
        int materialIndex;
        int p0, p1, p2;
    } pc;
    for (auto& n : cpu.nodes) {
        if (n.meshIndex < 0) continue;
        const Mesh& M = cpu.meshes[n.meshIndex];
        pc.model = n.worldTransform;
        for (auto& p : M.primitives) {
            pc.materialIndex = p.materialIndex >= 0 ? p.materialIndex : 0;
            vkCmdPushConstants(cmd, m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(PC), &pc);
            vkCmdDrawIndexed(cmd, p.indexCount, 1, p.firstIndex, p.vertexOffset, 0);
        }
    }

    vkCmdEndRendering(cmd);
}

}
```

---

## Task 11: renderer/TonemapPass — compute hdr → ldr

实现要点（与 Task 10 同模式）：
- compute pipeline，shader 是 tonemap.spv
- Set 0: SAMPLED_IMAGE (binding 0) + SAMPLER (binding 1) + STORAGE_IMAGE (binding 2)
- record: dispatch ceil(w/8) × ceil(h/8)
- record 前后做 layout transition：hdrColor → SHADER_READ_ONLY_OPTIMAL, ldrTonemap → GENERAL

代码省略，与 forward_pass 同结构。在实现时按 ForwardPass 的模板写。

---

## Task 12: app/App 集成 + cube 测试

修改 `src/app/app.h` 与 `app.cpp`：
- 增加 SceneCpu/SceneGpu/Camera/RenderTargets/ForwardPass/TonemapPass 成员
- 启动时加载 cube.gltf → 上传 GPU
- 主循环：
  1. dt 计算、camera 控制器更新
  2. updateFrame UBO（view/proj/sun）
  3. acquire swapchain
  4. transition hdrColor → COLOR_ATTACHMENT_OPTIMAL, depth → DEPTH_ATTACHMENT_OPTIMAL
  5. forwardPass.record
  6. transition hdrColor → SHADER_READ_ONLY_OPTIMAL，ldrTonemap → GENERAL
  7. tonemapPass.record
  8. transition ldrTonemap → TRANSFER_SRC, swapchainImage → TRANSFER_DST
  9. vkCmdBlitImage（ldrTonemap → swapchainImage）
  10. transition swapchainImage → PRESENT_SRC_KHR
  11. submit + present

---

## Task 13: 验证 cube

- [ ] 构建 + 跑 SomeGI.exe
- [ ] 命令行（或硬编码）加载 `assets/gltf/cube/cube.gltf`
- [ ] 看到立方体（可能旋转或静止），有方向光阴影/明暗变化
- [ ] WASD + 鼠标右键能移动相机
- [ ] 关闭程序无 validation 错误

如果 cube 不渲染：
1. 截 RenderDoc 帧分析
2. 检查 vertex/index buffer 上传
3. 检查 push constant 大小、UBO 内容
4. 退路：tonemap 阶段直接把 (1,0,1) 灌入 ldrTonemap，确认整条 blit/present 链路 OK

---

## Task 14: Sponza 测试

- [ ] 切换到 `assets/gltf/Sponza/Sponza.gltf`
- [ ] 加载时间 < 5 秒（67 张贴图，~50MB）
- [ ] 看到完整 Sponza 内部场景，有材质细节、法线
- [ ] 多个 mesh、多种材质
- [ ] 帧率在 GTX 30/RTX 40 系应该 60fps+

如果 Sponza 不渲染或 crash：
- maxTextures 是否设够（Sponza 有 ~67 张，设 maxTextures=128 留余量）
- 顶点数足够大时 vertexOffset 用 int32 可能正负溢出？检查 prim.vertexOffset 类型
- 黑漆漆？加 `gFrame.ambient = (0.1, 0.1, 0.1)` 看是否纯几何问题

---

## 自检（M1 完成判据）

- [x] cube.gltf 显示，能看出 PBR 着色（金属高光、粗糙度变化）
- [x] Sponza.gltf 显示，纹理 + 法线贴图正确
- [x] WASD/鼠标右键能漫游
- [x] 60 fps+ 流畅
- [x] 关闭程序 validation 干净

完成进入 M2（ImGui + UI 切换）。
