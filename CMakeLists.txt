cmake_minimum_required(VERSION 3.24)
project(SomeGI CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus /utf-8 /MP)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

find_package(Vulkan REQUIRED)
message(STATUS "Vulkan_INCLUDE_DIR: ${Vulkan_INCLUDE_DIR}")
message(STATUS "Vulkan_LIBRARY    : ${Vulkan_LIBRARY}")

get_filename_component(VULKAN_SDK_BIN "${Vulkan_LIBRARY}" DIRECTORY)
get_filename_component(VULKAN_SDK_BIN "${VULKAN_SDK_BIN}/../Bin" ABSOLUTE)
find_program(SLANGC_EXE slangc HINTS ${VULKAN_SDK_BIN})
if(NOT SLANGC_EXE)
    message(WARNING "slangc not found, shader compilation disabled until M1")
else()
    message(STATUS "slangc            : ${SLANGC_EXE}")
endif()

include(cmake/HeaderOnly.cmake)

# ---- Third-party ----
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

add_subdirectory(external/glm)

set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(external/vk-bootstrap)

add_header_only_library(cgltf ${CMAKE_SOURCE_DIR}/external/cgltf)
add_header_only_library(stb_headers ${CMAKE_SOURCE_DIR}/external/stb)

add_subdirectory(external/imgui_local)

# ---- Assets (download missing glTF test models from Khronos glTF-Sample-Models) ----
set(GLTF_SAMPLE_MODELS "")

set(CUBE_GLTF_DIR "${CMAKE_SOURCE_DIR}/assets/gltf/cube")
if(NOT EXISTS "${CUBE_GLTF_DIR}/cube.gltf")
    list(APPEND GLTF_SAMPLE_MODELS "2.0/Cube")
endif()

set(SPONZA_GLTF_DIR "${CMAKE_SOURCE_DIR}/assets/gltf/Sponza")
if(NOT EXISTS "${SPONZA_GLTF_DIR}/Sponza.gltf")
    list(APPEND GLTF_SAMPLE_MODELS "2.0/Sponza")
endif()

set(HELMET_GLTF_DIR "${CMAKE_SOURCE_DIR}/assets/gltf/DamagedHelmet")
if(NOT EXISTS "${HELMET_GLTF_DIR}/DamagedHelmet.gltf")
    list(APPEND GLTF_SAMPLE_MODELS "2.0/DamagedHelmet")
endif()


if(GLTF_SAMPLE_MODELS)
    message(STATUS "Downloading missing glTF models: ${GLTF_SAMPLE_MODELS} ...")
    set(GLTF_TMP "${CMAKE_BINARY_DIR}/_gltf_sample_tmp")
    file(MAKE_DIRECTORY "${GLTF_TMP}")
    execute_process(
        COMMAND git clone --depth 1 --filter=blob:none --sparse
                https://github.com/KhronosGroup/glTF-Sample-Models.git .
        WORKING_DIRECTORY "${GLTF_TMP}"
        RESULT_VARIABLE gltf_clone_res
        OUTPUT_VARIABLE gltf_out
        ERROR_VARIABLE  gltf_err
    )
    if(gltf_clone_res EQUAL 0)
        execute_process(
            COMMAND git sparse-checkout set ${GLTF_SAMPLE_MODELS}
            WORKING_DIRECTORY "${GLTF_TMP}"
            RESULT_VARIABLE gltf_sparse_res
        )
        if(gltf_sparse_res EQUAL 0)
            if("2.0/Cube" IN_LIST GLTF_SAMPLE_MODELS)
                file(COPY "${GLTF_TMP}/2.0/Cube/glTF/" DESTINATION "${CUBE_GLTF_DIR}")
                message(STATUS "Cube model restored to ${CUBE_GLTF_DIR}")
            endif()
            if("2.0/Sponza" IN_LIST GLTF_SAMPLE_MODELS)
                file(COPY "${GLTF_TMP}/2.0/Sponza/glTF/" DESTINATION "${SPONZA_GLTF_DIR}")
                message(STATUS "Sponza model restored to ${SPONZA_GLTF_DIR}")
            endif()
            if("2.0/DamagedHelmet" IN_LIST GLTF_SAMPLE_MODELS)
                file(COPY "${GLTF_TMP}/2.0/DamagedHelmet/glTF/" DESTINATION "${HELMET_GLTF_DIR}")
                message(STATUS "DamagedHelmet model restored to ${HELMET_GLTF_DIR}")
            endif()
        endif()
    else()
        message(WARNING "glTF sample models clone failed: ${gltf_err}")
    endif()
    file(REMOVE_RECURSE "${GLTF_TMP}")
endif()

# ---- Shaders ----
include(cmake/CompileSlang.cmake)
file(GLOB_RECURSE SHADER_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/shaders/*.slang
)
# common/ contains import-only modules (no entry points) — exclude.
list(FILTER SHADER_SOURCES EXCLUDE REGEX "/shaders/common/")
if(SHADER_SOURCES AND SLANGC_EXE)
    compile_slang(somegi_shaders SOURCES ${SHADER_SOURCES})
    # 需要 SPIR-V BaseInstance 补丁的 shader（顶点着色器使用 SV_InstanceID 的）
    set(SPV_PATCH_LIST
        gbuffer/gbuffer.spv
        forward/forward.spv
        gi/rsm/rsm_geometry.spv
        forward/forward_ibl.spv
    )
    add_custom_command(TARGET somegi_shaders POST_BUILD
        COMMAND python "${CMAKE_SOURCE_DIR}/cmake/patch_spv.py" ${SPV_PATCH_LIST}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/shaders"
        COMMENT "Patching SPIR-V InstanceIndex"
    )

endif()

# ---- glslang mesh shader compilation (Slang doesn't emit OpSetMeshOutputsEXT) ----
find_program(GLSLANG_EXE glslangValidator HINTS ${VULKAN_SDK_BIN})
if(GLSLANG_EXE)
    set(MESH_SHADER_DIR "${CMAKE_BINARY_DIR}/shaders/gbuffer")
    file(MAKE_DIRECTORY ${MESH_SHADER_DIR})
    add_custom_command(
        OUTPUT "${MESH_SHADER_DIR}/gbuffer_mesh_no_task_mesh.spv"
        COMMAND ${GLSLANG_EXE} --target-env vulkan1.3 -S mesh
            "${CMAKE_SOURCE_DIR}/shaders/gbuffer/gbuffer_mesh_no_task.glsl"
            -o "${MESH_SHADER_DIR}/gbuffer_mesh_no_task_mesh.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/gbuffer/gbuffer_mesh_no_task.glsl"
        COMMENT "glslang mesh: gbuffer_mesh_no_task.glsl"
    )
    add_custom_command(
        OUTPUT "${MESH_SHADER_DIR}/gbuffer_mesh_no_task_frag.spv"
        COMMAND ${GLSLANG_EXE} --target-env vulkan1.3 -S frag
            "${CMAKE_SOURCE_DIR}/shaders/gbuffer/gbuffer_mesh_no_task.frag.glsl"
            -o "${MESH_SHADER_DIR}/gbuffer_mesh_no_task_frag.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/gbuffer/gbuffer_mesh_no_task.frag.glsl"
        COMMENT "glslang frag: gbuffer_mesh_no_task.frag.glsl"
    )
    set(FORWARD_MESH_DIR "${CMAKE_BINARY_DIR}/shaders/forward")
    file(MAKE_DIRECTORY ${FORWARD_MESH_DIR})
    add_custom_command(
        OUTPUT "${FORWARD_MESH_DIR}/forward_mesh_no_task_mesh.spv"
        COMMAND ${GLSLANG_EXE} --target-env vulkan1.3 -S mesh
            "${CMAKE_SOURCE_DIR}/shaders/forward/forward_mesh_no_task.glsl"
            -o "${FORWARD_MESH_DIR}/forward_mesh_no_task_mesh.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/forward/forward_mesh_no_task.glsl"
        COMMENT "glslang mesh: forward_mesh_no_task.glsl"
    )
    add_custom_command(
        OUTPUT "${FORWARD_MESH_DIR}/forward_mesh_no_task_frag.spv"
        COMMAND ${GLSLANG_EXE} --target-env vulkan1.3 -S frag
            "${CMAKE_SOURCE_DIR}/shaders/forward/forward_mesh_no_task.frag.glsl"
            -o "${FORWARD_MESH_DIR}/forward_mesh_no_task_frag.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/forward/forward_mesh_no_task.frag.glsl"
        COMMENT "glslang frag: forward_mesh_no_task.frag.glsl"
    )
    # Task Shader
    add_custom_command(
        OUTPUT "${MESH_SHADER_DIR}/gbuffer_task.spv"
        COMMAND ${GLSLANG_EXE} --target-env vulkan1.3 -S task
            "${CMAKE_SOURCE_DIR}/shaders/gbuffer/gbuffer_task.glsl"
            -o "${MESH_SHADER_DIR}/gbuffer_task.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/gbuffer/gbuffer_task.glsl"
        COMMENT "glslang task: gbuffer_task.glsl"
    )
    # Mesh Shader with Task Payload
    add_custom_command(
        OUTPUT "${MESH_SHADER_DIR}/gbuffer_mesh_task_mesh.spv"
        COMMAND ${GLSLANG_EXE} --target-env vulkan1.3 -S mesh
            "${CMAKE_SOURCE_DIR}/shaders/gbuffer/gbuffer_mesh_task_mesh.glsl"
            -o "${MESH_SHADER_DIR}/gbuffer_mesh_task_mesh.spv"
        DEPENDS "${CMAKE_SOURCE_DIR}/shaders/gbuffer/gbuffer_mesh_task_mesh.glsl"
        COMMENT "glslang mesh: gbuffer_mesh_task_mesh.glsl"
    )
    add_custom_target(glslang_mesh_shaders DEPENDS
        "${MESH_SHADER_DIR}/gbuffer_mesh_no_task_mesh.spv"
        "${MESH_SHADER_DIR}/gbuffer_mesh_no_task_frag.spv"
        "${MESH_SHADER_DIR}/gbuffer_task.spv"
        "${MESH_SHADER_DIR}/gbuffer_mesh_task_mesh.spv"
        "${FORWARD_MESH_DIR}/forward_mesh_no_task_mesh.spv"
        "${FORWARD_MESH_DIR}/forward_mesh_no_task_frag.spv"
    )
    if(TARGET somegi_shaders)
        add_dependencies(somegi_shaders glslang_mesh_shaders)
    endif()
    if(TARGET SomeGI)
        add_dependencies(SomeGI glslang_mesh_shaders)
    endif()
    message(STATUS "glslangValidator   : ${GLSLANG_EXE}")
else()
    message(WARNING "glslangValidator not found, mesh shader compilation disabled")
endif()

# ---- Project ----
add_subdirectory(src/core)
add_subdirectory(src/scene)
add_subdirectory(src/gi)
add_subdirectory(src/renderer)
add_subdirectory(src/app)
