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
    add_custom_command(TARGET somegi_shaders POST_BUILD
        COMMAND python "${CMAKE_SOURCE_DIR}/cmake/patch_spv.py"
            "${CMAKE_BINARY_DIR}/shaders/gbuffer/gbuffer.spv"
            "${CMAKE_BINARY_DIR}/shaders/forward/forward.spv"
            "${CMAKE_BINARY_DIR}/shaders/gi/rsm/rsm_geometry.spv"
        COMMENT "Patching SPIR-V InstanceIndex"
    )

endif()

# ---- Project ----
add_subdirectory(src/core)
add_subdirectory(src/scene)
add_subdirectory(src/gi)
add_subdirectory(src/renderer)
add_subdirectory(src/app)
