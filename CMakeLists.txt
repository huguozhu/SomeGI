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

# ---- Shaders ----
include(cmake/CompileSlang.cmake)
file(GLOB_RECURSE SHADER_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/shaders/*.slang
)
# common/ contains import-only modules (no entry points) — exclude.
list(FILTER SHADER_SOURCES EXCLUDE REGEX "/shaders/common/")
if(SHADER_SOURCES AND SLANGC_EXE)
    compile_slang(somegi_shaders SOURCES ${SHADER_SOURCES})
endif()

# ---- Project ----
add_subdirectory(src/core)
add_subdirectory(src/scene)
add_subdirectory(src/gi)
add_subdirectory(src/renderer)
add_subdirectory(src/app)
