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
target_compile_definitions(somegi_core
    PUBLIC
        GLFW_INCLUDE_VULKAN
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_FORCE_RADIANS
    PRIVATE
        SOMEGI_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
        SOMEGI_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
)
