add_library(somegi_core STATIC
    vk_common.h
    window.h
    window.cpp
    device.h
    device.cpp
    swapchain.h
    swapchain.cpp
    buffer.h
    buffer.cpp
    image.h
    image.cpp
    vma_impl.cpp
    shader.h
    shader.cpp
)

target_include_directories(somegi_core PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/external)
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
