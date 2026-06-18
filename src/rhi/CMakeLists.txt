# src/rhi/CMakeLists.txt
add_library(somegi_rhi OBJECT
    # Phase 1: 核心资源
    vulkan/vk_device.cpp
    vulkan/vk_buffer.cpp
    vulkan/vk_texture.cpp
    vulkan/vk_shader.cpp
    vulkan/vk_sampler.cpp
    vulkan/vk_acceleration_structure.cpp
    vulkan/vk_swapchain.cpp
    # Phase 2: PSO + Descriptor
    vulkan/vk_pso.cpp
    vulkan/vk_descriptor.cpp
    # Phase 3: Command Buffer + Fence + Query
    vulkan/vk_command.cpp
    vulkan/vk_fence.cpp
    vulkan/vk_query_pool.cpp
)
target_include_directories(somegi_rhi PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_rhi PUBLIC somegi_core)
