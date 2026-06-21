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
    # Phase 4: Context (命令录制上下文)
    vulkan/vk_context.cpp
)

# D3D12 后端（仅 Windows）
if(WIN32)
    target_sources(somegi_rhi PRIVATE
        d3d12/d3d12_device.cpp
        d3d12/d3d12_buffer.cpp
        d3d12/d3d12_texture.cpp
        d3d12/d3d12_sampler.cpp
        d3d12/d3d12_shader.cpp
        d3d12/d3d12_pso.cpp
        d3d12/d3d12_descriptor.cpp
        d3d12/d3d12_swapchain.cpp
        d3d12/d3d12_command.cpp
        d3d12/d3d12_fence.cpp
        d3d12/d3d12_query_pool.cpp
        d3d12/d3d12_acceleration_structure.cpp
    )
    # 启用 D3D12 Agility SDK（支持最新 D3D12 特性）
    target_compile_definitions(somegi_rhi PUBLIC D3D12_SUPPORTED)
endif()

target_include_directories(somegi_rhi PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_rhi PUBLIC somegi_core)

# D3D12 链接库
if(WIN32)
    target_link_libraries(somegi_rhi PUBLIC d3d12 dxgi d3dcompiler)
endif()
