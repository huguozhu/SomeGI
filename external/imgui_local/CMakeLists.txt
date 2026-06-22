add_library(imgui STATIC
    ../imgui/imgui.cpp
    ../imgui/imgui_demo.cpp
    ../imgui/imgui_draw.cpp
    ../imgui/imgui_tables.cpp
    ../imgui/imgui_widgets.cpp
    ../imgui/backends/imgui_impl_glfw.cpp
    ../imgui/backends/imgui_impl_vulkan.cpp
)

# D3D12 后端（仅 Windows）
if(WIN32)
    target_sources(imgui PRIVATE ../imgui/backends/imgui_impl_dx12.cpp)
endif()

target_include_directories(imgui
    PUBLIC
        ../imgui
        ../imgui/backends
)

target_link_libraries(imgui
    PUBLIC Vulkan::Vulkan glfw
)

# D3D12 链接（仅 Windows）
if(WIN32)
    target_link_libraries(imgui PUBLIC d3d12 dxgi)
endif()
