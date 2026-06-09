add_library(somegi_renderer_vxgi OBJECT
    vxgi_resources.cpp
    vxgi_voxelize_pass.cpp
    vxgi_inject_pass.cpp
    vxgi_mipmap_pass.cpp
    vxgi_aniso_pass.cpp
    vxgi_relight_pass.cpp
    vxgi_resolve_6axis_pass.cpp
)
target_include_directories(somegi_renderer_vxgi PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_vxgi PUBLIC somegi_core)
