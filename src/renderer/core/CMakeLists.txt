add_library(somegi_renderer_core OBJECT
    render_targets.cpp
    barrier_manager.cpp
    render_pipeline.cpp
    mesh_pipeline.cpp
    gbuffer_pass.cpp
    forward_pass.cpp
    lighting_pass.cpp
    skybox_pass.cpp
    tonemap_pass.cpp
    taa_pass.cpp
    smaa_pass.cpp
    imgui_pass.cpp
    frame_renderer.cpp
)
target_include_directories(somegi_renderer_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_core PUBLIC somegi_core somegi_scene somegi_gi imgui)
