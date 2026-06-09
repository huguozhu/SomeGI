add_library(somegi_renderer_lumen OBJECT
    lumen_resources.cpp
    lumen_probe_pass.cpp
    lumen_gather_pass.cpp
    lumen_filter_pass.cpp
)
target_include_directories(somegi_renderer_lumen PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_lumen PUBLIC somegi_core)
