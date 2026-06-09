add_library(somegi_renderer_rsm OBJECT
    rsm_geometry_pass.cpp
    rsm_sample_pass.cpp
)
target_include_directories(somegi_renderer_rsm PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_rsm PUBLIC somegi_core)
